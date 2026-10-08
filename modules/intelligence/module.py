"""
MODULE 6 -- CENTRAL INTELLIGENCE / EVENT ENGINE   (owner: Person 6)
===================================================================

Plain Python, no Nano-specific anything, as the spec asks for this phase.

It sits at the end of the pipeline and does seven things to the event
stream:

1. **Normalisation**       -- fill road_segment, clamp severity, canonicalise
                              class names, guarantee every field the backend
                              expects exists.
2. **Deduplication**       -- the same pothole seen on 40 frames of one pass
                              is one observation, not 40 issues.
3. **Multi-bus correlation** -- BUS-1, BUS-2 and BUS-3 reporting a pothole at
                              GPS A all attach to ``POT-001``.
4. **Persistent issues**   -- first_seen / last_seen / observations / buses.
5. **Repair verification** -- a bus passing the segment without seeing the
                              defect is a "clean pass"; enough clean passes
                              close the issue; a later sighting REOPENS it.
6. **Priority scoring**    -- severity + persistence + traffic impact +
                              safety risk.
7. **Department routing**  -- pothole -> Roads/PWD, waterlogging -> Drainage,
                              incidents -> Traffic Police, ...

State lives in the SQLite database, so correlation works **across pipeline
runs and across buses** -- run bus 1's video, then bus 2's, and the second
run attaches to the issues the first one opened.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from core.event import Event, EventType, Severity, SourceModule, make_event
from core.geo import haversine_m, road_segment_id
from core.logging_setup import get_logger

from .priority import compute_priority
from .routing import route_department

#: class/event -> issue id prefix
ISSUE_PREFIX: Dict[str, str] = {
    "pothole": "POT",
    "damaged_road": "DMG",
    "road_damage_other": "RDO",
    "waterlogging": "WTR",
    "missing_divider": "DIV",
    "missing_zebra_crossing": "ZBR",
    "damaged_zebra_crossing": "ZBR",
    "zebra_crossing": "INF",
    "divider": "INF",
    "traffic_signboard": "INF",
    "road_repair": "REP",
}

EVENT_PREFIX: Dict[str, str] = {
    "traffic_congestion": "CON",
    "traffic_bottleneck": "BTL",
    "pedestrian_vehicle_risk": "PED",
    "rash_driving": "RSH",
    "collision": "COL",
    "possible_collision": "PCL",
    "abnormal_movement": "ABN",
    "hit_and_run": "HNR",
    "road_defect": "RDF",
}

#: Event types that never become a persistent issue.
NON_ISSUE_TYPES = {
    EventType.TRAFFIC_OBSERVATION,
    EventType.VEHICLE_IDENTIFICATION,
    EventType.PERSISTENT_ISSUE,
}

#: Per-event-type dedup window (seconds).  A pothole seen an hour later is
#: the same pothole; two collisions a minute apart are two collisions.
DEDUP_WINDOW: Dict[str, float] = {
    "road_defect": 86_400.0 * 365,
    "traffic_congestion": 900.0,
    "traffic_bottleneck": 900.0,
    "pedestrian_vehicle_risk": 30.0,
    "rash_driving": 60.0,
    "collision": 60.0,
    "possible_collision": 60.0,
    "abnormal_movement": 60.0,
    "hit_and_run": 300.0,
}

#: Road-defect classes that indicate healthy infrastructure rather than a
#: fault; they still get recorded but are never given a priority band above
#: LOW and are used for repair verification.
INFRASTRUCTURE_OK = {"zebra_crossing", "divider", "traffic_signboard", "road_repair"}


class IntelligenceEngine:
    """Not a per-frame module -- it consumes the event stream."""

    name = "intelligence"
    owner = "Person 6"

    def __init__(self, config, database=None):
        self.platform_config = config
        self.config = config.section("modules.intelligence")
        self.log = get_logger("module.intelligence")
        self.enabled = bool(self.config.get("enabled", True))
        self.db = database

        dedup = self.config.get("dedup", {}) or {}
        self.gps_radius_m = float(dedup.get("gps_radius_m", 35.0))
        self.same_bus_window_s = float(dedup.get("same_bus_window_s", 60.0))
        self.default_window_s = float(dedup.get("time_window_s", 300.0))

        persistence = self.config.get("persistence", {}) or {}
        self.resolve_after_clean_passes = int(
            persistence.get("resolve_after_clean_passes", 2)
        )

        self.segment_size_m = float(config.get("gps.segment_size_m", 60.0))
        self.department_overrides = config.section("departments")
        self.weights = (self.config.get("priority", {}) or {}).get("weights")

        # in-memory index: (prefix-key) -> issue dict
        self._issues: Dict[str, Dict[str, Any]] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        # (bus_id, segment) currently being traversed, and what was seen there
        self._current_pass: Dict[str, Tuple[Optional[str], set]] = {}
        self._segment_cache: Dict[str, Dict[str, Any]] = {}

        self.stats_counters: Dict[str, int] = defaultdict(int)
        self._loaded = False

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        """Load existing issues so correlation spans runs and buses."""
        if self.db is None:
            self.log.warning(
                "no database -- issues will only live for this run "
                "(multi-run correlation needs backend.database)"
            )
            self._loaded = True
            return
        try:
            for issue in self.db.all_issues():
                self._issues[issue["issue_id"]] = issue
                match = re.match(r"([A-Z]+)-(\d+)", issue["issue_id"] or "")
                if match:
                    prefix, number = match.group(1), int(match.group(2))
                    self._counters[prefix] = max(self._counters[prefix], number)
            for segment in self.db.list_segments():
                self._segment_cache[segment["road_segment"]] = segment
        except Exception as exc:                                 # noqa: BLE001
            self.log.warning("could not preload issues: %s", exc)
        self._loaded = True
        self.log.info(
            "intelligence engine ready (%s existing issues, %s segments known)",
            len(self._issues), len(self._segment_cache),
        )

    # -- main entry point ---------------------------------------------------

    def process_events(self, events: List[Event]) -> List[Event]:
        """Enrich each event in place and return any NEW events the engine
        itself produced (issue opened / reopened / resolved)."""
        if not self.enabled or not events:
            return []

        generated: List[Event] = []
        for event in events:
            try:
                generated.extend(self._process_one(event))
            except Exception as exc:                             # noqa: BLE001
                self.log.error(
                    "failed on %s (%s): %s", event.event_type, event.event_id, exc
                )
        return generated

    # -- 1. normalisation ---------------------------------------------------

    def _normalise(self, event: Event) -> None:
        if not event.road_segment and event.gps is not None:
            event.road_segment = road_segment_id(event.gps, self.segment_size_m)
        if event.severity is None:
            event.severity = Severity.INFO
        event.severity = str(event.severity).upper()
        if event.confidence is None:
            event.confidence = 0.0
        event.confidence = round(float(event.confidence), 4)
        event.metadata.setdefault("normalised", True)
        self.stats_counters["normalised"] += 1

    # -- main per-event flow ------------------------------------------------

    def _process_one(self, event: Event) -> List[Event]:
        self._normalise(event)

        # Route + priority apply to every event, issue or not.
        event.department = route_department(
            event.event_type, event.class_name, self.department_overrides
        )

        generated: List[Event] = []

        # A traffic observation tells us the bus is on a segment -- that is
        # what drives repair verification.
        if event.event_type == EventType.TRAFFIC_OBSERVATION:
            generated.extend(self._track_pass(event))
            self._score(event, issue=None)
            return generated

        if event.event_type in NON_ISSUE_TYPES:
            self._score(event, issue=None)
            return generated

        generated.extend(self._track_pass(event, seen_class=event.class_name))

        issue, is_new, reopened = self._attach_to_issue(event)
        if issue is None:
            self._score(event, issue=None)
            return generated

        event.issue_id = issue["issue_id"]
        self._score(event, issue=issue)
        issue["priority_score"] = event.priority_score
        issue["priority_band"] = event.metadata.get("priority_band")
        issue["department"] = event.department
        self._persist_issue(issue, event)

        if is_new:
            self.stats_counters["issues_opened"] += 1
            generated.append(self._issue_event(issue, event, "OPENED"))
        elif reopened:
            self.stats_counters["issues_reopened"] += 1
            generated.append(self._issue_event(issue, event, "REOPENED"))
        else:
            self.stats_counters["duplicates_merged"] += 1
        return generated

    # -- 2/3/4. dedup, multi-bus correlation, persistence --------------------

    def _issue_prefix(self, event: Event) -> str:
        if event.class_name and event.class_name in ISSUE_PREFIX:
            return ISSUE_PREFIX[event.class_name]
        return EVENT_PREFIX.get(event.event_type, "EVT")

    def _match_existing(self, event: Event) -> Optional[Dict[str, Any]]:
        """Find an open issue that this event is another sighting of.

        Matching is by (issue class) AND (same road segment OR within
        ``gps_radius_m``) AND within the class's dedup time window.
        """
        prefix = self._issue_prefix(event)
        window = DEDUP_WINDOW.get(event.event_type, self.default_window_s)
        now = event.epoch

        best = None
        best_distance = float("inf")
        for issue in self._issues.values():
            if issue.get("issue_type") != event.event_type:
                continue
            if (issue.get("class_name") or "") != (event.class_name or ""):
                continue
            if not str(issue.get("issue_id", "")).startswith(prefix):
                continue

            last_seen = float(issue.get("last_seen_epoch") or 0.0)
            if window and last_seen and (now - last_seen) > window:
                continue

            same_segment = (
                event.road_segment
                and issue.get("road_segment") == event.road_segment
            )
            distance = float("inf")
            if event.gps is not None and issue.get("latitude") is not None:
                distance = haversine_m(
                    event.gps.latitude, event.gps.longitude,
                    issue["latitude"], issue["longitude"],
                )
            if same_segment or distance <= self.gps_radius_m:
                if distance < best_distance:
                    best, best_distance = issue, distance
        return best

    def _attach_to_issue(self, event: Event) -> Tuple[Optional[Dict], bool, bool]:
        issue = self._match_existing(event)
        reopened = False

        if issue is None:
            prefix = self._issue_prefix(event)
            self._counters[prefix] += 1
            issue_id = f"{prefix}-{self._counters[prefix]:03d}"
            issue = {
                "issue_id": issue_id,
                "issue_type": event.event_type,
                "class_name": event.class_name,
                "status": "OPEN",
                "severity": event.severity,
                "first_seen": event.timestamp,
                "last_seen": event.timestamp,
                "first_seen_epoch": event.epoch,
                "last_seen_epoch": event.epoch,
                "observation_count": 0,
                "bus_count": 0,
                "buses": [],
                "latitude": event.gps.latitude if event.gps else None,
                "longitude": event.gps.longitude if event.gps else None,
                "road_segment": event.road_segment,
                "priority_score": 0.0,
                "priority_band": "LOW",
                "department": event.department,
                "clean_passes": 0,
                "reopened_count": 0,
                "repair_observed": 0,
                "resolved_at": None,
                "last_event_id": event.event_id,
                "metadata": {},
            }
            self._issues[issue_id] = issue
            is_new = True
        else:
            is_new = False
            if issue.get("status") == "RESOLVED":
                # Repair verification: it is back.
                issue["status"] = "REOPENED"
                issue["reopened_count"] = int(issue.get("reopened_count", 0)) + 1
                issue["clean_passes"] = 0
                issue["resolved_at"] = None
                reopened = True
                self.log.info(
                    "%s REOPENED -- detected again after being marked resolved",
                    issue["issue_id"],
                )
            event.duplicate_of = issue.get("last_event_id")

        # --- multi-bus correlation ---
        buses = list(issue.get("buses") or [])
        if event.bus_id and event.bus_id not in buses:
            buses.append(event.bus_id)
        issue["buses"] = buses
        issue["bus_count"] = len(buses)

        issue["observation_count"] = int(issue.get("observation_count", 0)) + 1
        issue["last_seen"] = event.timestamp
        issue["last_seen_epoch"] = event.epoch
        issue["last_event_id"] = event.event_id
        issue["severity"] = Severity.max(issue.get("severity"), event.severity)
        if issue.get("latitude") is None and event.gps is not None:
            issue["latitude"] = event.gps.latitude
            issue["longitude"] = event.gps.longitude
        if not issue.get("road_segment"):
            issue["road_segment"] = event.road_segment
        return issue, is_new, reopened

    # -- 5. repair verification --------------------------------------------

    def _track_pass(self, event: Event, seen_class: Optional[str] = None) -> List[Event]:
        """Remember what each bus saw on each segment it traversed.

        When a bus leaves a segment, any open defect issue on that segment
        that the bus did NOT report counts as a clean pass.  Enough clean
        passes and the issue is marked RESOLVED.
        """
        bus = event.bus_id or "UNKNOWN"
        segment = event.road_segment
        if not segment:
            return []

        previous_segment, seen = self._current_pass.get(bus, (None, set()))
        generated: List[Event] = []

        if previous_segment and previous_segment != segment:
            generated.extend(self._close_pass(bus, previous_segment, seen))
            seen = set()

        if seen_class:
            seen.add(seen_class)
        self._current_pass[bus] = (segment, seen)
        return generated

    def _close_pass(self, bus: str, segment: str, seen: set) -> List[Event]:
        generated: List[Event] = []
        repair_seen = "road_repair" in seen

        for issue in list(self._issues.values()):
            if issue.get("road_segment") != segment:
                continue
            if issue.get("issue_type") != EventType.ROAD_DEFECT:
                continue
            if issue.get("class_name") in INFRASTRUCTURE_OK:
                continue
            if issue.get("status") not in ("OPEN", "REOPENED"):
                continue

            if repair_seen and not issue.get("repair_observed"):
                issue["repair_observed"] = 1
                self.log.info(
                    "%s -- repair work observed on %s", issue["issue_id"], segment
                )

            if issue.get("class_name") in seen:
                issue["clean_passes"] = 0
                continue

            issue["clean_passes"] = int(issue.get("clean_passes", 0)) + 1
            if issue["clean_passes"] >= self.resolve_after_clean_passes:
                issue["status"] = "RESOLVED"
                issue["resolved_at"] = _now_iso()
                self.stats_counters["issues_resolved"] += 1
                self.log.info(
                    "%s RESOLVED -- %s clean passes of %s with no detection",
                    issue["issue_id"], issue["clean_passes"], segment,
                )
                generated.append(self._issue_event(issue, None, "RESOLVED"))
            self._save_issue(issue)
        return generated

    def flush_passes(self) -> List[Event]:
        """Called at end of run so the last segment traversal still counts."""
        generated: List[Event] = []
        for bus, (segment, seen) in list(self._current_pass.items()):
            if segment:
                generated.extend(self._close_pass(bus, segment, seen))
        self._current_pass.clear()
        return generated

    # -- 6. priority --------------------------------------------------------

    def _segment_stats(self, segment: Optional[str]) -> Dict[str, Any]:
        if not segment:
            return {}
        if segment in self._segment_cache:
            return self._segment_cache[segment]
        if self.db is not None:
            for row in self.db.list_segments():
                self._segment_cache[row["road_segment"]] = row
        return self._segment_cache.get(segment, {})

    def _score(self, event: Event, issue: Optional[Dict[str, Any]]) -> None:
        segment_stats = self._segment_stats(event.road_segment)
        observation_count = int(issue.get("observation_count", 1)) if issue else 1
        bus_count = int(issue.get("bus_count", 1)) if issue else 1
        age_days = 0.0
        if issue and issue.get("first_seen_epoch"):
            age_days = max(
                0.0,
                (event.epoch - float(issue["first_seen_epoch"])) / 86_400.0,
            )

        result = compute_priority(
            event_type=event.event_type,
            class_name=event.class_name,
            severity=event.severity,
            observation_count=observation_count,
            bus_count=bus_count,
            age_days=age_days,
            segment_stats=segment_stats,
            incident_risk_at_segment=int(
                segment_stats.get("incident_count") or 0
            ),
            weights=self.weights,
        )
        if event.class_name in INFRASTRUCTURE_OK:
            result["priority_score"] = min(result["priority_score"], 20.0)
            result["priority_band"] = "LOW"

        event.priority_score = result["priority_score"]
        event.metadata["priority_band"] = result["priority_band"]
        event.metadata["priority_components"] = result["components"]

    # -- persistence --------------------------------------------------------

    def _persist_issue(self, issue: Dict[str, Any], event: Event) -> None:
        self._save_issue(issue)
        if self.db is not None:
            try:
                self.db.add_issue_observation(issue["issue_id"], event)
            except Exception as exc:                             # noqa: BLE001
                self.log.debug("could not store observation: %s", exc)

    def _save_issue(self, issue: Dict[str, Any]) -> None:
        if self.db is None:
            return
        try:
            self.db.upsert_issue(dict(issue))
        except Exception as exc:                                 # noqa: BLE001
            self.log.warning("could not persist issue %s: %s", issue.get("issue_id"), exc)

    # -- issue lifecycle events --------------------------------------------

    def _issue_event(self, issue: Dict[str, Any], event: Optional[Event], state: str) -> Event:
        return make_event(
            EventType.PERSISTENT_ISSUE,
            SourceModule.INTELLIGENCE,
            confidence=float(event.confidence) if event else 0.9,
            severity=issue.get("severity", Severity.INFO),
            class_name=issue.get("class_name"),
            bus_id=event.bus_id if event else None,
            camera_id=event.camera_id if event else None,
            gps=(
                {"latitude": issue.get("latitude"), "longitude": issue.get("longitude")}
                if issue.get("latitude") is not None else None
            ),
            road_segment=issue.get("road_segment"),
            evidence_image=event.evidence_image if event else None,
            metadata={
                "issue_state": state,
                "issue_id": issue["issue_id"],
                "issue_type": issue.get("issue_type"),
                "first_seen": issue.get("first_seen"),
                "last_seen": issue.get("last_seen"),
                "observations": issue.get("observation_count"),
                "buses": issue.get("buses"),
                "bus_count": issue.get("bus_count"),
                "clean_passes": issue.get("clean_passes"),
                "reopened_count": issue.get("reopened_count"),
                "repair_observed": bool(issue.get("repair_observed")),
                "department": issue.get("department"),
                "priority_score": issue.get("priority_score"),
                "priority_band": issue.get("priority_band"),
                "triggering_event_id": event.event_id if event else None,
            },
            issue_id=issue["issue_id"],
            priority_score=issue.get("priority_score"),
            department=issue.get("department"),
        )

    # -- reporting ----------------------------------------------------------

    def issues(self) -> List[Dict[str, Any]]:
        return sorted(
            self._issues.values(),
            key=lambda i: (i.get("priority_score") or 0.0),
            reverse=True,
        )

    def stats(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = defaultdict(int)
        for issue in self._issues.values():
            by_status[issue.get("status", "OPEN")] += 1
        return {
            "module": self.name,
            "owner": self.owner,
            "status": "READY" if self.enabled else "DISABLED",
            "issues_total": len(self._issues),
            "issues_by_status": dict(by_status),
            **dict(self.stats_counters),
        }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
