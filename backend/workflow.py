"""
backend.workflow
================

The shared event lifecycle used by Central Command, the department
dashboards and the mobile app.

    DETECTED -> VERIFIED -> ASSIGNED -> IN_PROGRESS -> RESOLVED

What a "managed event" (``EV-1024``) is
---------------------------------------
The pipeline already deduplicates: the intelligence engine folds every
sighting of the same pothole (same class, same place, inside the class's
time window, any bus) into ONE persistent issue (``POT-001``).  A managed
event is created from that issue, so four pothole frames produce one
``EV-`` record, not four.

On top of the engine, short-lived event types (pedestrian risk, abnormal
movement, rash driving, collisions, congestion) get one more grouping rule:
repeated issues of the same type from the same bus and camera, in the same
recording, within ``workflow.group_window_s`` seconds of *video* time and
``workflow.group_radius_m`` metres are one managed event.  The engine's own
windows are measured in wall-clock time, which on a slow CPU replay covers
only a fraction of a second of video -- that is what used to turn one
pedestrian situation into a dozen issues.

The lifecycle lives in this module's own tables (``cases``,
``case_issues``, ``case_activity``, ``notifications``,
``notification_reads``).  The pipeline's tables are never written here, so a
new pipeline run can never overwrite a human decision.  The two are bridged
in one direction by :meth:`WorkflowStore.sync`:

* a new issue becomes a DETECTED event (or joins an existing one),
* new sightings update observation counts and buses,
* a sighting AFTER an event was marked RESOLVED re-opens it,
* the engine's own repair verification (clean passes) is logged as a note.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import HTTPException

from backend import rbac

STATUSES = ["DETECTED", "VERIFIED", "ASSIGNED", "IN_PROGRESS", "RESOLVED"]
STATUS_INDEX = {s: i for i, s in enumerate(STATUSES)}

ACTIONS: Dict[str, Tuple[str, str]] = {
    "verify": ("DETECTED", "VERIFIED"),
    "assign": ("VERIFIED", "ASSIGNED"),
    "start": ("ASSIGNED", "IN_PROGRESS"),
    "resolve": ("IN_PROGRESS", "RESOLVED"),
}
ACTION_LABEL = {
    "verify": "Verify", "assign": "Assign",
    "start": "Start work", "resolve": "Mark resolved",
}
ACTION_ORDER = ["verify", "assign", "start", "resolve"]

SEVERITY_RANK = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

CLASS_TITLE = {
    "pothole": "Pothole",
    "damaged_road": "Damaged road",
    "road_damage_other": "Road damage",
    "waterlogging": "Waterlogging",
    "missing_divider": "Missing divider",
    "missing_zebra_crossing": "Missing zebra crossing",
    "damaged_zebra_crossing": "Damaged zebra crossing",
    "missing_traffic_signboard": "Missing traffic sign",
    "damaged_traffic_signboard": "Damaged traffic sign",
}
TYPE_TITLE = {
    "traffic_congestion": "Traffic congestion",
    "traffic_bottleneck": "Traffic bottleneck",
    "rash_driving": "Rash driving",
    "collision": "Collision",
    "possible_collision": "Possible collision",
    "abnormal_movement": "Abnormal movement",
    "hit_and_run": "Hit-and-run",
    "pedestrian_vehicle_risk": "Pedestrian risk",
}
CATEGORY = {
    rbac.ROADS_INFRASTRUCTURE: "Road issue",
    rbac.TRAFFIC_DEPARTMENT: "Traffic issue",
    rbac.PUBLIC_SAFETY: "Safety incident",
    rbac.TRANSPORT_AUTHORITY: "Transport issue",
}

WORKFLOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id              TEXT PRIMARY KEY,
    case_no              INTEGER UNIQUE,
    event_type           TEXT NOT NULL,
    class_name           TEXT,
    title                TEXT,
    owner_department     TEXT NOT NULL,
    status               TEXT NOT NULL,
    severity             TEXT,
    priority_score       REAL,
    priority_band        TEXT,
    confidence           REAL,
    bus_id               TEXT,
    camera_id            TEXT,
    buses                TEXT,
    bus_count            INTEGER DEFAULT 1,
    observation_count    INTEGER DEFAULT 1,
    latitude             REAL,
    longitude            REAL,
    road_segment         TEXT,
    primary_event_id     TEXT,
    primary_issue_id     TEXT,
    recording_name       TEXT,
    last_video_time_s    REAL,
    first_seen           TEXT,
    last_seen            TEXT,
    first_seen_epoch     REAL,
    last_seen_epoch      REAL,
    assignee_user_id     TEXT,
    assigned_by          TEXT,
    assigned_at          TEXT,
    verified_by          TEXT,
    verified_at          TEXT,
    started_at           TEXT,
    resolved_at          TEXT,
    resolved_by          TEXT,
    resolution_note      TEXT,
    observations_at_resolve INTEGER,
    reopened_count       INTEGER DEFAULT 0,
    created_at           TEXT,
    updated_at           TEXT,
    version              INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_cases_owner   ON cases(owner_department, status);
CREATE INDEX IF NOT EXISTS idx_cases_type    ON cases(event_type, class_name);
CREATE INDEX IF NOT EXISTS idx_cases_assign  ON cases(assignee_user_id);

CREATE TABLE IF NOT EXISTS case_issues (
    issue_id            TEXT PRIMARY KEY,
    case_id             TEXT NOT NULL,
    observations_seen   INTEGER DEFAULT 0,
    status_seen         TEXT,
    buses_seen          TEXT,
    linked_at           TEXT
);
CREATE INDEX IF NOT EXISTS idx_case_issues_case ON case_issues(case_id);

CREATE TABLE IF NOT EXISTS case_activity (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id          TEXT NOT NULL,
    at               TEXT,
    at_epoch         REAL,
    actor_type       TEXT,
    actor_id         TEXT,
    actor_name       TEXT,
    actor_department TEXT,
    action           TEXT,
    from_status      TEXT,
    to_status        TEXT,
    note             TEXT
);
CREATE INDEX IF NOT EXISTS idx_activity_case ON case_activity(case_id, id);

CREATE TABLE IF NOT EXISTS notifications (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT,
    created_epoch REAL,
    department    TEXT,
    user_id       TEXT,
    severity      TEXT,
    kind          TEXT,
    title         TEXT,
    body          TEXT,
    case_id       TEXT
);
CREATE INDEX IF NOT EXISTS idx_notif_dept ON notifications(department, id);
CREATE INDEX IF NOT EXISTS idx_notif_user ON notifications(user_id, id);

CREATE TABLE IF NOT EXISTS notification_reads (
    notification_id INTEGER NOT NULL,
    user_id         TEXT NOT NULL,
    read_at         REAL,
    PRIMARY KEY (notification_id, user_id)
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _json_list(value) -> List[Any]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        out = json.loads(value)
        return out if isinstance(out, list) else []
    except Exception:                                            # noqa: BLE001
        return []


def case_title(event_type: str, class_name: Optional[str]) -> str:
    if event_type == "road_defect":
        return CLASS_TITLE.get(class_name or "", (class_name or "Road defect").replace("_", " ").capitalize())
    return TYPE_TITLE.get(event_type, event_type.replace("_", " ").capitalize())


class WorkflowStore:
    def __init__(self, database, config, recordings=None, auth=None, log=None):
        self.db = database
        self.config = config
        self.recordings = recordings
        self.auth = auth
        self.log = log
        section = config.section("workflow")
        self.group_window_s = float(section.get("group_window_s", 20.0))
        self.group_radius_m = float(section.get("group_radius_m", 60.0))
        self.min_sync_interval_s = float(section.get("min_sync_interval_s", 2.0))
        self.first_case_no = int(section.get("first_case_number", 1001))
        self._sync_lock = threading.Lock()
        self._last_sync = 0.0

    def initialise(self) -> "WorkflowStore":
        self.db.executescript(WORKFLOW_SCHEMA)
        return self

    # =================================================================
    # sync: pipeline issues -> managed events
    # =================================================================

    def sync(self, force: bool = False) -> Dict[str, int]:
        if not force and time.time() - self._last_sync < self.min_sync_interval_s:
            return {}
        if not self._sync_lock.acquire(blocking=force):
            return {}
        stats = {"created": 0, "merged": 0, "updated": 0, "reopened": 0}
        try:
            issues = self.db.query("SELECT * FROM issues ORDER BY first_seen_epoch ASC, issue_id ASC")
            links = {row["issue_id"]: row for row in self.db.query("SELECT * FROM case_issues")}
            for issue in issues:
                owner = rbac.owner_department(issue.get("issue_type"), issue.get("class_name"))
                if owner is None:
                    continue
                try:
                    if issue["issue_id"] in links:
                        outcome = self._refresh_link(links[issue["issue_id"]], issue)
                    else:
                        outcome = self._attach_issue(issue, owner)
                    if outcome:
                        stats[outcome] = stats.get(outcome, 0) + 1
                except Exception as exc:                         # noqa: BLE001
                    if self.log:
                        self.log.warning("workflow sync failed on %s: %s", issue.get("issue_id"), exc)
            self._last_sync = time.time()
        finally:
            self._sync_lock.release()
        if self.log and (stats["created"] or stats["reopened"]):
            self.log.info("workflow sync: %s", stats)
        return stats

    # -- new issue ----------------------------------------------------------

    def _issue_events(self, issue_id: str) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT event_id, event_type, bus_id, camera_id, timestamp, ts_epoch, latitude, "
            "longitude, confidence, severity, track_id, class_name, bounding_box, "
            "evidence_image, frame_number, video_time_s, road_segment, run_id, metadata "
            "FROM events WHERE issue_id = ? AND event_type != 'persistent_issue' "
            "ORDER BY ts_epoch ASC",
            (issue_id,),
        )
        for row in rows:
            row["bounding_box"] = _json_list(row.get("bounding_box")) or None
            try:
                row["metadata"] = json.loads(row["metadata"]) if row.get("metadata") else {}
            except Exception:                                    # noqa: BLE001
                row["metadata"] = {}
        return rows

    def _recording_of(self, event: Dict[str, Any]) -> Optional[str]:
        if self.recordings is None:
            return None
        try:
            link = self.recordings.resolve(event)
        except Exception:                                        # noqa: BLE001
            return None
        return link["name"] if link else None

    def _attach_issue(self, issue: Dict[str, Any], owner: str) -> Optional[str]:
        events = self._issue_events(issue["issue_id"])
        if not events:
            return None          # the issue row lands a moment before its events
        first = events[0]
        recording = self._recording_of(first)
        target = self._group_target(issue, first, recording)
        now = _now_iso()
        if target is not None:
            with self.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO case_issues(issue_id, case_id, observations_seen, status_seen, "
                    "buses_seen, linked_at) VALUES (?,?,?,?,?,?)",
                    (issue["issue_id"], target["case_id"], issue.get("observation_count") or 0,
                     issue.get("status"), json.dumps(_json_list(issue.get("buses"))), now),
                )
            self._recompute(target["case_id"], latest_event=events[-1], recording=recording)
            return "merged"

        best = max(events, key=lambda e: ((1 if e.get("evidence_image") else 0), e.get("confidence") or 0))
        row = self.db.query_one("SELECT MAX(case_no) AS n FROM cases")
        case_no = max(self.first_case_no, (row["n"] or self.first_case_no - 1) + 1)
        case_id = f"EV-{case_no}"
        title = case_title(issue["issue_type"], issue.get("class_name"))
        severity = issue.get("severity") or first.get("severity") or "INFO"
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO cases(case_id, case_no, event_type, class_name, title,
                   owner_department, status, severity, priority_score, priority_band,
                   confidence, bus_id, camera_id, buses, bus_count, observation_count,
                   latitude, longitude, road_segment, primary_event_id, primary_issue_id,
                   recording_name, last_video_time_s, first_seen, last_seen,
                   first_seen_epoch, last_seen_epoch, created_at, updated_at, version)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                (
                    case_id, case_no, issue["issue_type"], issue.get("class_name"), title,
                    owner, "DETECTED", severity, issue.get("priority_score"),
                    issue.get("priority_band"), best.get("confidence"),
                    first.get("bus_id"), first.get("camera_id"),
                    json.dumps(_json_list(issue.get("buses")) or [first.get("bus_id")]),
                    issue.get("bus_count") or 1, issue.get("observation_count") or len(events),
                    issue.get("latitude") if issue.get("latitude") is not None else first.get("latitude"),
                    issue.get("longitude") if issue.get("longitude") is not None else first.get("longitude"),
                    issue.get("road_segment") or first.get("road_segment"),
                    best["event_id"], issue["issue_id"], recording,
                    events[-1].get("video_time_s"),
                    issue.get("first_seen") or first.get("timestamp"),
                    issue.get("last_seen") or events[-1].get("timestamp"),
                    issue.get("first_seen_epoch") or first.get("ts_epoch"),
                    issue.get("last_seen_epoch") or events[-1].get("ts_epoch"),
                    now, now,
                ),
            )
            conn.execute(
                "INSERT INTO case_issues(issue_id, case_id, observations_seen, status_seen, "
                "buses_seen, linked_at) VALUES (?,?,?,?,?,?)",
                (issue["issue_id"], case_id, issue.get("observation_count") or 0,
                 issue.get("status"), json.dumps(_json_list(issue.get("buses"))), now),
            )
            confidence = best.get("confidence")
            note = f"Detected by {first.get('bus_id') or 'bus'} / {first.get('camera_id') or 'camera'}"
            if confidence is not None:
                note += f" at {round(float(confidence) * 100)}% confidence"
            self._activity(
                conn, case_id, at=first.get("timestamp") or now, at_epoch=first.get("ts_epoch"),
                actor_type="bus", actor_id=first.get("bus_id"), actor_name=first.get("bus_id") or "Bus",
                actor_department=None, action="DETECTED", from_status=None,
                to_status="DETECTED", note=note,
            )
            self._notify_new_case(conn, case_id, issue, first, owner, title, severity)
        return "created"

    def _group_target(self, issue, first_event, recording) -> Optional[Dict[str, Any]]:
        if issue["issue_type"] == "road_defect" or self.group_window_s <= 0:
            return None
        rows = self.db.query(
            "SELECT * FROM cases WHERE event_type = ? AND IFNULL(class_name,'') = ? "
            "AND IFNULL(bus_id,'') = ? AND IFNULL(camera_id,'') = ? AND status != 'RESOLVED' "
            "ORDER BY last_seen_epoch DESC LIMIT 20",
            (issue["issue_type"], issue.get("class_name") or "",
             first_event.get("bus_id") or "", first_event.get("camera_id") or ""),
        )
        for case in rows:
            if recording and case.get("recording_name"):
                if case["recording_name"] != recording:
                    continue
                t_new, t_old = first_event.get("video_time_s"), case.get("last_video_time_s")
                if t_new is None or t_old is None or abs(float(t_new) - float(t_old)) > self.group_window_s:
                    continue
            else:
                e_new, e_old = first_event.get("ts_epoch"), case.get("last_seen_epoch")
                if e_new is None or e_old is None or abs(float(e_new) - float(e_old)) > self.group_window_s:
                    continue
            lat, lon = first_event.get("latitude"), first_event.get("longitude")
            if lat is not None and case.get("latitude") is not None:
                if _haversine_m(lat, lon, case["latitude"], case["longitude"]) > self.group_radius_m:
                    continue
            return case
        return None

    # -- existing issue -----------------------------------------------------

    def _refresh_link(self, link: Dict[str, Any], issue: Dict[str, Any]) -> Optional[str]:
        observations = int(issue.get("observation_count") or 0)
        seen = int(link.get("observations_seen") or 0)
        status_changed = (issue.get("status") or "") != (link.get("status_seen") or "")
        if observations <= seen and not status_changed:
            return None
        case = self.get_case(link["case_id"])
        if case is None:
            return None
        outcome = "updated"
        old_buses = set(_json_list(link.get("buses_seen")))
        new_buses = [b for b in _json_list(issue.get("buses")) if b not in old_buses]
        latest = None
        if observations > seen:
            events = self._issue_events(issue["issue_id"])
            latest = events[-1] if events else None
        now = _now_iso()
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE case_issues SET observations_seen = ?, status_seen = ?, buses_seen = ? "
                "WHERE issue_id = ?",
                (observations, issue.get("status"), json.dumps(_json_list(issue.get("buses"))),
                 issue["issue_id"]),
            )
            if new_buses and case["status"] != "RESOLVED":
                self._activity(
                    conn, case["case_id"], at=issue.get("last_seen") or now,
                    at_epoch=issue.get("last_seen_epoch"), actor_type="bus",
                    actor_id=new_buses[0], actor_name=", ".join(new_buses),
                    actor_department=None, action="OBSERVED", from_status=case["status"],
                    to_status=case["status"],
                    note=f"Also reported by {', '.join(new_buses)}",
                )
            if (observations > seen and case["status"] == "RESOLVED"
                    and observations > int(case.get("observations_at_resolve") or 0)):
                by = (latest or {}).get("bus_id") or (new_buses[0] if new_buses else "a bus")
                conn.execute(
                    "UPDATE cases SET status = 'DETECTED', reopened_count = reopened_count + 1, "
                    "verified_by = NULL, verified_at = NULL, started_at = NULL, "
                    "updated_at = ?, version = version + 1 WHERE case_id = ?",
                    (now, case["case_id"]),
                )
                self._activity(
                    conn, case["case_id"], at=issue.get("last_seen") or now,
                    at_epoch=issue.get("last_seen_epoch"), actor_type="bus", actor_id=by,
                    actor_name=by, actor_department=None, action="REOPENED",
                    from_status="RESOLVED", to_status="DETECTED",
                    note=f"Detected again by {by} after it was marked resolved",
                )
                self._notify(
                    conn, department=case["owner_department"], severity="HIGH", kind="reopened",
                    title=f"{case['title']} is back",
                    body=f"{case['case_id']} was resolved but {by} detected it again",
                    case_id=case["case_id"],
                )
                if case.get("assignee_user_id"):
                    self._notify(
                        conn, user_id=case["assignee_user_id"], severity="HIGH", kind="reopened",
                        title=f"{case['case_id']} re-opened",
                        body=f"{case['title']} was detected again by {by}", case_id=case["case_id"],
                    )
                outcome = "reopened"
            if status_changed and issue.get("status") == "RESOLVED" and case["status"] != "RESOLVED":
                passes = issue.get("clean_passes") or 0
                self._activity(
                    conn, case["case_id"], at=issue.get("resolved_at") or now, at_epoch=None,
                    actor_type="system", actor_id="intelligence", actor_name="Repair verification",
                    actor_department=None, action="NOTE", from_status=case["status"],
                    to_status=case["status"],
                    note=f"Buses passed {passes} time(s) without detecting it -- it may already be repaired",
                )
                self._notify(
                    conn, department=case["owner_department"], severity="INFO", kind="repair_seen",
                    title=f"{case['case_id']} no longer detected",
                    body=f"{case['title']}: {passes} clean pass(es) by buses", case_id=case["case_id"],
                )
        self._recompute(case["case_id"], latest_event=latest)
        return outcome

    def _recompute(self, case_id: str, latest_event=None, recording=None) -> None:
        issues = self.db.query(
            "SELECT i.* FROM issues i JOIN case_issues c ON c.issue_id = i.issue_id WHERE c.case_id = ?",
            (case_id,),
        )
        if not issues:
            return
        buses: List[str] = []
        for issue in issues:
            for bus in _json_list(issue.get("buses")):
                if bus not in buses:
                    buses.append(bus)
        severity = max((i.get("severity") or "INFO" for i in issues), key=lambda s: SEVERITY_RANK.get(s, 0))
        best_priority = max(issues, key=lambda i: i.get("priority_score") or 0)
        last = max(issues, key=lambda i: i.get("last_seen_epoch") or 0)
        fields = {
            "observation_count": sum(int(i.get("observation_count") or 0) for i in issues),
            "buses": json.dumps(buses), "bus_count": len(buses) or 1,
            "severity": severity,
            "priority_score": best_priority.get("priority_score"),
            "priority_band": best_priority.get("priority_band"),
            "last_seen": last.get("last_seen"), "last_seen_epoch": last.get("last_seen_epoch"),
        }
        if latest_event is not None and latest_event.get("video_time_s") is not None:
            fields["last_video_time_s"] = latest_event["video_time_s"]
        if recording:
            fields["recording_name"] = recording
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(
            f"UPDATE cases SET {assignments} WHERE case_id = ?", (*fields.values(), case_id)
        )

    # =================================================================
    # activity + notifications
    # =================================================================

    @staticmethod
    def _activity(conn, case_id, *, at, at_epoch, actor_type, actor_id, actor_name,
                  actor_department, action, from_status, to_status, note) -> None:
        conn.execute(
            "INSERT INTO case_activity(case_id, at, at_epoch, actor_type, actor_id, actor_name, "
            "actor_department, action, from_status, to_status, note) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (case_id, at, at_epoch if at_epoch is not None else time.time(), actor_type, actor_id,
             actor_name, actor_department, action, from_status, to_status, note),
        )

    @staticmethod
    def _notify(conn, *, department: Optional[str] = None, user_id: Optional[str] = None,
                severity: str = "INFO", kind: str, title: str, body: str = "",
                case_id: Optional[str] = None) -> None:
        conn.execute(
            "INSERT INTO notifications(created_at, created_epoch, department, user_id, severity, "
            "kind, title, body, case_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (_now_iso(), time.time(), department, user_id, severity, kind, title, body, case_id),
        )

    def _notify_new_case(self, conn, case_id, issue, first_event, owner, title, severity) -> None:
        where = issue.get("road_segment") or first_event.get("road_segment") or "unknown segment"
        bus = first_event.get("bus_id") or "a bus"
        body = f"{case_id} reported by {bus} on {where}"
        self._notify(conn, department=owner, severity=severity, kind="new_case",
                     title=f"{title} detected", body=body, case_id=case_id)
        critical = (
            severity == "CRITICAL" or issue.get("priority_band") == "CRITICAL"
            or issue.get("issue_type") == "hit_and_run"
        )
        if critical:
            self._notify(conn, department=rbac.CENTRAL_COMMAND, severity="CRITICAL",
                         kind="critical", title=f"Critical: {title}", body=body, case_id=case_id)
        if issue.get("issue_type") in ("traffic_congestion", "traffic_bottleneck"):
            delay = (first_event.get("metadata") or {}).get("estimated_delay") or 0
            if delay and float(delay) > 0:
                self._notify(
                    conn, department=rbac.TRANSPORT_AUTHORITY, severity="MEDIUM", kind="route_delay",
                    title=f"Route delay ~{float(delay):.1f} min",
                    body=f"{title} on {where} ({case_id}), reported by {bus}", case_id=case_id,
                )

    # =================================================================
    # reads
    # =================================================================

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self.db.query_one("SELECT * FROM cases WHERE case_id = ?", (case_id,))

    def _scope_clause(self, scope: rbac.Scope) -> Tuple[str, List[Any]]:
        if scope.is_central:
            return "1=1", []
        if scope.is_field:
            return "c.owner_department = ?", [scope.department]
        types = sorted(scope.visible_types or [])
        if not types:
            return "c.owner_department = ?", [scope.department]
        marks = ",".join("?" * len(types))
        return f"(c.owner_department = ? OR c.event_type IN ({marks}))", [scope.department, *types]

    def _decorate(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        row["buses"] = _json_list(row.get("buses"))
        row["status_index"] = STATUS_INDEX.get(row.get("status"), 0)
        row["category"] = CATEGORY.get(row.get("owner_department"), "Event")
        row["owner_department_label"] = rbac.DEPARTMENT_LABEL.get(row.get("owner_department"), row.get("owner_department"))
        row["gps"] = (
            {"latitude": row["latitude"], "longitude": row["longitude"]}
            if row.get("latitude") is not None else None
        )
        if row.get("primary_event_id"):
            ev = self.db.query_one(
                "SELECT evidence_image FROM events WHERE event_id = ?", (row["primary_event_id"],)
            )
            row["thumbnail"] = f"/evidence/{ev['evidence_image']}" if ev and ev.get("evidence_image") else None
        return row

    def list_cases(
        self, scope: rbac.Scope, *, status: Optional[str] = None, open_only: bool = False,
        department: Optional[str] = None, event_type: Optional[str] = None,
        class_name: Optional[str] = None, bus_id: Optional[str] = None,
        assigned: Optional[str] = None, severity: Optional[str] = None,
        q: Optional[str] = None, sort: str = "priority", near: Optional[Tuple[float, float]] = None,
        limit: int = 100, offset: int = 0, viewer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        where, params = self._scope_clause(scope)
        clauses = [where]
        if status:
            wanted = [s.strip().upper() for s in status.split(",") if s.strip()]
            clauses.append(f"c.status IN ({','.join('?' * len(wanted))})")
            params.extend(wanted)
        if open_only:
            clauses.append("c.status != 'RESOLVED'")
        if department and department != "ALL":
            clauses.append("c.owner_department = ?")
            params.append(department)
        if event_type:
            types = [t.strip() for t in event_type.split(",") if t.strip()]
            clauses.append(f"c.event_type IN ({','.join('?' * len(types))})")
            params.extend(types)
        if class_name:
            classes = [t.strip() for t in class_name.split(",") if t.strip()]
            clauses.append(f"c.class_name IN ({','.join('?' * len(classes))})")
            params.extend(classes)
        if bus_id:
            clauses.append("(c.bus_id = ? OR c.buses LIKE ?)")
            params.extend([bus_id, f'%"{bus_id}"%'])
        if severity:
            wanted = [s.strip().upper() for s in severity.split(",") if s.strip()]
            clauses.append(f"c.severity IN ({','.join('?' * len(wanted))})")
            params.extend(wanted)
        if assigned == "me" and viewer_id:
            clauses.append("c.assignee_user_id = ?")
            params.append(viewer_id)
        elif assigned == "unassigned":
            clauses.append("c.assignee_user_id IS NULL")
        elif assigned:
            clauses.append("c.assignee_user_id = ?")
            params.append(assigned)
        if q:
            like = f"%{q.strip()}%"
            clauses.append("(c.case_id LIKE ? OR c.title LIKE ? OR c.road_segment LIKE ? OR c.bus_id LIKE ?)")
            params.extend([like, like, like, like])
        order = {
            "recent": "c.last_seen_epoch DESC",
            "oldest": "c.first_seen_epoch ASC",
            "status": "CASE c.status WHEN 'DETECTED' THEN 0 WHEN 'VERIFIED' THEN 1 WHEN 'ASSIGNED' THEN 2 "
                      "WHEN 'IN_PROGRESS' THEN 3 ELSE 4 END, c.priority_score DESC",
        }.get(sort, "CASE WHEN c.status = 'RESOLVED' THEN 1 ELSE 0 END, "
                    "IFNULL(c.priority_score,0) DESC, c.last_seen_epoch DESC")
        sql_where = " AND ".join(clauses)
        total = self.db.query_one(f"SELECT COUNT(*) AS n FROM cases c WHERE {sql_where}", params)["n"]
        rows = self.db.query(
            f"SELECT c.*, u.name AS assignee_name FROM cases c "
            f"LEFT JOIN users u ON u.user_id = c.assignee_user_id "
            f"WHERE {sql_where} ORDER BY {order} LIMIT ? OFFSET ?",
            [*params, int(limit) if sort != "distance" else 1000, int(offset) if sort != "distance" else 0],
        )
        items = [self._decorate(r) for r in rows]
        if near is not None:
            for item in items:
                if item.get("latitude") is not None:
                    item["distance_m"] = round(_haversine_m(near[0], near[1], item["latitude"], item["longitude"]))
            if sort == "distance":
                items.sort(key=lambda i: i.get("distance_m", float("inf")))
                items = items[int(offset): int(offset) + int(limit)]
        return {"total": total, "items": items}

    def summary(self, scope: rbac.Scope, viewer_id: Optional[str] = None) -> Dict[str, Any]:
        where, params = self._scope_clause(scope)
        rows = self.db.query(
            f"SELECT status, owner_department, event_type, class_name, severity, "
            f"assignee_user_id, resolved_at, priority_band FROM cases c WHERE {where}", params,
        )
        out: Dict[str, Any] = {
            "total": len(rows), "open": 0, "by_status": {s: 0 for s in STATUSES},
            "by_department": {}, "by_type": {}, "by_class": {}, "open_by_severity": {},
            "assigned_to_me": 0, "assigned_to_me_open": 0, "unassigned_open": 0,
            "critical_open": 0, "resolved_24h": 0,
        }
        day_ago = time.time() - 86_400
        for r in rows:
            status = r["status"]
            out["by_status"][status] = out["by_status"].get(status, 0) + 1
            dept = r["owner_department"]
            bucket = out["by_department"].setdefault(dept, {"total": 0, "open": 0})
            bucket["total"] += 1
            key = r["class_name"] if r["event_type"] == "road_defect" and r["class_name"] else r["event_type"]
            out["by_type"][r["event_type"]] = out["by_type"].get(r["event_type"], 0) + 1
            out["by_class"][key] = out["by_class"].get(key, 0) + 1
            is_open = status != "RESOLVED"
            if is_open:
                out["open"] += 1
                bucket["open"] += 1
                sev = r["severity"] or "INFO"
                out["open_by_severity"][sev] = out["open_by_severity"].get(sev, 0) + 1
                if sev == "CRITICAL" or r["priority_band"] == "CRITICAL":
                    out["critical_open"] += 1
                if not r["assignee_user_id"]:
                    out["unassigned_open"] += 1
            if viewer_id and r["assignee_user_id"] == viewer_id:
                out["assigned_to_me"] += 1
                if is_open:
                    out["assigned_to_me_open"] += 1
            if status == "RESOLVED" and r["resolved_at"]:
                try:
                    if datetime.fromisoformat(r["resolved_at"]).timestamp() >= day_ago:
                        out["resolved_24h"] += 1
                except ValueError:
                    pass
        out["open_by_class"] = {}
        for r in rows:
            if r["status"] != "RESOLVED":
                key = r["class_name"] if r["event_type"] == "road_defect" and r["class_name"] else r["event_type"]
                out["open_by_class"][key] = out["open_by_class"].get(key, 0) + 1
        return out

    def activity(self, case_id: str) -> List[Dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM case_activity WHERE case_id = ? ORDER BY at_epoch ASC, id ASC", (case_id,)
        )

    def observations(self, case_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT e.event_id, e.event_type, e.bus_id, e.camera_id, e.timestamp, e.confidence, "
            "e.severity, e.evidence_image, e.frame_number, e.video_time_s, e.track_id, e.issue_id "
            "FROM events e JOIN case_issues c ON c.issue_id = e.issue_id "
            "WHERE c.case_id = ? AND e.event_type != 'persistent_issue' "
            "ORDER BY e.ts_epoch DESC LIMIT ?",
            (case_id, int(limit)),
        )
        return rows

    def allowed_actions(self, case: Dict[str, Any], who) -> List[Dict[str, Any]]:
        out = []
        current = STATUS_INDEX.get(case["status"], 0)
        for action in ACTION_ORDER:
            frm, to = ACTIONS[action]
            done = current >= STATUS_INDEX[to]
            reassign = action == "assign" and case["status"] == "ASSIGNED"
            reason = None
            enabled = case["status"] == frm or reassign
            if enabled:
                reason = self._permission_error(case, who, action)
                enabled = reason is None
            out.append({
                "action": action, "label": ACTION_LABEL[action], "to_status": to,
                "done": done and not reassign, "enabled": enabled,
                "reassign": reassign and enabled, "reason": reason,
            })
        return out

    def assignees(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT user_id, name, role, department FROM users WHERE is_active = 1 "
            "AND department = ? ORDER BY CASE role WHEN 'FIELD_OFFICER' THEN 0 ELSE 1 END, name",
            (case["owner_department"],),
        )
        for row in rows:
            row["role_label"] = rbac.ROLE_LABEL.get(row["role"], row["role"])
            row["open_tasks"] = self.db.query_one(
                "SELECT COUNT(*) AS n FROM cases WHERE assignee_user_id = ? AND status != 'RESOLVED'",
                (row["user_id"],),
            )["n"]
        return rows

    # =================================================================
    # transitions
    # =================================================================

    def _permission_error(self, case: Dict[str, Any], who, action: str) -> Optional[str]:
        scope = who.scope
        if scope.is_central:
            return None
        if case["owner_department"] != scope.department:
            label = rbac.DEPARTMENT_LABEL.get(case["owner_department"], case["owner_department"])
            return f"Only {label} can update this event"
        if scope.is_field and action in ("start", "resolve"):
            if case.get("assignee_user_id") != who.id:
                return "Assigned to someone else"
        return None

    def transition(self, case_id: str, who, action: str, *, assignee_user_id: Optional[str] = None,
                   note: Optional[str] = None, expected_status: Optional[str] = None) -> Dict[str, Any]:
        case = self.get_case(case_id)
        if case is None or not who.scope.sees_case(case):
            raise HTTPException(404, "event not found")
        if action not in ACTIONS:
            raise HTTPException(422, f"unknown action {action!r}")
        frm, to = ACTIONS[action]
        reassign = action == "assign" and case["status"] == "ASSIGNED"
        if expected_status and expected_status != case["status"]:
            raise HTTPException(409, {
                "message": f"{case_id} changed to {case['status']} before your update",
                "current_status": case["status"],
            })
        if case["status"] != frm and not reassign:
            raise HTTPException(409, {
                "message": f"{case_id} is {case['status']}; '{ACTION_LABEL[action]}' needs {frm}",
                "current_status": case["status"],
            })
        error = self._permission_error(case, who, action)
        if error:
            raise HTTPException(403, error)

        now = _now_iso()
        note = (note or "").strip()[:500] or None
        fields: Dict[str, Any] = {"status": to, "updated_at": now}
        assignee = None
        if action == "verify":
            fields.update(verified_by=who.id, verified_at=now)
        elif action == "assign":
            target_id = assignee_user_id or (who.id if who.scope.is_field else None)
            if not target_id:
                raise HTTPException(422, "choose who to assign this event to")
            if who.scope.is_field and target_id != who.id:
                raise HTTPException(403, "field officers can only assign events to themselves")
            assignee = self.db.query_one(
                "SELECT * FROM users WHERE user_id = ? AND is_active = 1", (target_id,)
            )
            if assignee is None:
                raise HTTPException(422, "assignee not found or inactive")
            if assignee["department"] != case["owner_department"]:
                raise HTTPException(422, "assignee must belong to the responsible department")
            fields.update(assignee_user_id=assignee["user_id"], assigned_by=who.id, assigned_at=now)
        elif action == "start":
            fields.update(started_at=now)
        elif action == "resolve":
            fields.update(resolved_at=now, resolved_by=who.id, resolution_note=note,
                          observations_at_resolve=case.get("observation_count") or 0)

        dept_label = rbac.DEPARTMENT_LABEL.get(who.user["department"], who.user["department"])
        if action == "verify":
            text = f"Verified by {who.name} ({dept_label})"
        elif action == "assign":
            text = ("Reassigned" if reassign else "Assigned") + f" to {assignee['name']} by {who.name}"
        elif action == "start":
            text = f"Work started by {who.name}"
        else:
            text = f"Resolved by {who.name}"
        if note:
            text += f" -- {note}"

        with self.db.transaction() as conn:
            current = conn.execute(
                "SELECT status, version FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            if current["status"] != case["status"]:
                raise HTTPException(409, {
                    "message": f"{case_id} changed to {current['status']} before your update",
                    "current_status": current["status"],
                })
            assignments = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(
                f"UPDATE cases SET {assignments}, version = version + 1 WHERE case_id = ?",
                (*fields.values(), case_id),
            )
            self._activity(
                conn, case_id, at=now, at_epoch=time.time(), actor_type="user",
                actor_id=who.id, actor_name=who.name, actor_department=who.user["department"],
                action="REASSIGNED" if reassign else to, from_status=case["status"],
                to_status=to, note=text,
            )
            if action == "assign" and assignee["user_id"] != who.id:
                self._notify(
                    conn, user_id=assignee["user_id"], severity=case.get("severity") or "INFO",
                    kind="assignment", title=f"{case_id} assigned to you",
                    body=f"{case['title']} on {case.get('road_segment') or 'the map'} -- by {who.name}",
                    case_id=case_id,
                )
            if action == "resolve":
                if who.scope.is_field or who.scope.is_central:
                    self._notify(
                        conn, department=case["owner_department"], severity="INFO", kind="resolved",
                        title=f"{case_id} resolved", body=f"{case['title']} -- {text}", case_id=case_id,
                    )
                if case.get("severity") == "CRITICAL" or case.get("event_type") == "hit_and_run":
                    self._notify(
                        conn, department=rbac.CENTRAL_COMMAND, severity="INFO", kind="resolved",
                        title=f"{case_id} resolved", body=f"{case['title']} -- {text}", case_id=case_id,
                    )
        return self.get_case(case_id)

    # =================================================================
    # notifications
    # =================================================================

    def _notification_clause(self, who, include_all: bool = False) -> Tuple[str, List[Any]]:
        scope = who.scope
        if scope.is_central and include_all:
            return "1=1", []
        if scope.is_field:
            return ("(n.user_id = ? OR (n.department = ? AND n.severity IN ('HIGH','CRITICAL')))",
                    [who.id, scope.department])
        return "(n.user_id = ? OR n.department = ?)", [who.id, scope.department]

    def notifications(self, who, *, unread_only: bool = False, limit: int = 50,
                      department: Optional[str] = None) -> Dict[str, Any]:
        include_all = bool(department)
        where, params = self._notification_clause(who, include_all=include_all)
        clauses = [where]
        if include_all and department and department != "ALL":
            clauses.append("n.department = ?")
            params.append(department)
        sql_where = " AND ".join(clauses)
        base = (
            f"FROM notifications n LEFT JOIN notification_reads r "
            f"ON r.notification_id = n.id AND r.user_id = ? WHERE {sql_where}"
        )
        rows = self.db.query(
            f"SELECT n.*, r.read_at {base} {'AND r.read_at IS NULL' if unread_only else ''} "
            f"ORDER BY n.id DESC LIMIT ?",
            [who.id, *params, int(limit)],
        )
        for row in rows:
            row["read"] = row.get("read_at") is not None
            row["department_label"] = rbac.DEPARTMENT_LABEL.get(row.get("department"), row.get("department"))
        return {"items": rows, "unread": self.unread_count(who)}

    def unread_count(self, who) -> int:
        where, params = self._notification_clause(who)
        row = self.db.query_one(
            f"SELECT COUNT(*) AS n FROM notifications n LEFT JOIN notification_reads r "
            f"ON r.notification_id = n.id AND r.user_id = ? WHERE {where} AND r.read_at IS NULL",
            [who.id, *params],
        )
        return row["n"] if row else 0

    def mark_read(self, who, ids: Optional[Iterable[int]] = None) -> int:
        if ids is None:
            where, params = self._notification_clause(who)
            rows = self.db.query(f"SELECT n.id FROM notifications n WHERE {where}", params)
            ids = [r["id"] for r in rows]
        ids = list(ids)
        now = time.time()
        with self.db.transaction() as conn:
            for nid in ids:
                conn.execute(
                    "INSERT OR IGNORE INTO notification_reads(notification_id, user_id, read_at) VALUES (?,?,?)",
                    (int(nid), who.id, now),
                )
        return len(ids)

    def notify_department(self, department: str, *, kind: str, title: str, body: str = "",
                          severity: str = "INFO", case_id: Optional[str] = None) -> None:
        with self.db.transaction() as conn:
            self._notify(conn, department=department, severity=severity, kind=kind,
                         title=title, body=body, case_id=case_id)

    # =================================================================
    # change token for polling clients
    # =================================================================

    def state_token(self) -> str:
        row = self.db.query_one(
            "SELECT (SELECT COUNT(*) FROM cases) AS c, (SELECT IFNULL(MAX(id),0) FROM case_activity) AS a, "
            "(SELECT IFNULL(MAX(id),0) FROM notifications) AS n, "
            "(SELECT IFNULL(SUM(observation_count),0) FROM cases) AS o, "
            "(SELECT COUNT(*) FROM events) AS e"
        )
        return f"{row['c']}.{row['a']}.{row['n']}.{row['o']}.{row['e']}"
