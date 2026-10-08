"""
backend.fleet
=============

Fleet status and transport analytics for the Transport Authority.

The registry (which buses exist, which route each runs, depot, service
status) is master data kept in the ``fleet`` table and seeded once from
``config/platform.yaml -> fleet``.  Everything else is DERIVED from what the
buses actually reported -- nothing is simulated here:

* last known position   latest GPS fix carried by any of the bus's events
* trips (O/D)           one trip per pipeline run (``run_id``); runs from
                        before run ids were stamped are split on gaps
* distance / coverage   haversine along reported positions, distinct road
                        segments
* route delays          Person 2's ``estimated_delay`` on congestion events
* traffic impact        congestion and incident cases on the route's segments

A bus that reported nothing is listed as such, not given a made-up position.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

FLEET_SCHEMA = """
CREATE TABLE IF NOT EXISTS fleet (
    bus_id        TEXT PRIMARY KEY,
    registration  TEXT,
    route_id      TEXT,
    depot         TEXT,
    status        TEXT DEFAULT 'IN_SERVICE',
    cameras       TEXT,
    notes         TEXT,
    updated_at    TEXT
);
"""

SERVICE_STATUSES = ["IN_SERVICE", "MAINTENANCE", "OUT_OF_SERVICE"]
TRIP_GAP_S = 600.0
MAX_JUMP_M = 2_000.0


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _meta(row) -> Dict[str, Any]:
    try:
        return json.loads(row.get("metadata") or "{}")
    except Exception:                                            # noqa: BLE001
        return {}


class FleetStore:
    def __init__(self, database, config, log=None):
        self.db = database
        self.config = config
        self.log = log
        section = config.section("fleet")
        self.active_window_s = float(section.get("active_window_minutes", 30)) * 60
        self.routes_cfg = {r["route_id"]: r for r in (section.get("routes") or []) if r.get("route_id")}
        self.segment_size_m = float(config.get("gps.segment_size_m", 60.0))

    def initialise(self) -> "FleetStore":
        self.db.executescript(FLEET_SCHEMA)
        row = self.db.query_one("SELECT COUNT(*) AS n FROM fleet")
        if row and not row["n"]:
            for bus in self.config.section("fleet").get("buses") or []:
                if not bus.get("bus_id"):
                    continue
                self.db.execute(
                    "INSERT OR IGNORE INTO fleet(bus_id, registration, route_id, depot, status, "
                    "cameras, notes, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (bus["bus_id"], bus.get("registration"), bus.get("route_id"), bus.get("depot"),
                     bus.get("status", "IN_SERVICE"), json.dumps(bus.get("cameras") or []),
                     bus.get("notes"), datetime.now(timezone.utc).isoformat(timespec="seconds")),
                )
        return self

    # -- registry ------------------------------------------------------------

    def routes(self) -> List[Dict[str, Any]]:
        return list(self.routes_cfg.values())

    def registry(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for row in self.db.query("SELECT * FROM fleet"):
            try:
                row["cameras"] = json.loads(row.get("cameras") or "[]")
            except Exception:                                    # noqa: BLE001
                row["cameras"] = []
            out[row["bus_id"]] = row
        return out

    def update_bus(self, bus_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {k: fields[k] for k in ("registration", "route_id", "depot", "status", "notes") if k in fields}
        if "status" in allowed and allowed["status"] not in SERVICE_STATUSES:
            raise ValueError(f"status must be one of {SERVICE_STATUSES}")
        if "cameras" in fields:
            allowed["cameras"] = json.dumps(fields["cameras"] or [])
        existing = self.db.query_one("SELECT bus_id FROM fleet WHERE bus_id = ?", (bus_id,))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if existing is None:
            self.db.execute("INSERT INTO fleet(bus_id, status, cameras, updated_at) VALUES (?,?,?,?)",
                            (bus_id, "IN_SERVICE", "[]", now))
        if allowed:
            allowed["updated_at"] = now
            assignments = ", ".join(f"{k} = ?" for k in allowed)
            self.db.execute(f"UPDATE fleet SET {assignments} WHERE bus_id = ?", (*allowed.values(), bus_id))
        return self.registry().get(bus_id)

    # -- per-bus derivation --------------------------------------------------

    def _bus_ids(self) -> List[str]:
        seen = [r["bus_id"] for r in self.db.query(
            "SELECT DISTINCT bus_id FROM events WHERE bus_id IS NOT NULL")]
        ids = list(self.registry().keys())
        for bus in seen:
            if bus not in ids:
                ids.append(bus)
        return sorted(ids)

    def _positions(self, bus_id: str) -> List[Dict[str, Any]]:
        return self.db.query(
            "SELECT event_id, event_type, timestamp, ts_epoch, latitude, longitude, road_segment, "
            "video_time_s, run_id, metadata FROM events WHERE bus_id = ? AND latitude IS NOT NULL "
            "AND event_type != 'persistent_issue' ORDER BY ts_epoch ASC",
            (bus_id,),
        )

    def _trips(self, positions: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        trips: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for row in positions:
            if current:
                prev = current[-1]
                new_run = (row.get("run_id") or prev.get("run_id")) and row.get("run_id") != prev.get("run_id")
                gap = (row["ts_epoch"] or 0) - (prev["ts_epoch"] or 0) > TRIP_GAP_S
                if new_run or gap:
                    trips.append(current)
                    current = []
            current.append(row)
        if current:
            trips.append(current)
        return trips

    @staticmethod
    def _distance(points: List[Dict[str, Any]]) -> float:
        total = 0.0
        ordered = sorted(points, key=lambda p: (p.get("video_time_s") is None, p.get("video_time_s") or 0, p["ts_epoch"] or 0))
        for a, b in zip(ordered, ordered[1:]):
            step = _haversine_m(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
            if step <= MAX_JUMP_M:
                total += step
        return total

    def _trip_summary(self, trip: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
        ordered = sorted(trip, key=lambda p: (p.get("video_time_s") is None, p.get("video_time_s") or 0, p["ts_epoch"] or 0))
        first, last = ordered[0], ordered[-1]
        video_times = [p["video_time_s"] for p in trip if p.get("video_time_s") is not None]
        duration = (max(video_times) - min(video_times)) if len(video_times) >= 2 else None
        types: Dict[str, int] = {}
        for p in trip:
            types[p["event_type"]] = types.get(p["event_type"], 0) + 1
        return {
            "trip": index + 1,
            "run_id": first.get("run_id"),
            "started": first["timestamp"],
            "origin": {"latitude": first["latitude"], "longitude": first["longitude"], "segment": first.get("road_segment")},
            "destination": {"latitude": last["latitude"], "longitude": last["longitude"], "segment": last.get("road_segment")},
            "distance_m": round(self._distance(trip)),
            "duration_s": round(duration, 1) if duration is not None else None,
            "events": len(trip),
            "events_by_type": types,
            "segments": len({p["road_segment"] for p in trip if p.get("road_segment")}),
        }

    def _cases_for_bus(self, bus_id: str, scope=None) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT case_id, title, event_type, class_name, status, severity, owner_department, "
            "assignee_user_id, road_segment, last_seen FROM cases WHERE bus_id = ? OR buses LIKE ? "
            "ORDER BY last_seen_epoch DESC",
            (bus_id, f'%"{bus_id}"%'),
        )
        # Transport sees which events a bus raised only where its own
        # visibility allows -- same rule as GET /cases.
        return [r for r in rows if scope is None or scope.sees_case(r)]

    def bus_summary(self, bus_id: str, registry: Dict[str, Dict[str, Any]], scope=None) -> Dict[str, Any]:
        reg = registry.get(bus_id) or {}
        positions = self._positions(bus_id)
        counts = self.db.query(
            "SELECT event_type, COUNT(*) AS n FROM events WHERE bus_id = ? AND event_type != 'persistent_issue' "
            "GROUP BY event_type", (bus_id,),
        )
        last = positions[-1] if positions else None
        last_epoch = last["ts_epoch"] if last else None
        if reg.get("status") in ("MAINTENANCE", "OUT_OF_SERVICE"):
            state = reg["status"]
        elif last_epoch is None:
            state = "NO_REPORTS"
        elif time.time() - last_epoch <= self.active_window_s:
            state = "REPORTING"
        else:
            state = "IDLE"
        cases = self._cases_for_bus(bus_id, scope)
        route = self.routes_cfg.get(reg.get("route_id") or "")
        trips = self._trips(positions)
        return {
            "bus_id": bus_id,
            "registered": bool(reg),
            "registration": reg.get("registration"),
            "route_id": reg.get("route_id"),
            "route_name": (route or {}).get("name"),
            "depot": reg.get("depot"),
            "service_status": reg.get("status") or ("UNREGISTERED" if not reg else "IN_SERVICE"),
            "cameras": reg.get("cameras") or [],
            "state": state,
            "last_report": last["timestamp"] if last else None,
            "last_position": {"latitude": last["latitude"], "longitude": last["longitude"],
                              "segment": last.get("road_segment")} if last else None,
            "events_total": sum(c["n"] for c in counts),
            "events_by_type": {c["event_type"]: c["n"] for c in counts},
            "cases_open": sum(1 for c in cases if c["status"] != "RESOLVED"),
            "cases_total": len(cases),
            "trips": len(trips),
            "distance_m": round(sum(self._distance(t) for t in trips)),
            "segments_covered": len({p["road_segment"] for p in positions if p.get("road_segment")}),
        }

    def overview(self, scope=None) -> Dict[str, Any]:
        registry = self.registry()
        buses = [self.bus_summary(bus_id, registry, scope) for bus_id in self._bus_ids()]
        segments = self.db.query_one(
            "SELECT COUNT(DISTINCT road_segment) AS n FROM events WHERE road_segment IS NOT NULL")["n"]
        return {
            "buses": buses,
            "totals": {
                "registered": sum(1 for b in buses if b["registered"]),
                "reporting": sum(1 for b in buses if b["state"] == "REPORTING"),
                "idle": sum(1 for b in buses if b["state"] == "IDLE"),
                "no_reports": sum(1 for b in buses if b["state"] == "NO_REPORTS"),
                "maintenance": sum(1 for b in buses if b["state"] in ("MAINTENANCE", "OUT_OF_SERVICE")),
                "trips": sum(b["trips"] for b in buses),
                "distance_m": sum(b["distance_m"] for b in buses),
                "segments_covered": segments,
                "approx_road_km": round(segments * self.segment_size_m / 1000.0, 2),
                "events": sum(b["events_total"] for b in buses),
                "cases_open": sum(b["cases_open"] for b in buses),
            },
            "active_window_minutes": round(self.active_window_s / 60),
        }

    def bus_detail(self, bus_id: str, scope=None) -> Optional[Dict[str, Any]]:
        registry = self.registry()
        if bus_id not in registry and not self.db.query_one(
                "SELECT 1 AS x FROM events WHERE bus_id = ? LIMIT 1", (bus_id,)):
            return None
        summary = self.bus_summary(bus_id, registry, scope)
        positions = self._positions(bus_id)
        trips = [self._trip_summary(t, i) for i, t in enumerate(self._trips(positions))]
        trail = [
            {"latitude": p["latitude"], "longitude": p["longitude"], "timestamp": p["timestamp"],
             "video_time_s": p.get("video_time_s")}
            for p in positions if p["event_type"] == "traffic_observation"
        ] or [
            {"latitude": p["latitude"], "longitude": p["longitude"], "timestamp": p["timestamp"],
             "video_time_s": p.get("video_time_s")} for p in positions
        ]
        summary.update({
            "trips_detail": list(reversed(trips)),
            "trail": trail[-400:],
            "cases": self._cases_for_bus(bus_id, scope)[:100],
        })
        return summary

    # -- routes --------------------------------------------------------------

    def route_performance(self) -> List[Dict[str, Any]]:
        registry = self.registry()
        by_route: Dict[str, List[str]] = {}
        for bus_id in self._bus_ids():
            route = (registry.get(bus_id) or {}).get("route_id") or "UNASSIGNED"
            by_route.setdefault(route, []).append(bus_id)
        for route_id in self.routes_cfg:
            by_route.setdefault(route_id, [])

        out = []
        for route_id, buses in sorted(by_route.items()):
            cfg = self.routes_cfg.get(route_id, {})
            if not buses:
                out.append({
                    "route_id": route_id, "name": cfg.get("name"), "origin": cfg.get("origin"),
                    "destination": cfg.get("destination"), "buses": [], "segments": 0,
                    "observations": 0, "avg_vehicles": None, "avg_density_pct": None,
                    "congested_share": None, "delay_minutes": 0.0, "congestion_events": 0,
                    "incidents": 0, "road_defects": 0, "impact": [],
                })
                continue
            marks = ",".join("?" * len(buses))
            segs = [r["road_segment"] for r in self.db.query(
                f"SELECT DISTINCT road_segment FROM events WHERE bus_id IN ({marks}) "
                f"AND road_segment IS NOT NULL", buses)]
            seg_rows = []
            if segs:
                seg_marks = ",".join("?" * len(segs))
                seg_rows = self.db.query(f"SELECT * FROM segments WHERE road_segment IN ({seg_marks})", segs)
            samples = sum(r.get("samples") or 0 for r in seg_rows)
            congested = sum(r.get("congested_samples") or 0 for r in seg_rows)
            vehicles = sum(r.get("vehicle_total") or 0 for r in seg_rows)
            density = sum(r.get("density_total") or 0 for r in seg_rows)

            congestion = self.db.query(
                f"SELECT issue_id, metadata FROM events WHERE bus_id IN ({marks}) "
                f"AND event_type IN ('traffic_congestion','traffic_bottleneck')", buses)
            delay_by_issue: Dict[str, float] = {}
            for row in congestion:
                key = row.get("issue_id") or id(row)
                delay = float(_meta(row).get("estimated_delay") or 0.0)
                delay_by_issue[key] = max(delay_by_issue.get(key, 0.0), delay)
            counts = self.db.query_one(
                f"SELECT SUM(CASE WHEN event_type IN ('collision','possible_collision','hit_and_run',"
                f"'rash_driving','abnormal_movement') THEN 1 ELSE 0 END) AS incidents, "
                f"SUM(CASE WHEN event_type = 'road_defect' THEN 1 ELSE 0 END) AS defects "
                f"FROM events WHERE bus_id IN ({marks})", buses)
            impact = []
            if segs:
                seg_marks = ",".join("?" * len(segs))
                impact = self.db.query(
                    f"SELECT case_id, title, event_type, status, severity, road_segment, last_seen "
                    f"FROM cases WHERE road_segment IN ({seg_marks}) AND event_type IN "
                    f"('traffic_congestion','traffic_bottleneck','collision','possible_collision',"
                    f"'hit_and_run','abnormal_movement','rash_driving') ORDER BY last_seen_epoch DESC LIMIT 20",
                    segs)
            out.append({
                "route_id": route_id,
                "name": cfg.get("name") or ("Buses without a route" if route_id == "UNASSIGNED" else route_id),
                "origin": cfg.get("origin"), "destination": cfg.get("destination"),
                "buses": buses, "segments": len(segs), "observations": samples,
                "avg_vehicles": round(vehicles / samples, 2) if samples else None,
                "avg_density_pct": round(density / samples, 1) if samples else None,
                "congested_share": round(congested / samples, 3) if samples else None,
                "delay_minutes": round(sum(delay_by_issue.values()), 2),
                "congestion_events": len(congestion),
                "incidents": counts["incidents"] or 0, "road_defects": counts["defects"] or 0,
                "impact": impact,
            })
        return out

    def od_matrix(self) -> List[Dict[str, Any]]:
        """Origin -> destination segment pairs over every trip of every bus."""
        pairs: Dict[tuple, Dict[str, Any]] = {}
        for bus_id in self._bus_ids():
            for i, trip in enumerate(self._trips(self._positions(bus_id))):
                summary = self._trip_summary(trip, i)
                key = (summary["origin"]["segment"], summary["destination"]["segment"])
                bucket = pairs.setdefault(key, {
                    "origin_segment": key[0], "destination_segment": key[1],
                    "origin": summary["origin"], "destination": summary["destination"],
                    "trips": 0, "buses": [], "distance_m": 0, "events": 0,
                })
                bucket["trips"] += 1
                bucket["distance_m"] += summary["distance_m"]
                bucket["events"] += summary["events"]
                if bus_id not in bucket["buses"]:
                    bucket["buses"].append(bus_id)
        return sorted(pairs.values(), key=lambda p: p["trips"], reverse=True)
