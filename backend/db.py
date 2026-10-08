"""
backend.db
==========

SQLite storage for the platform.  Deliberately small -- six tables, no ORM,
no migrations framework -- because the spec says not to over-engineer it and
because a single file database is the easiest thing to hand to a teammate on
Windows.

    events              every common-schema Event, exactly as produced
    issues              persistent issues (POT-001 ...) from the
                        intelligence engine
    issue_observations  which event saw which issue, from which bus
    vehicles            ANPR identifications per (bus, track)
    segments            per-road-segment traffic aggregate (heatmap rows)
    runs                one row per pipeline run, for the dashboard header

Everything else the API needs is a query over ``events``, which is why road
defects / traffic / incidents / plates do not each get their own table.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    source_module   TEXT NOT NULL,
    bus_id          TEXT,
    camera_id       TEXT,
    timestamp       TEXT,
    ts_epoch        REAL,
    latitude        REAL,
    longitude       REAL,
    confidence      REAL,
    severity        TEXT,
    track_id        INTEGER,
    class_name      TEXT,
    bounding_box    TEXT,
    evidence_image  TEXT,
    evidence_video  TEXT,
    frame_number    INTEGER,
    video_time_s    REAL,
    road_segment    TEXT,
    issue_id        TEXT,
    priority_score  REAL,
    department      TEXT,
    duplicate_of    TEXT,
    run_id          TEXT,
    metadata        TEXT,
    created_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_events_type   ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_time   ON events(ts_epoch);
CREATE INDEX IF NOT EXISTS idx_events_issue  ON events(issue_id);
CREATE INDEX IF NOT EXISTS idx_events_seg    ON events(road_segment);
CREATE INDEX IF NOT EXISTS idx_events_track  ON events(bus_id, track_id);

CREATE TABLE IF NOT EXISTS issues (
    issue_id          TEXT PRIMARY KEY,
    issue_type        TEXT,
    class_name        TEXT,
    status            TEXT,
    severity          TEXT,
    first_seen        TEXT,
    last_seen         TEXT,
    first_seen_epoch  REAL,
    last_seen_epoch   REAL,
    observation_count INTEGER DEFAULT 0,
    bus_count         INTEGER DEFAULT 0,
    buses             TEXT,
    latitude          REAL,
    longitude         REAL,
    road_segment      TEXT,
    priority_score    REAL,
    priority_band     TEXT,
    department        TEXT,
    clean_passes      INTEGER DEFAULT 0,
    reopened_count    INTEGER DEFAULT 0,
    repair_observed   INTEGER DEFAULT 0,
    resolved_at       TEXT,
    last_event_id     TEXT,
    metadata          TEXT
);
CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status);
CREATE INDEX IF NOT EXISTS idx_issues_seg    ON issues(road_segment, class_name);

CREATE TABLE IF NOT EXISTS issue_observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id    TEXT NOT NULL,
    event_id    TEXT NOT NULL,
    bus_id      TEXT,
    timestamp   TEXT,
    ts_epoch    REAL,
    latitude    REAL,
    longitude   REAL,
    confidence  REAL,
    severity    TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_issue ON issue_observations(issue_id);

CREATE TABLE IF NOT EXISTS vehicles (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    bus_id              TEXT,
    track_id            INTEGER,
    class_name          TEXT,
    registration_number TEXT,
    plate_confidence    REAL,
    plate_valid         INTEGER,
    first_seen          TEXT,
    last_seen           TEXT,
    event_id            TEXT,
    incident_event_id   TEXT,
    incident_type       TEXT,
    evidence_image      TEXT,
    UNIQUE(bus_id, track_id)
);

CREATE TABLE IF NOT EXISTS segments (
    road_segment          TEXT PRIMARY KEY,
    latitude              REAL,
    longitude             REAL,
    samples               INTEGER DEFAULT 0,
    vehicle_total         INTEGER DEFAULT 0,
    density_total         REAL DEFAULT 0,
    speed_total           REAL DEFAULT 0,
    congested_samples     INTEGER DEFAULT 0,
    max_traffic_level     TEXT,
    defect_count          INTEGER DEFAULT 0,
    incident_count        INTEGER DEFAULT 0,
    last_seen             TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    bus_id      TEXT,
    camera_id   TEXT,
    source      TEXT,
    started_at  TEXT,
    finished_at TEXT,
    frames      INTEGER,
    events      INTEGER,
    summary     TEXT
);
"""

_TRAFFIC_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _json(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except Exception:                                            # noqa: BLE001
        return json.dumps(str(value))


def _unjson(value, default=None):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:                                            # noqa: BLE001
        return default


class Database:
    """Thin, thread-safe wrapper around one SQLite file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    # -- lifecycle ----------------------------------------------------------

    def initialise(self) -> "Database":
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        return self

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:                                    # noqa: BLE001
                pass

    # -- generic helpers ----------------------------------------------------
    # Used by the platform layer (auth, workflow, fleet, recordings) so every
    # table lives in this ONE database file, behind this ONE lock.

    def executescript(self, script: str) -> None:
        with self._lock:
            self._conn.executescript(script)
            self._conn.commit()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._lock:
            cursor = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cursor.rowcount

    def query(self, sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(sql, tuple(params)).fetchone()
        return dict(row) if row else None

    @contextmanager
    def transaction(self):
        """Several writes that must land together (a status change and its
        activity-log row).  Holds the lock for the whole block, commits on
        success, rolls back on any exception and re-raises it."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    # -- events -------------------------------------------------------------

    def insert_event(self, event, run_id: Optional[str] = None) -> None:
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        gps = data.get("gps") or {}
        # The orchestrator stamps metadata.run_id on every event it
        # dispatches; storing it in the column lets evidence be linked to
        # exactly the annotated recording of that run.
        run_id = run_id or (data.get("metadata") or {}).get("run_id")
        row = (
            data.get("event_id"),
            data.get("event_type"),
            data.get("source_module"),
            data.get("bus_id"),
            data.get("camera_id"),
            data.get("timestamp"),
            getattr(event, "epoch", time.time()),
            gps.get("latitude"),
            gps.get("longitude"),
            data.get("confidence"),
            data.get("severity"),
            data.get("track_id"),
            data.get("class_name"),
            _json(data.get("bounding_box")),
            data.get("evidence_image"),
            data.get("evidence_video"),
            data.get("frame_number"),
            data.get("video_time_s"),
            data.get("road_segment"),
            data.get("issue_id"),
            data.get("priority_score"),
            data.get("department"),
            data.get("duplicate_of"),
            run_id,
            _json(data.get("metadata")),
            time.time(),
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO events VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )
            self._conn.commit()
        self._post_insert(data)

    def _post_insert(self, data: Dict[str, Any]) -> None:
        """Keep the derived tables (vehicles / segments) in step."""
        metadata = data.get("metadata") or {}
        if data.get("event_type") == "vehicle_identification":
            self.upsert_vehicle(
                bus_id=data.get("bus_id"),
                track_id=data.get("track_id"),
                class_name=data.get("class_name"),
                registration_number=metadata.get("registration_number"),
                plate_confidence=metadata.get("plate_confidence"),
                plate_valid=metadata.get("plate_valid_format"),
                timestamp=data.get("timestamp"),
                event_id=data.get("event_id"),
                incident_event_id=metadata.get("triggered_by_event_id"),
                incident_type=metadata.get("triggered_by_event_type"),
                evidence_image=data.get("evidence_image"),
            )

        segment = data.get("road_segment")
        if not segment:
            return
        gps = data.get("gps") or {}
        kind = data.get("event_type")
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO segments(road_segment, latitude, longitude) "
                "VALUES (?,?,?)",
                (segment, gps.get("latitude"), gps.get("longitude")),
            )
            if kind == "traffic_observation":
                level = metadata.get("traffic_level", "LOW")
                self._conn.execute(
                    """UPDATE segments SET
                         samples = samples + 1,
                         vehicle_total = vehicle_total + ?,
                         density_total = density_total + ?,
                         speed_total = speed_total + ?,
                         congested_samples = congested_samples +
                             CASE WHEN ? IN ('CONGESTED','BUILDING') THEN 1 ELSE 0 END,
                         max_traffic_level = CASE
                             WHEN max_traffic_level IS NULL THEN ?
                             WHEN ? = 'CRITICAL' THEN 'CRITICAL'
                             WHEN ? = 'HIGH' AND max_traffic_level != 'CRITICAL' THEN 'HIGH'
                             WHEN ? = 'MEDIUM' AND max_traffic_level = 'LOW' THEN 'MEDIUM'
                             ELSE max_traffic_level END,
                         last_seen = ?
                       WHERE road_segment = ?""",
                    (
                        metadata.get("vehicle_count", 0) or 0,
                        metadata.get("density_percentage", 0.0) or 0.0,
                        metadata.get("average_speed_px_s", 0.0) or 0.0,
                        metadata.get("congestion_status"),
                        level, level, level, level,
                        data.get("timestamp"), segment,
                    ),
                )
            elif kind == "road_defect":
                self._conn.execute(
                    "UPDATE segments SET defect_count = defect_count + 1, "
                    "last_seen = ? WHERE road_segment = ?",
                    (data.get("timestamp"), segment),
                )
            elif kind in (
                "collision", "possible_collision", "hit_and_run",
                "rash_driving", "abnormal_movement",
            ):
                self._conn.execute(
                    "UPDATE segments SET incident_count = incident_count + 1, "
                    "last_seen = ? WHERE road_segment = ?",
                    (data.get("timestamp"), segment),
                )
            self._conn.commit()

    def update_event_fields(self, event_id: str, **fields) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE events SET {assignments} WHERE event_id = ?",
                (*fields.values(), event_id),
            )
            self._conn.commit()

    def query_events(
        self,
        event_type: Optional[str] = None,
        event_types: Optional[Iterable[str]] = None,
        source_module: Optional[str] = None,
        bus_id: Optional[str] = None,
        severity: Optional[str] = None,
        issue_id: Optional[str] = None,
        road_segment: Optional[str] = None,
        since: Optional[float] = None,
        include_duplicates: bool = True,
        limit: int = 200,
        offset: int = 0,
        order: str = "desc",
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if event_types:
            types = list(event_types)
            clauses.append(f"event_type IN ({','.join('?' * len(types))})")
            params.extend(types)
        if source_module:
            clauses.append("source_module = ?")
            params.append(source_module)
        if bus_id:
            clauses.append("bus_id = ?")
            params.append(bus_id)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if issue_id:
            clauses.append("issue_id = ?")
            params.append(issue_id)
        if road_segment:
            clauses.append("road_segment = ?")
            params.append(road_segment)
        if since is not None:
            clauses.append("ts_epoch >= ?")
            params.append(since)
        if not include_duplicates:
            clauses.append("duplicate_of IS NULL")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        direction = "DESC" if str(order).lower() == "desc" else "ASC"
        sql = (
            f"SELECT * FROM events {where} "
            f"ORDER BY ts_epoch {direction} LIMIT ? OFFSET ?"
        )
        params.extend([int(limit), int(offset)])
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._event_row(row) for row in rows]

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return self._event_row(row) if row else None

    @staticmethod
    def _event_row(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["bounding_box"] = _unjson(data.get("bounding_box"))
        data["metadata"] = _unjson(data.get("metadata"), {})
        latitude = data.pop("latitude", None)
        longitude = data.pop("longitude", None)
        data["gps"] = (
            {"latitude": latitude, "longitude": longitude}
            if latitude is not None and longitude is not None
            else None
        )
        return data

    def count_events(self) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_type, COUNT(*) AS n FROM events GROUP BY event_type"
            ).fetchall()
        return {row["event_type"]: row["n"] for row in rows}

    # -- issues -------------------------------------------------------------

    def upsert_issue(self, issue: Dict[str, Any]) -> None:
        columns = list(issue.keys())
        placeholders = ",".join("?" * len(columns))
        values = [
            _json(v) if isinstance(v, (list, dict)) else v for v in issue.values()
        ]
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO issues({','.join(columns)}) "
                f"VALUES ({placeholders})",
                values,
            )
            self._conn.commit()

    def get_issue(self, issue_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM issues WHERE issue_id = ?", (issue_id,)
            ).fetchone()
        return self._issue_row(row) if row else None

    def list_issues(
        self,
        status: Optional[str] = None,
        class_name: Optional[str] = None,
        department: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if class_name:
            clauses.append("class_name = ?")
            params.append(class_name)
        if department:
            clauses.append("department = ?")
            params.append(department)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM issues {where} "
                f"ORDER BY priority_score DESC, last_seen_epoch DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._issue_row(row) for row in rows]

    def all_issues(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM issues").fetchall()
        return [self._issue_row(row) for row in rows]

    @staticmethod
    def _issue_row(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["buses"] = _unjson(data.get("buses"), [])
        data["metadata"] = _unjson(data.get("metadata"), {})
        return data

    def add_issue_observation(self, issue_id: str, event) -> None:
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        gps = data.get("gps") or {}
        with self._lock:
            self._conn.execute(
                "INSERT INTO issue_observations"
                "(issue_id, event_id, bus_id, timestamp, ts_epoch, latitude, "
                " longitude, confidence, severity) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    issue_id, data.get("event_id"), data.get("bus_id"),
                    data.get("timestamp"), getattr(event, "epoch", time.time()),
                    gps.get("latitude"), gps.get("longitude"),
                    data.get("confidence"), data.get("severity"),
                ),
            )
            self._conn.commit()

    def issue_observations(self, issue_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM issue_observations WHERE issue_id = ? "
                "ORDER BY ts_epoch DESC LIMIT ?",
                (issue_id, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    # -- vehicles -----------------------------------------------------------

    def upsert_vehicle(self, **kwargs) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO vehicles
                   (bus_id, track_id, class_name, registration_number,
                    plate_confidence, plate_valid, first_seen, last_seen,
                    event_id, incident_event_id, incident_type, evidence_image)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(bus_id, track_id) DO UPDATE SET
                     registration_number = excluded.registration_number,
                     plate_confidence    = excluded.plate_confidence,
                     plate_valid         = excluded.plate_valid,
                     last_seen           = excluded.last_seen,
                     event_id            = excluded.event_id,
                     incident_event_id   = excluded.incident_event_id,
                     incident_type       = excluded.incident_type,
                     evidence_image      = excluded.evidence_image""",
                (
                    kwargs.get("bus_id"), kwargs.get("track_id"),
                    kwargs.get("class_name"), kwargs.get("registration_number"),
                    kwargs.get("plate_confidence"),
                    1 if kwargs.get("plate_valid") else 0,
                    kwargs.get("timestamp"), kwargs.get("timestamp"),
                    kwargs.get("event_id"), kwargs.get("incident_event_id"),
                    kwargs.get("incident_type"), kwargs.get("evidence_image"),
                ),
            )
            self._conn.commit()

    def get_vehicle(self, track_id: int, bus_id: Optional[str] = None):
        sql = "SELECT * FROM vehicles WHERE track_id = ?"
        params: List[Any] = [track_id]
        if bus_id:
            sql += " AND bus_id = ?"
            params.append(bus_id)
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def list_vehicles(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM vehicles ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(row) for row in rows]

    # -- segments / heatmap -------------------------------------------------

    def list_segments(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM segments ORDER BY samples DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            samples = max(1, data.get("samples") or 0)
            data["avg_vehicle_count"] = round((data.get("vehicle_total") or 0) / samples, 2)
            data["avg_density_percentage"] = round((data.get("density_total") or 0) / samples, 1)
            data["avg_speed_px_s"] = round((data.get("speed_total") or 0) / samples, 2)
            data["congested_ratio"] = round((data.get("congested_samples") or 0) / samples, 3)
            out.append(data)
        return out

    # -- runs ---------------------------------------------------------------

    def start_run(self, run_id: str, bus_id, camera_id, source) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs(run_id, bus_id, camera_id, source, "
                "started_at) VALUES (?,?,?,?,datetime('now'))",
                (run_id, bus_id, camera_id, str(source)),
            )
            self._conn.commit()

    def finish_run(self, run_id: str, frames: int, events: int, summary: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET finished_at = datetime('now'), frames = ?, "
                "events = ?, summary = ? WHERE run_id = ?",
                (frames, events, _json(summary), run_id),
            )
            self._conn.commit()

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["summary"] = _unjson(data.get("summary"), {})
            out.append(data)
        return out
