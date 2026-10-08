"""
backend.app
===========

FastAPI backend + the dashboard's static files.

One backend, one SQLite database, two clients:

    frontend/index.html   the web dashboards (Central Command + departments)
    mobile/index.html     the mobile Command & Field app  (served at /mobile/)

Both log in through ``POST /auth/login`` and read and write the same tables,
so a status change made on a phone is what the web dashboard shows next.

Run it with::

    python run_backend.py            # http://127.0.0.1:8000   (web)
                                     # http://127.0.0.1:8000/mobile/  (mobile)

Original endpoints (unchanged paths and response shapes, now scoped to the
logged-in user's department -- see backend/rbac.py)
-----------------------------------------------------------------------------
GET  /health                    service + database status
POST /events                    ingest an event (used when backend.mode=http)
GET  /events                    filter by type/module/severity/bus/segment/since
GET  /events/{event_id}
GET  /road-defects              module 1
GET  /traffic                   module 2 (observations + congestion)
GET  /incidents                 module 4 (all incident types, filterable)
GET  /incidents/{incident_id}   one incident + state timeline + evidence
GET  /hit-and-run               hit-and-run alerts only
GET  /anpr                      every plate read, UNKNOWNs included
GET  /vehicles                  module 5 (ANPR identifications)
GET  /vehicles/{track_id}       one vehicle, with its incident + plate
GET  /issues                    module 6 persistent issues
GET  /issues/{issue_id}         one issue with its observation history
GET  /analytics                 counts, severities, departments, priorities
GET  /heatmap                   per-road-segment rows for the map
GET  /runs                      pipeline runs
GET  /videos, /videos/{name}    annotated recordings (Live Analysis, evidence)
GET  /evidence/{path}           evidence images / clips

Platform endpoints (backend/platform_routes.py)
-----------------------------------------------
/auth/*  /users  /departments  /roles  /cases/*  /notifications/*
/fleet/*  /map/layers  /sync/state
"""

from __future__ import annotations

import hmac
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Depends, FastAPI, HTTPException, Query, Request   # noqa: E402
from fastapi.middleware.cors import CORSMiddleware               # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles                       # noqa: E402

from core.config import load_config                               # noqa: E402
from core.event import Event                                      # noqa: E402
from core.logging_setup import get_logger, setup_logging          # noqa: E402
from backend import rbac                                          # noqa: E402
from backend.auth import (                                        # noqa: E402
    AuthStore, CurrentUser, current_user, optional_user, require_roles, require_types,
)
from backend.db import Database                                   # noqa: E402
from backend.fleet import FleetStore                              # noqa: E402
from backend.platform_routes import build_router                  # noqa: E402
from backend.recordings import RecordingIndex                     # noqa: E402
from backend.workflow import WorkflowStore                        # noqa: E402

INCIDENT_TYPES = [
    "collision", "possible_collision", "hit_and_run",
    "rash_driving", "abnormal_movement", "pedestrian_vehicle_risk",
]
TRAFFIC_TYPES = ["traffic_observation", "traffic_congestion", "traffic_bottleneck"]

#: metadata keys that identify a vehicle's owner -- only Central Command and
#: Public Safety may read them (rbac.Scope.can_view_plates).
PLATE_KEYS = ("registration_number", "plate_confidence", "plate_text", "plate_valid_format")

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


# ------------------------------------------------------------------ scoping

def _redact(row: Dict[str, Any], scope: rbac.Scope) -> Dict[str, Any]:
    """Strip registration numbers for users who may not see them."""
    if scope.can_view_plates or not row:
        return row
    meta = row.get("metadata") or {}
    if not any(k in meta for k in PLATE_KEYS):
        return row
    row = dict(row)
    row["metadata"] = {k: v for k, v in meta.items() if k not in PLATE_KEYS}
    row["metadata"]["plate_redacted"] = True
    return row


def _scoped_events(database: Database, scope: rbac.Scope,
                   requested: Optional[List[str]] = None, **filters) -> List[Dict[str, Any]]:
    """``Database.query_events`` limited to the event types this user may
    read.  ``requested`` narrows further; an explicitly requested type the
    user may not read is refused rather than silently widened."""
    if requested is not None:
        hidden = [t for t in requested if not scope.sees_type(t)]
        if hidden and len(hidden) == len(requested):
            raise HTTPException(403, "this data belongs to another department")
    types = scope.restrict(requested)
    if types is not None and not types:
        return []
    rows = database.query_events(event_types=types, **filters)
    return [_redact(r, scope) for r in rows]


def _visible_issue(issue: Dict[str, Any], scope: rbac.Scope) -> bool:
    return scope.sees_type(issue.get("issue_type"))


def create_app(config=None) -> FastAPI:
    config = config or load_config()
    setup_logging(config.get("platform.log_level", "INFO"))
    log = get_logger("backend")

    database = Database(config.path("backend.database")).initialise()
    evidence_dir = config.path("backend.evidence_dir")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    frontend_dir = PROJECT_ROOT / "frontend"
    mobile_dir = PROJECT_ROOT / "mobile"
    runs_dir = (config.path("output.events_jsonl") or Path("outputs/runs/x")).parent

    # The platform layer: every store sits on the SAME database object.
    auth = AuthStore(database, config, log).initialise()
    recordings = RecordingIndex(database, config, log).initialise()
    workflow = WorkflowStore(database, config, recordings=recordings, auth=auth, log=log).initialise()
    fleet = FleetStore(database, config, log).initialise()

    app = FastAPI(
        title="AI Urban Intelligence Platform",
        version="2.0.0",
        description="Buses as mobile urban sensors -- one backend for the "
                    "Central Command web dashboard, the department dashboards "
                    "and the mobile Command & Field app.",
    )
    # Browser clients on another origin authenticate with the Bearer token
    # (never with the cookie -- credentials are not allowed cross-origin).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.db = database
    app.state.config = config
    app.state.auth = auth
    app.state.recordings = recordings
    app.state.workflow = workflow
    app.state.fleet = fleet

    try:
        stats = workflow.sync(force=True)
        log.info("workflow ready -- %s", stats or "no new issues")
    except Exception as exc:                                     # noqa: BLE001
        log.warning("initial workflow sync failed (%s) -- will retry on first request", exc)

    app.include_router(build_router())

    def need_plates(who: CurrentUser = Depends(current_user)) -> CurrentUser:
        if not who.scope.can_view_plates:
            raise HTTPException(403, "registration numbers are restricted to Public Safety")
        return who

    # ---------------------------------------------------------------- health

    @app.get("/health")
    def health(request: Request) -> Dict[str, Any]:
        who = optional_user(request)
        if who is None:
            return {"status": "ok", "auth_required": True}
        counts = {k: v for k, v in database.count_events().items() if who.scope.sees_type(k)}
        return {
            "status": "ok",
            "database": str(config.path("backend.database")),
            "events": sum(counts.values()),
            "events_by_type": counts,
            "issues": sum(1 for i in database.all_issues() if _visible_issue(i, who.scope)),
            "user": who.user["email"],
            "role": who.scope.role,
        }

    # ---------------------------------------------------------------- events

    def _ingest_allowed(request: Request) -> None:
        token = config.get("auth.ingest_token")
        if token:
            sent = request.headers.get("x-ingest-token") or ""
            if not hmac.compare_digest(str(sent), str(token)):
                raise HTTPException(401, "a valid X-Ingest-Token is required")
            return
        host = request.client.host if request.client else ""
        if host in _LOOPBACK:
            return
        who = optional_user(request)
        if who is None or not who.scope.is_central:
            raise HTTPException(403, "ingestion is only accepted from this machine, a "
                                     "Central Command session, or with auth.ingest_token")

    @app.post("/events")
    def ingest(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        _ingest_allowed(request)
        try:
            event = Event.from_dict(payload)
        except Exception as exc:                                 # noqa: BLE001
            raise HTTPException(400, f"invalid event: {exc}") from exc
        database.insert_event(event)
        return {"ok": True, "event_id": event.event_id}

    @app.post("/events/batch")
    def ingest_batch(payload: List[Dict[str, Any]], request: Request) -> Dict[str, Any]:
        _ingest_allowed(request)
        stored = 0
        for item in payload:
            try:
                database.insert_event(Event.from_dict(item))
                stored += 1
            except Exception as exc:                             # noqa: BLE001
                log.warning("skipped bad event: %s", exc)
        return {"ok": True, "stored": stored}

    @app.get("/events")
    def events(
        event_type: Optional[str] = None,
        source_module: Optional[str] = None,
        severity: Optional[str] = None,
        bus_id: Optional[str] = None,
        issue_id: Optional[str] = None,
        road_segment: Optional[str] = None,
        since: Optional[float] = None,
        include_duplicates: bool = True,
        limit: int = Query(100, le=2000),
        offset: int = 0,
        who: CurrentUser = Depends(current_user),
    ) -> List[Dict[str, Any]]:
        return _scoped_events(
            database, who.scope, [event_type] if event_type else None,
            source_module=source_module, severity=severity, bus_id=bus_id,
            issue_id=issue_id, road_segment=road_segment, since=since,
            include_duplicates=include_duplicates, limit=limit, offset=offset,
        )

    @app.get("/events/{event_id}")
    def event_detail(event_id: str, who: CurrentUser = Depends(current_user)) -> Dict[str, Any]:
        event = database.get_event(event_id)
        if event is None or not who.scope.sees_type(event.get("event_type")):
            raise HTTPException(404, "event not found")
        return _redact(event, who.scope)

    # ------------------------------------------------------- per-module views

    @app.get("/road-defects")
    def road_defects(limit: int = Query(100, le=2000), include_duplicates: bool = True,
                     who: CurrentUser = Depends(require_types("road_defect"))):
        return _scoped_events(database, who.scope, ["road_defect"], limit=limit,
                              include_duplicates=include_duplicates)

    @app.get("/traffic")
    def traffic(limit: int = Query(200, le=2000), congestion_only: bool = False,
                who: CurrentUser = Depends(require_types(*TRAFFIC_TYPES))):
        types = ["traffic_congestion", "traffic_bottleneck"] if congestion_only else TRAFFIC_TYPES
        return _scoped_events(database, who.scope, types, limit=limit)

    @app.get("/incidents")
    def incidents(
        limit: int = Query(200, le=2000),
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        min_confidence: float = 0.0,
        who: CurrentUser = Depends(require_types(*INCIDENT_TYPES)),
    ):
        """The incident feed the dashboard's Incidents page reads.

        ``event_type`` accepts one canonical type, or the shorthand
        ``possible`` for everything that is suspected rather than confirmed.
        """
        if event_type == "possible":
            requested = ["possible_collision", "abnormal_movement"]
        elif event_type:
            requested = [event_type]
        else:
            requested = INCIDENT_TYPES
        rows = _scoped_events(database, who.scope, requested, limit=limit)

        if status:
            rows = [
                r for r in rows
                if (r.get("metadata") or {}).get("status") == status
            ]
        if min_confidence:
            rows = [r for r in rows if (r.get("confidence") or 0) >= min_confidence]
        return rows

    @app.get("/incidents/{incident_id}")
    def incident_detail(incident_id: str,
                        who: CurrentUser = Depends(require_types(*INCIDENT_TYPES))):
        """One incident, fully assembled for the investigation view:
        state-machine timeline, evidence frames, the vehicle's other events
        and the ANPR result if identification was requested."""
        rows = _scoped_events(database, who.scope, INCIDENT_TYPES, limit=2000)
        record = next(
            (
                r for r in rows
                if (r.get("metadata") or {}).get("incident_id") == incident_id
                or r.get("event_id") == incident_id
            ),
            None,
        )
        if record is None:
            raise HTTPException(404, "incident not found")

        metadata = record.get("metadata") or {}
        track_id = metadata.get("vehicle_track_id") or record.get("track_id")

        related = [
            r for r in _scoped_events(database, who.scope, None, limit=2000)
            if r.get("track_id") == track_id
            and r.get("bus_id") == record.get("bus_id")
        ]
        identification = next(
            (r for r in related if r["event_type"] == "vehicle_identification"), None
        )
        vehicle = None
        if track_id is not None and who.scope.can_view_plates:
            vehicle = database.get_vehicle(track_id, record.get("bus_id"))
        link = database.query_one(
            "SELECT case_id FROM case_issues WHERE issue_id = ?", (record.get("issue_id") or "",)
        )
        return {
            "incident": record,
            "incident_id": incident_id,
            "case_id": link["case_id"] if link else None,
            "timeline": metadata.get("timeline") or [],
            "evidence": metadata.get("evidence_frames") or {},
            "vehicle": vehicle,
            "identification": identification,
            "related_events": related,
        }

    @app.get("/hit-and-run")
    def hit_and_run(limit: int = Query(100, le=1000),
                    who: CurrentUser = Depends(require_types("hit_and_run"))):
        """Hit-and-run alerts, newest first -- the dedicated page's feed."""
        return _scoped_events(database, who.scope, ["hit_and_run"], limit=limit)

    @app.get("/anpr")
    def anpr(limit: int = Query(100, le=1000), who: CurrentUser = Depends(need_plates)):
        """Every plate read, including the honest UNKNOWNs."""
        return database.query_events(event_type="vehicle_identification", limit=limit)

    @app.get("/vehicles")
    def vehicles(limit: int = Query(100, le=1000), who: CurrentUser = Depends(need_plates)):
        return database.list_vehicles(limit=limit)

    @app.get("/vehicles/{track_id}")
    def vehicle_detail(track_id: int, bus_id: Optional[str] = None,
                       who: CurrentUser = Depends(need_plates)):
        vehicle = database.get_vehicle(track_id, bus_id)
        if vehicle is None:
            raise HTTPException(404, "vehicle not found")
        related = _scoped_events(database, who.scope, None, limit=50)
        vehicle["events"] = [
            e for e in related
            if e.get("track_id") == track_id
            and (bus_id is None or e.get("bus_id") == bus_id)
        ]
        return vehicle

    # ------------------------------------------------------------- issues

    @app.get("/issues")
    def issues(
        status: Optional[str] = None,
        class_name: Optional[str] = None,
        department: Optional[str] = None,
        limit: int = Query(200, le=1000),
        who: CurrentUser = Depends(current_user),
    ):
        rows = database.list_issues(
            status=status, class_name=class_name, department=department, limit=1000,
        )
        return [i for i in rows if _visible_issue(i, who.scope)][: int(limit)]

    @app.get("/issues/{issue_id}")
    def issue_detail(issue_id: str, who: CurrentUser = Depends(current_user)):
        issue = database.get_issue(issue_id)
        if issue is None or not _visible_issue(issue, who.scope):
            raise HTTPException(404, "issue not found")
        issue["observations"] = database.issue_observations(issue_id)
        issue["events"] = _scoped_events(database, who.scope, None, issue_id=issue_id, limit=100)
        row = database.query_one("SELECT case_id FROM case_issues WHERE issue_id = ?", (issue_id,))
        issue["case_id"] = row["case_id"] if row else None
        return issue

    # ---------------------------------------------------------- analytics

    @app.get("/analytics")
    def analytics(who: CurrentUser = Depends(current_user)) -> Dict[str, Any]:
        scope = who.scope
        counts = {k: v for k, v in database.count_events().items() if scope.sees_type(k)}
        all_issues = [i for i in database.all_issues() if _visible_issue(i, scope)]

        by_status: Dict[str, int] = {}
        by_department: Dict[str, int] = {}
        by_band: Dict[str, int] = {}
        for issue in all_issues:
            by_status[issue.get("status") or "OPEN"] = by_status.get(issue.get("status") or "OPEN", 0) + 1
            department = issue.get("department") or "Unassigned"
            by_department[department] = by_department.get(department, 0) + 1
            band = issue.get("priority_band") or "LOW"
            by_band[band] = by_band.get(band, 0) + 1

        recent = _scoped_events(database, scope, None, limit=500)
        by_severity: Dict[str, int] = {}
        by_bus: Dict[str, int] = {}
        for event in recent:
            severity = event.get("severity") or "INFO"
            by_severity[severity] = by_severity.get(severity, 0) + 1
            bus = event.get("bus_id") or "unknown"
            by_bus[bus] = by_bus.get(bus, 0) + 1

        timeline = []
        if scope.sees_type("traffic_observation"):
            traffic_rows = database.query_events(
                event_type="traffic_observation", limit=200, order="asc"
            )
            timeline = [
                {
                    "timestamp": row.get("timestamp"),
                    "video_time_s": row.get("video_time_s"),
                    "vehicle_count": (row.get("metadata") or {}).get("vehicle_count"),
                    "density_percentage": (row.get("metadata") or {}).get("density_percentage"),
                    "traffic_level": (row.get("metadata") or {}).get("traffic_level"),
                    "average_speed_px_s": (row.get("metadata") or {}).get("average_speed_px_s"),
                }
                for row in traffic_rows
            ]

        return {
            "events_total": sum(counts.values()),
            "events_by_type": counts,
            "events_by_severity": by_severity,
            "events_by_bus": by_bus,
            "issues_total": len(all_issues),
            "issues_by_status": by_status,
            "issues_by_department": by_department,
            "issues_by_priority_band": by_band,
            "top_issues": sorted(
                all_issues, key=lambda i: (i.get("priority_score") or 0), reverse=True
            )[:10],
            "traffic_timeline": timeline,
            "segments": len(database.list_segments()),
        }

    @app.get("/heatmap")
    def heatmap(who: CurrentUser = Depends(current_user)) -> Dict[str, Any]:
        scope = who.scope
        segments = database.list_segments()
        if not scope.sees_any(TRAFFIC_TYPES):
            # Segment rows mix traffic samples with defect/incident counts;
            # a department without traffic visibility gets only its own count.
            keep = {"road_segment", "latitude", "longitude", "last_seen"}
            if scope.sees_type("road_defect"):
                keep.add("defect_count")
            if scope.sees_any(INCIDENT_TYPES):
                keep.add("incident_count")
            segments = [{k: v for k, v in s.items() if k in keep} for s in segments]
        issues = [i for i in database.all_issues() if _visible_issue(i, scope)]
        return {
            "segments": [s for s in segments if s.get("latitude") is not None],
            "issues": [
                {
                    "issue_id": i["issue_id"],
                    "class_name": i.get("class_name"),
                    "issue_type": i.get("issue_type"),
                    "status": i.get("status"),
                    "severity": i.get("severity"),
                    "latitude": i.get("latitude"),
                    "longitude": i.get("longitude"),
                    "priority_score": i.get("priority_score"),
                    "priority_band": i.get("priority_band"),
                    "department": i.get("department"),
                    "observation_count": i.get("observation_count"),
                    "bus_count": i.get("bus_count"),
                }
                for i in issues if i.get("latitude") is not None
            ],
        }

    @app.get("/runs")
    def runs(limit: int = Query(20, le=100), who: CurrentUser = Depends(current_user)):
        return database.list_runs(limit=limit)

    @app.get("/videos")
    def videos(who: CurrentUser = Depends(require_roles(rbac.CENTRAL_COMMAND))) -> Dict[str, Any]:
        """Annotated videos the pipeline produced, for the Live Analysis page."""
        found = []
        if runs_dir.exists():
            for path in runs_dir.glob("*.mp4"):
                found.append({
                    "name": path.name,
                    "url": f"/videos/{path.name}",
                    "size_mb": round(path.stat().st_size / 1e6, 2),
                    "modified": path.stat().st_mtime,
                })
        found.sort(key=lambda v: v["modified"], reverse=True)
        return {"videos": found, "directory": str(runs_dir)}

    def _may_stream(scope: rbac.Scope, name: str) -> bool:
        """Central Command opens every recording (Live Analysis).  Anyone
        else only a recording that is the linked evidence of an event type
        their department can read."""
        if scope.is_central:
            return True
        types = sorted(scope.visible_types or [])
        if not types:
            return False
        marks = ",".join("?" * len(types))
        row = database.query_one(
            f"SELECT 1 AS ok FROM evidence_links l JOIN events e ON e.event_id = l.event_id "
            f"WHERE l.video_name = ? AND e.event_type IN ({marks}) LIMIT 1",
            (name, *types),
        )
        return row is not None

    @app.get("/videos/{name}")
    def video_file(name: str, who: CurrentUser = Depends(current_user)):
        target = (runs_dir / name).resolve()
        if not str(target).startswith(str(runs_dir.resolve())):
            raise HTTPException(400, "invalid path")
        if not target.exists():
            raise HTTPException(404, "video not found")
        if not _may_stream(who.scope, name):
            raise HTTPException(403, "this recording is not evidence for your department")
        return FileResponse(target, media_type="video/mp4")

    # ------------------------------------------------------------ evidence

    @app.get("/evidence/{path:path}")
    def evidence(path: str, who: CurrentUser = Depends(current_user)):
        target = (evidence_dir / path).resolve()
        if not str(target).startswith(str(evidence_dir.resolve())):
            raise HTTPException(400, "invalid path")
        folder = path.replace("\\", "/").split("/", 1)[0]
        kind = rbac.EVIDENCE_FOLDER_TYPE.get(folder)
        allowed = who.scope.is_central or (kind is not None and who.scope.sees_type(kind))
        if kind == "vehicle_identification" and not who.scope.can_view_plates:
            allowed = False
        if not allowed:
            raise HTTPException(403, "this evidence belongs to another department")
        if not target.exists():
            raise HTTPException(404, "evidence not found")
        return FileResponse(target, headers={"Cache-Control": "private, max-age=3600"})

    # ------------------------------------------------------------ frontend

    if mobile_dir.exists():
        # Mobile Command & Field app -- same origin, same session cookie.
        @app.get("/mobile", include_in_schema=False)
        def mobile_root():
            return RedirectResponse("/mobile/", status_code=307)

        app.mount(
            "/mobile", StaticFiles(directory=str(mobile_dir), html=True), name="mobile"
        )

    if frontend_dir.exists():
        app.mount(
            "/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend"
        )

        @app.get("/")
        def index():
            return FileResponse(frontend_dir / "index.html",
                                headers={"Cache-Control": "no-cache"})
    else:                                                        # pragma: no cover
        @app.get("/")
        def index():
            return JSONResponse({"service": "AI Urban Intelligence Platform"})

    log.info("backend ready -- database %s", config.path("backend.database"))
    return app


app = create_app()
