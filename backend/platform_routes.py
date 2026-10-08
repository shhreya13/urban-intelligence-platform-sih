"""
backend.platform_routes
=======================

The multi-department endpoints added on top of the original API.  They are
mounted into the same FastAPI app, read and write the same SQLite database,
and are used identically by the web dashboard and the mobile app.

    /auth/*            login, logout, me, forgot / change password
    /users, /departments, /roles              user management
    /cases/*           managed events (EV-xxxx): list, detail, lifecycle
    /notifications/*   role-aware notifications
    /fleet/*           fleet status, routes, O/D, per-bus detail
    /map/layers        role-scoped GIS layers
    /sync/state        cheap change token so clients refresh only on change
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import rbac
from backend.auth import (
    SESSION_COOKIE, AuthStore, CurrentUser, client_ip, current_user, require_roles,
)

INCIDENT_TYPES = [
    "collision", "possible_collision", "hit_and_run",
    "rash_driving", "abnormal_movement", "pedestrian_vehicle_risk",
]
DAMAGE_CLASSES = {
    "damaged_road", "road_damage_other", "damaged_zebra_crossing",
    "damaged_traffic_signboard",
}
MISSING_CLASSES = {"missing_divider", "missing_zebra_crossing", "missing_traffic_signboard"}


# ------------------------------------------------------------------ bodies

class LoginBody(BaseModel):
    login: str = Field(..., description="email or username")
    password: str
    client: str = "web"


class ForgotBody(BaseModel):
    login: str


class ChangePasswordBody(BaseModel):
    current_password: str
    new_password: str


class NewUserBody(BaseModel):
    name: str
    email: str
    password: str
    role: str
    department: Optional[str] = None
    username: Optional[str] = None
    can_manage_users: bool = False


class UpdateUserBody(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    is_active: Optional[bool] = None
    can_manage_users: Optional[bool] = None


class ResetPasswordBody(BaseModel):
    password: str


class TransitionBody(BaseModel):
    action: str
    assignee_user_id: Optional[str] = None
    note: Optional[str] = None
    expected_status: Optional[str] = None


class ReadBody(BaseModel):
    ids: Optional[List[int]] = None


class BusBody(BaseModel):
    registration: Optional[str] = None
    route_id: Optional[str] = None
    depot: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
    cameras: Optional[List[str]] = None


def _require_json(request: Request) -> None:
    """Cookie-authenticated writes must be JSON: a cross-site HTML form
    cannot send ``application/json`` without a CORS preflight."""
    if request.headers.get("authorization"):
        return
    if "application/json" not in (request.headers.get("content-type") or ""):
        raise HTTPException(415, "send application/json")


def build_router() -> APIRouter:
    router = APIRouter()

    def stores(request: Request):
        s = request.app.state
        return s.db, s.auth, s.workflow, s.fleet, s.recordings

    # ================================================================ auth

    @router.post("/auth/login", tags=["auth"])
    def login(body: LoginBody, request: Request, response: Response):
        db, auth, workflow, *_ = stores(request)
        ip = client_ip(request)
        locked = auth.check_locked(ip, body.login)
        if locked:
            raise HTTPException(429, f"too many failed attempts -- try again in {locked}s")
        user = auth.authenticate(body.login, body.password)
        if user is None:
            auth.record_failure(ip, body.login)
            raise HTTPException(401, "email/username or password is incorrect")
        if not user["is_active"]:
            raise HTTPException(403, "this account is deactivated -- contact Central Command")
        auth.clear_failures(ip, body.login)
        client = "mobile" if body.client == "mobile" else "web"
        token = auth.create_session(user, client=client, user_agent=request.headers.get("user-agent", ""))
        ttl = int(auth.mobile_ttl_s if client == "mobile" else auth.web_ttl_s)
        response.set_cookie(
            SESSION_COOKIE, token, max_age=ttl, httponly=True, samesite="lax",
            secure=request.url.scheme == "https", path="/",
        )
        from backend.auth import scope_for
        return {"token": token, "user": AuthStore.public(user), "permissions": scope_for(user).as_dict()}

    @router.post("/auth/logout", tags=["auth"])
    def logout(request: Request, response: Response, who: CurrentUser = Depends(current_user)):
        request.app.state.auth.revoke(who.token)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    @router.get("/auth/me", tags=["auth"])
    def me(request: Request, who: CurrentUser = Depends(current_user)):
        return {"user": AuthStore.public(who.user), "permissions": who.scope.as_dict()}

    @router.post("/auth/forgot-password", tags=["auth"])
    def forgot(body: ForgotBody, request: Request):
        db, auth, workflow, *_ = stores(request)
        ip = client_ip(request)
        if auth.check_locked(ip, "forgot:" + body.login):
            raise HTTPException(429, "too many requests -- try again shortly")
        auth.record_failure(ip, "forgot:" + body.login)
        user = auth.find_by_login(body.login)
        if user is not None and user["is_active"]:
            workflow.notify_department(
                rbac.CENTRAL_COMMAND, kind="password_reset", severity="MEDIUM",
                title="Password reset requested",
                body=f"{user['name']} ({user['email']}, {rbac.ROLE_LABEL.get(user['role'])}) "
                     f"asked for a password reset. Reset it in Users / Departments.",
            )
        return {"ok": True, "message": "If the account exists, Central Command has been asked to reset "
                                       "the password. They will give you a new one."}

    @router.post("/auth/change-password", tags=["auth"])
    def change_password(body: ChangePasswordBody, request: Request, response: Response,
                        who: CurrentUser = Depends(current_user)):
        _require_json(request)
        auth: AuthStore = request.app.state.auth
        from backend.auth import verify_password
        if not verify_password(body.current_password, who.user["password_hash"]):
            raise HTTPException(403, "current password is incorrect")
        auth.set_password(who.id, body.new_password)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True, "message": "Password changed -- log in again"}

    # =============================================================== users

    def _manager(who: CurrentUser) -> None:
        if not who.scope.can_manage_users:
            raise HTTPException(403, "you are not authorised to manage users")

    def _may_manage(who: CurrentUser, department: str, role: str) -> None:
        if who.scope.is_central:
            return
        if department != who.scope.department or role == rbac.CENTRAL_COMMAND:
            raise HTTPException(403, "you can only manage users in your own department")

    @router.get("/users", tags=["users"])
    def list_users(request: Request, department: Optional[str] = None,
                   who: CurrentUser = Depends(current_user)):
        _manager(who)
        auth: AuthStore = request.app.state.auth
        if not who.scope.is_central:
            department = who.scope.department
        return [AuthStore.public(u) for u in auth.list_users(department)]

    @router.post("/users", tags=["users"])
    def create_user(body: NewUserBody, request: Request, who: CurrentUser = Depends(current_user)):
        _require_json(request)
        _manager(who)
        department = body.department or (body.role if body.role != rbac.FIELD_OFFICER else None)
        if not department:
            raise HTTPException(422, "department is required for a field officer")
        _may_manage(who, department, body.role)
        user = request.app.state.auth.create_user(
            name=body.name, email=body.email, password=body.password, role=body.role,
            department=department, username=body.username, created_by=who.id,
            can_manage_users=body.can_manage_users and who.scope.is_central,
        )
        return AuthStore.public(user)

    @router.patch("/users/{user_id}", tags=["users"])
    def update_user(user_id: str, body: UpdateUserBody, request: Request,
                    who: CurrentUser = Depends(current_user)):
        _require_json(request)
        _manager(who)
        auth: AuthStore = request.app.state.auth
        target = auth.get_user(user_id)
        if target is None:
            raise HTTPException(404, "user not found")
        _may_manage(who, target["department"], target["role"])
        if body.department or body.role:
            _may_manage(who, body.department or target["department"], body.role or target["role"])
        if user_id == who.id and body.is_active is False:
            raise HTTPException(422, "you cannot deactivate your own account")
        if body.can_manage_users is not None and not who.scope.is_central:
            raise HTTPException(403, "only Central Command can grant user management")
        role = body.role
        department = body.department
        if role and role != rbac.FIELD_OFFICER and not department:
            department = role
        user = auth.update_user(
            user_id, name=body.name, role=role, department=department,
            is_active=body.is_active, can_manage_users=body.can_manage_users,
        )
        return AuthStore.public(user)

    @router.post("/users/{user_id}/reset-password", tags=["users"])
    def reset_password(user_id: str, body: ResetPasswordBody, request: Request,
                       who: CurrentUser = Depends(current_user)):
        _require_json(request)
        _manager(who)
        auth: AuthStore = request.app.state.auth
        target = auth.get_user(user_id)
        if target is None:
            raise HTTPException(404, "user not found")
        _may_manage(who, target["department"], target["role"])
        auth.set_password(user_id, body.password)
        return {"ok": True}

    @router.get("/departments", tags=["users"])
    def departments(request: Request, who: CurrentUser = Depends(current_user)):
        db = request.app.state.db
        users = db.query("SELECT department, role, is_active FROM users")
        cases = db.query("SELECT owner_department, status FROM cases")
        out = []
        for dept in rbac.DEPARTMENTS:
            out.append({
                "department": dept,
                "label": rbac.DEPARTMENT_LABEL[dept],
                "users": sum(1 for u in users if u["department"] == dept),
                "active_users": sum(1 for u in users if u["department"] == dept and u["is_active"]),
                "field_officers": sum(1 for u in users if u["department"] == dept and u["role"] == rbac.FIELD_OFFICER),
                "open_events": sum(1 for c in cases if c["owner_department"] == dept and c["status"] != "RESOLVED"),
                "visible_event_types": sorted(rbac.VISIBLE_TYPES.get(dept, rbac.ALL_TYPES)),
            })
        return out

    @router.get("/roles", tags=["users"])
    def roles(who: CurrentUser = Depends(current_user)):
        return [{"role": r, "label": rbac.ROLE_LABEL[r],
                 "departments": rbac.OPERATING_DEPARTMENTS if r == rbac.FIELD_OFFICER else [r],
                 "pages": rbac.PAGES[r]} for r in rbac.ROLES]

    # ======================================================== managed events

    def _case_or_404(request: Request, case_id: str, who: CurrentUser) -> Dict[str, Any]:
        case = request.app.state.workflow.get_case(case_id)
        if case is None or not who.scope.sees_case(case):
            raise HTTPException(404, "event not found")
        return case

    def _user_name(db, user_id: Optional[str]) -> Optional[str]:
        if not user_id:
            return None
        row = db.query_one("SELECT name FROM users WHERE user_id = ?", (user_id,))
        return row["name"] if row else user_id

    def _evidence(request: Request, case: Dict[str, Any], who: CurrentUser) -> Dict[str, Any]:
        db, _, _, _, recordings = stores(request)
        event = db.get_event(case["primary_event_id"]) if case.get("primary_event_id") else None
        if event is None:
            return {"available": False}
        meta = event.get("metadata") or {}
        link = recordings.resolve(event) if recordings else None
        frames = meta.get("evidence_frames") or {}
        labelled = [("frame_before", "Before"), ("incident_frame", "Impact"),
                    ("frame_after", "After"), ("vehicle_crop", "Vehicle")]
        incident_frames = [
            {"label": label, "url": f"/evidence/{frames[key]}"} for key, label in labelled if frames.get(key)
        ]
        clip = frames.get("clip") or event.get("evidence_video")
        plate = None
        if who.scope.can_view_plates and case["owner_department"] == rbac.PUBLIC_SAFETY:
            ident = None
            for row in db.query_events(event_type="vehicle_identification", limit=500):
                m = row.get("metadata") or {}
                if m.get("triggered_by_event_id") == event["event_id"] or (
                    row.get("track_id") is not None and row.get("track_id") == event.get("track_id")
                    and row.get("bus_id") == event.get("bus_id")
                ):
                    ident = row
                    break
            if ident is not None:
                m = ident.get("metadata") or {}
                plate = {
                    "registration_number": m.get("registration_number") or "UNKNOWN",
                    "confidence": m.get("plate_confidence"),
                    "unverified": bool(m.get("unverified_partial_read")),
                    "image": f"/evidence/{ident['evidence_image']}" if ident.get("evidence_image") else None,
                }
            elif meta.get("registration_number"):
                plate = {"registration_number": meta["registration_number"],
                         "confidence": meta.get("plate_confidence"), "unverified": False, "image": None}
        return {
            "available": True,
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "class_name": event.get("class_name"),
            "confidence": event.get("confidence"),
            "timestamp": event.get("timestamp"),
            "bus_id": event.get("bus_id"),
            "camera_id": event.get("camera_id"),
            "gps": event.get("gps"),
            "road_segment": event.get("road_segment"),
            "frame_number": event.get("frame_number"),
            "video_time_s": event.get("video_time_s"),
            "bounding_box": event.get("bounding_box"),
            "detection_image": f"/evidence/{event['evidence_image']}" if event.get("evidence_image") else None,
            "frame_url": f"/cases/{case['case_id']}/frame.jpg" if link else None,
            "video": link,
            "incident_frames": incident_frames,
            "clip": f"/evidence/{clip}" if clip else None,
            "synthetic": bool(meta.get("synthetic")),
            "state": meta.get("state"),
            "plate": plate,
            "model": meta.get("model") or meta.get("detector"),
        }

    def case_detail(request: Request, case: Dict[str, Any], who: CurrentUser) -> Dict[str, Any]:
        db, _, workflow, *_ = stores(request)
        detail = workflow._decorate(case)                       # noqa: SLF001
        detail["assignee_name"] = _user_name(db, case.get("assignee_user_id"))
        detail["verified_by_name"] = _user_name(db, case.get("verified_by"))
        detail["resolved_by_name"] = _user_name(db, case.get("resolved_by"))
        detail["activity"] = workflow.activity(case["case_id"])
        detail["allowed_actions"] = workflow.allowed_actions(case, who)
        detail["can_act"] = any(a["enabled"] for a in detail["allowed_actions"])
        detail["evidence"] = _evidence(request, case, who)
        detail["observations"] = workflow.observations(case["case_id"])
        detail["issues"] = [r["issue_id"] for r in db.query(
            "SELECT issue_id FROM case_issues WHERE case_id = ?", (case["case_id"],))]
        return detail

    @router.get("/cases", tags=["events"])
    def list_cases(
        request: Request,
        status: Optional[str] = None, open_only: bool = False,
        department: Optional[str] = None, event_type: Optional[str] = None,
        class_name: Optional[str] = None, bus_id: Optional[str] = None,
        assigned: Optional[str] = None, severity: Optional[str] = None,
        q: Optional[str] = None, sort: str = "priority",
        near: Optional[str] = None,
        limit: int = Query(100, le=2000), offset: int = 0,
        who: CurrentUser = Depends(current_user),
    ):
        workflow = request.app.state.workflow
        workflow.sync()
        point = None
        if near:
            try:
                lat, lon = [float(v) for v in near.split(",")[:2]]
                point = (lat, lon)
            except ValueError:
                raise HTTPException(422, "near must be 'lat,lon'")
        return workflow.list_cases(
            who.scope, status=status, open_only=open_only, department=department,
            event_type=event_type, class_name=class_name, bus_id=bus_id, assigned=assigned,
            severity=severity, q=q, sort=sort, near=point, limit=limit, offset=offset,
            viewer_id=who.id,
        )

    @router.get("/cases/summary", tags=["events"])
    def cases_summary(request: Request, who: CurrentUser = Depends(current_user)):
        workflow = request.app.state.workflow
        workflow.sync()
        return workflow.summary(who.scope, viewer_id=who.id)

    @router.get("/cases/lookup", tags=["events"])
    def case_lookup(request: Request, issue_id: Optional[str] = None,
                    event_id: Optional[str] = None, who: CurrentUser = Depends(current_user)):
        """Which managed event (EV-xxxx) a pipeline issue or raw event belongs
        to -- lets the original pages (Road Assets, Incidents, Persistent
        Issues) open the shared event workflow."""
        db = request.app.state.db
        request.app.state.workflow.sync()
        if not issue_id and event_id:
            row = db.query_one("SELECT issue_id FROM events WHERE event_id = ?", (event_id,))
            issue_id = row["issue_id"] if row else None
        if not issue_id:
            raise HTTPException(404, "no managed event for this record")
        link = db.query_one("SELECT case_id FROM case_issues WHERE issue_id = ?", (issue_id,))
        case = request.app.state.workflow.get_case(link["case_id"]) if link else None
        if case is None or not who.scope.sees_case(case):
            raise HTTPException(404, "no managed event for this record")
        return {"case_id": case["case_id"], "status": case["status"], "issue_id": issue_id}

    @router.get("/cases/{case_id}", tags=["events"])
    def case(case_id: str, request: Request, who: CurrentUser = Depends(current_user)):
        return case_detail(request, _case_or_404(request, case_id, who), who)

    @router.get("/cases/{case_id}/assignees", tags=["events"])
    def case_assignees(case_id: str, request: Request, who: CurrentUser = Depends(current_user)):
        case = _case_or_404(request, case_id, who)
        if not who.scope.owns_case(case):
            raise HTTPException(403, "only the responsible department can assign this event")
        people = request.app.state.workflow.assignees(case)
        if who.scope.is_field:
            people = [p for p in people if p["user_id"] == who.id]
        return people

    @router.post("/cases/{case_id}/transition", tags=["events"])
    def transition(case_id: str, body: TransitionBody, request: Request,
                   who: CurrentUser = Depends(current_user)):
        _require_json(request)
        workflow = request.app.state.workflow
        workflow.transition(
            case_id, who, body.action.lower(), assignee_user_id=body.assignee_user_id,
            note=body.note, expected_status=body.expected_status,
        )
        return case_detail(request, workflow.get_case(case_id), who)

    @router.get("/cases/{case_id}/frame.jpg", tags=["events"])
    def case_frame(case_id: str, request: Request, who: CurrentUser = Depends(current_user)):
        case = _case_or_404(request, case_id, who)
        db, _, _, _, recordings = stores(request)
        event = db.get_event(case["primary_event_id"]) if case.get("primary_event_id") else None
        link = recordings.resolve(event) if event else None
        if not link:
            raise HTTPException(404, "no annotated recording is linked to this event")
        path = recordings.frame_file(link["name"], link["frame_index"])
        if path is None:
            raise HTTPException(404, "the frame could not be read from the recording")
        return FileResponse(path, media_type="image/jpeg",
                            headers={"Cache-Control": "private, max-age=3600"})

    # ======================================================== notifications

    @router.get("/notifications", tags=["notifications"])
    def notifications(request: Request, unread_only: bool = False,
                      limit: int = Query(50, le=300), department: Optional[str] = None,
                      who: CurrentUser = Depends(current_user)):
        workflow = request.app.state.workflow
        workflow.sync()
        if department and not who.scope.is_central:
            department = None
        return workflow.notifications(who, unread_only=unread_only, limit=limit, department=department)

    @router.get("/notifications/unread-count", tags=["notifications"])
    def unread(request: Request, who: CurrentUser = Depends(current_user)):
        workflow = request.app.state.workflow
        workflow.sync()
        return {"unread": workflow.unread_count(who)}

    @router.post("/notifications/read", tags=["notifications"])
    def mark_read(body: ReadBody, request: Request, who: CurrentUser = Depends(current_user)):
        _require_json(request)
        count = request.app.state.workflow.mark_read(who, body.ids)
        return {"ok": True, "marked": count, "unread": request.app.state.workflow.unread_count(who)}

    # ================================================================ fleet

    fleet_roles = require_roles(rbac.CENTRAL_COMMAND, rbac.TRANSPORT_AUTHORITY)

    @router.get("/fleet", tags=["fleet"])
    def fleet_overview(request: Request, who: CurrentUser = Depends(fleet_roles)):
        data = request.app.state.fleet.overview(who.scope)
        data["routes"] = request.app.state.fleet.routes()
        data["service_statuses"] = ["IN_SERVICE", "MAINTENANCE", "OUT_OF_SERVICE"]
        return data

    @router.get("/fleet/routes", tags=["fleet"])
    def fleet_routes(request: Request, who: CurrentUser = Depends(fleet_roles)):
        return request.app.state.fleet.route_performance()

    @router.get("/fleet/od", tags=["fleet"])
    def fleet_od(request: Request, who: CurrentUser = Depends(fleet_roles)):
        return request.app.state.fleet.od_matrix()

    @router.get("/fleet/{bus_id}", tags=["fleet"])
    def fleet_bus(bus_id: str, request: Request, who: CurrentUser = Depends(fleet_roles)):
        detail = request.app.state.fleet.bus_detail(bus_id, who.scope)
        if detail is None:
            raise HTTPException(404, "bus not found")
        return detail

    @router.patch("/fleet/{bus_id}", tags=["fleet"])
    def fleet_update(bus_id: str, body: BusBody, request: Request, who: CurrentUser = Depends(fleet_roles)):
        _require_json(request)
        try:
            return request.app.state.fleet.update_bus(bus_id, body.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # ============================================================= GIS map

    @router.get("/map/layers", tags=["map"])
    def map_layers(request: Request, layers: Optional[str] = None,
                   department: Optional[str] = None,
                   who: CurrentUser = Depends(current_user)):
        db, _, workflow, fleet, _ = stores(request)
        scope = who.scope
        allowed = scope.map_layers
        wanted = [l for l in (layers.split(",") if layers else scope.default_layers) if l in allowed]
        # Central Command may narrow the whole map to one department.
        dept = department if (scope.is_central and department and department != "ALL") else None
        if dept:
            relevant = {
                rbac.ROADS_INFRASTRUCTURE: {"potholes", "damaged_roads", "waterlogging", "road_assets", "persistent_issues"},
                rbac.TRAFFIC_DEPARTMENT: {"traffic", "incidents"},
                rbac.PUBLIC_SAFETY: {"incidents", "anpr", "persistent_issues"},
                rbac.TRANSPORT_AUTHORITY: {"buses", "traffic"},
            }.get(dept, set(wanted))
            wanted = [l for l in wanted if l in relevant]
        workflow.sync()
        out: Dict[str, List[Dict[str, Any]]] = {}
        cases = workflow.list_cases(scope, limit=2000, department=dept)["items"] if set(wanted) & {
            "potholes", "damaged_roads", "waterlogging", "incidents", "road_assets", "traffic"} else []

        def marker(c: Dict[str, Any], kind: str) -> Optional[Dict[str, Any]]:
            if c.get("latitude") is None:
                return None
            return {
                "kind": kind, "lat": c["latitude"], "lon": c["longitude"],
                "case_id": c["case_id"], "title": c["title"], "status": c["status"],
                "severity": c.get("severity"), "subtitle": f"{c['case_id']} - {c.get('road_segment') or ''}",
                "observations": c.get("observation_count"), "bus_id": c.get("bus_id"),
            }

        def take(kind: str, predicate) -> List[Dict[str, Any]]:
            return [m for m in (marker(c, kind) for c in cases if predicate(c)) if m]

        if "potholes" in wanted:
            out["potholes"] = take("potholes", lambda c: c["class_name"] == "pothole")
        if "damaged_roads" in wanted:
            out["damaged_roads"] = take("damaged_roads", lambda c: c["class_name"] in DAMAGE_CLASSES)
        if "waterlogging" in wanted:
            out["waterlogging"] = take("waterlogging", lambda c: c["class_name"] == "waterlogging")
        if "incidents" in wanted:
            out["incidents"] = take("incidents", lambda c: c["event_type"] in INCIDENT_TYPES)
        if "road_assets" in wanted:
            assets = take("road_assets", lambda c: c["class_name"] in MISSING_CLASSES)
            for issue in db.query(
                "SELECT issue_id, class_name, latitude, longitude, road_segment, observation_count, "
                "last_seen FROM issues WHERE issue_type = 'road_defect' AND class_name IN "
                "('zebra_crossing','divider','traffic_signboard','road_repair') AND latitude IS NOT NULL"
            ):
                assets.append({
                    "kind": "road_assets", "lat": issue["latitude"], "lon": issue["longitude"],
                    "case_id": None, "issue_id": issue["issue_id"], "status": "PRESENT",
                    "severity": "INFO",
                    "title": (issue["class_name"] or "").replace("_", " ").capitalize() + " (present)",
                    "subtitle": f"{issue['issue_id']} - seen {issue['observation_count']}x",
                })
            out["road_assets"] = assets
        if "traffic" in wanted:
            segments = [s for s in db.list_segments() if s.get("latitude") is not None and s.get("samples")]
            out["traffic"] = [{
                "kind": "traffic", "lat": s["latitude"], "lon": s["longitude"],
                "case_id": None, "segment": s["road_segment"],
                "level": s.get("max_traffic_level") or "LOW",
                "congested_ratio": s.get("congested_ratio"),
                "avg_vehicles": s.get("avg_vehicle_count"),
                "title": f"{s['road_segment']}",
                "subtitle": f"peak {s.get('max_traffic_level') or 'LOW'} - avg {s.get('avg_vehicle_count')} vehicles",
                "severity": {"CRITICAL": "CRITICAL", "HIGH": "HIGH", "MEDIUM": "MEDIUM"}.get(
                    s.get("max_traffic_level") or "", "LOW"),
            } for s in segments]
            out["traffic"] += take("traffic", lambda c: c["event_type"] in ("traffic_congestion", "traffic_bottleneck"))
        if "anpr" in wanted and scope.can_view_plates:
            reads = db.query_events(event_type="vehicle_identification", limit=500)
            out["anpr"] = [{
                "kind": "anpr", "lat": r["gps"]["latitude"], "lon": r["gps"]["longitude"],
                "case_id": None, "event_id": r["event_id"], "severity": r.get("severity"),
                "status": "READ",
                "title": (r.get("metadata") or {}).get("registration_number") or "UNKNOWN",
                "subtitle": f"track #{r.get('track_id')} - {(r.get('metadata') or {}).get('triggered_by_event_type') or ''}",
            } for r in reads if r.get("gps")]
        if "persistent_issues" in wanted:
            rows = db.query(
                "SELECT * FROM issues WHERE latitude IS NOT NULL AND (observation_count >= 2 "
                "OR bus_count >= 2 OR reopened_count > 0)")
            link = {r["issue_id"]: r["case_id"] for r in db.query("SELECT issue_id, case_id FROM case_issues")}
            out["persistent_issues"] = [{
                "kind": "persistent_issues", "lat": i["latitude"], "lon": i["longitude"],
                "case_id": link.get(i["issue_id"]), "issue_id": i["issue_id"],
                "severity": i.get("severity"), "status": i.get("status"),
                "title": f"{i['issue_id']} ({(i.get('class_name') or i.get('issue_type') or '').replace('_', ' ')})",
                "subtitle": f"seen {i['observation_count']}x by {i['bus_count']} bus(es)",
            } for i in rows if scope.sees_type(i.get("issue_type"))
                and rbac.owner_department(i.get("issue_type"), i.get("class_name")) is not None
                and (dept is None or rbac.owner_department(i.get("issue_type"), i.get("class_name")) == dept)]
        if "buses" in wanted:
            overview = fleet.overview(scope)
            out["buses"] = [{
                "kind": "buses", "lat": b["last_position"]["latitude"], "lon": b["last_position"]["longitude"],
                "case_id": None, "bus_id": b["bus_id"], "status": b["state"], "severity": "INFO",
                "title": b["bus_id"], "subtitle": f"{b['state'].replace('_', ' ').lower()} - last report {b['last_report']}",
            } for b in overview["buses"] if b.get("last_position")]
        return {"layers": out, "allowed": allowed, "defaults": scope.default_layers,
                "requested": wanted, "department": dept or "ALL"}

    # ================================================================= sync

    @router.get("/sync/state", tags=["sync"])
    def sync_state(request: Request, who: CurrentUser = Depends(current_user)):
        workflow = request.app.state.workflow
        workflow.sync()
        return {"token": workflow.state_token(), "unread": workflow.unread_count(who)}

    return router
