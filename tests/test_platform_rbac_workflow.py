"""
Multi-department platform tests: auth, RBAC, deduplication, evidence and
the DETECTED -> VERIFIED -> ASSIGNED -> IN_PROGRESS -> RESOLVED workflow.

The database is rebuilt from the pipeline's own shipped output
(outputs/runs/events.jsonl) with scripts/import_events.py, exactly like
the demo does.  No model is run and nothing is re-detected.

Every test gets its own copy of that database, so the tests are
independent and never touch outputs/ in the project.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVENTS_JSONL = ROOT / "outputs" / "runs" / "events.jsonl"
PASSWORD = "Test-Only-Pa55word!"
DEMO = {
    "admin": "CENTRAL_COMMAND",
    "traffic": "TRAFFIC_DEPARTMENT",
    "roads": "ROADS_INFRASTRUCTURE",
    "safety": "PUBLIC_SAFETY",
    "transport": "TRANSPORT_AUTHORITY",
    "field": "FIELD_OFFICER",
}

pytestmark = pytest.mark.skipif(not EVENTS_JSONL.exists(), reason="shipped events.jsonl not present")


@pytest.fixture(scope="session")
def platform(tmp_path_factory):
    """Build one template DB, then hand out fresh copies + app instances."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    base = tmp_path_factory.mktemp("platform")
    template = base / "template.db"
    env = {
        "UIP_BACKEND__DATABASE": str(base / "import_time.db"),     # used by backend.app's module-level app
        "UIP_AUTH__DEMO_PASSWORD": PASSWORD,
        "UIP_AUTH__DEMO_CREDENTIALS_FILE": str(base / "creds.txt"),
        "UIP_WORKFLOW__FRAME_CACHE_DIR": str(base / "frames"),
    }
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        from scripts import import_events
        assert import_events.main(["--db", str(template), "--sync"]) == 0
        from core.config import load_config
        import backend.app as backend_app                       # noqa: F401  (module-level app uses the tmp DB)

        counter = {"n": 0}

        def make_app():
            counter["n"] += 1
            db = base / f"case_{counter['n']}.db"
            shutil.copyfile(template, db)
            config = load_config(overrides={"backend": {"database": str(db)}})
            return backend_app.create_app(config), db

        yield make_app
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture
def env(platform):
    """A fresh app + a login helper that returns one TestClient per user."""
    from fastapi.testclient import TestClient
    app, db = platform()
    clients = []

    def login(user: str, client_kind: str = "web") -> TestClient:
        c = TestClient(app)
        r = c.post("/auth/login", json={"login": f"{user}@urbanintel.local", "password": PASSWORD,
                                        "client": client_kind})
        assert r.status_code == 200, r.text
        clients.append(c)
        return c

    login.app, login.db = app, db
    yield login
    for c in clients:
        c.close()


def _cases(client, **params):
    params.setdefault("limit", 500)
    r = client.get("/cases", params=params)
    assert r.status_code == 200, r.text
    return r.json()["items"]


def _move(client, case_id, action, **extra):
    detail = client.get(f"/cases/{case_id}").json()
    return client.post(f"/cases/{case_id}/transition",
                       json={"action": action, "expected_status": detail["status"], **extra})


# --------------------------------------------------------------------------
# authentication
# --------------------------------------------------------------------------

def test_every_demo_account_logs_in_with_its_role(env):
    for user, role in DEMO.items():
        c = env(user)
        me = c.get("/auth/me").json()
        assert me["permissions"]["role"] == role
    # username (not only e-mail) works too
    from fastapi.testclient import TestClient
    c = TestClient(env.app)
    assert c.post("/auth/login", json={"login": "roads", "password": PASSWORD}).status_code == 200


def test_wrong_password_and_anonymous_are_rejected(env):
    from fastapi.testclient import TestClient
    anon = TestClient(env.app)
    assert anon.post("/auth/login", json={"login": "roads@urbanintel.local", "password": "nope"}).status_code == 401
    for path in ("/events", "/cases", "/incidents", "/road-defects", "/issues", "/analytics", "/notifications"):
        assert anon.get(path).status_code == 401, path


def test_passwords_are_hashed_never_stored_in_plaintext(env):
    env("admin")                                             # make sure the app initialised the DB
    con = sqlite3.connect(env.db)
    rows = con.execute("SELECT email, password_hash FROM users").fetchall()
    con.close()
    assert len(rows) >= 6
    for email, digest in rows:
        assert PASSWORD not in digest, email
        assert digest.startswith("pbkdf2"), digest[:20]


def test_logout_ends_the_session(env):
    c = env("roads")
    assert c.post("/auth/logout").status_code == 200
    assert c.get("/cases").status_code == 401


# --------------------------------------------------------------------------
# RBAC: what each department can see
# --------------------------------------------------------------------------

def test_central_command_sees_every_managed_event(env):
    central = _cases(env("admin"))
    departments = {c["owner_department"] for c in central}
    assert len(central) >= 2 and {"ROADS_INFRASTRUCTURE", "PUBLIC_SAFETY"} <= departments


def test_departments_only_see_their_own_events(env):
    central = _cases(env("admin"))
    by_dept = {}
    for c in central:
        by_dept.setdefault(c["owner_department"], set()).add(c["case_id"])
    roads = {c["case_id"] for c in _cases(env("roads"))}
    safety = {c["case_id"] for c in _cases(env("safety"))}
    field = {c["case_id"] for c in _cases(env("field"))}
    assert roads == by_dept["ROADS_INFRASTRUCTURE"]
    assert safety == by_dept["PUBLIC_SAFETY"]
    assert field == by_dept["ROADS_INFRASTRUCTURE"]          # the demo field officer works for Roads
    assert not roads & safety


def test_transport_is_read_only(env):
    transport = env("transport")
    for c in _cases(transport):
        detail = transport.get(f"/cases/{c['case_id']}").json()
        assert detail["can_act"] is False
        assert not any(a["enabled"] for a in detail["allowed_actions"])


@pytest.mark.parametrize("user,path", [
    ("roads", "/incidents"), ("roads", "/anpr"), ("roads", "/hit-and-run"), ("roads", "/users"),
    ("roads", "/fleet"), ("traffic", "/road-defects"), ("traffic", "/users"),
    ("safety", "/road-defects"), ("transport", "/users"), ("field", "/users"), ("field", "/anpr"),
])
def test_other_departments_pages_are_forbidden(env, user, path):
    assert env(user).get(path).status_code == 403


def test_url_tampering_on_a_case_id_does_not_leak(env):
    safety_case = next(c for c in _cases(env("admin")) if c["owner_department"] == "PUBLIC_SAFETY")
    roads = env("roads")
    cid = safety_case["case_id"]
    assert roads.get(f"/cases/{cid}").status_code == 404
    assert roads.get(f"/cases/{cid}/frame.jpg").status_code == 404
    assert roads.post(f"/cases/{cid}/transition", json={"action": "verify"}).status_code == 404


def test_evidence_folders_follow_department(env):
    roads, safety = env("roads"), env("safety")
    road_img = next((ROOT / "outputs" / "evidence" / "road_defect").glob("*.jpg"), None)
    ped_img = next((ROOT / "outputs" / "evidence" / "pedestrian_risk").glob("*.jpg"), None)
    if road_img:
        assert roads.get(f"/evidence/road_defect/{road_img.name}").status_code == 200
        assert safety.get(f"/evidence/road_defect/{road_img.name}").status_code == 403
    if ped_img:
        assert safety.get(f"/evidence/pedestrian_risk/{ped_img.name}").status_code == 200
        assert roads.get(f"/evidence/pedestrian_risk/{ped_img.name}").status_code == 403


def test_plates_are_redacted_for_departments_without_plate_access():
    from backend.app import _redact
    from backend.auth import scope_for
    row = {"event_type": "hit_and_run_candidate",
           "metadata": {"registration_number": "TN01AB1234", "plate_confidence": 0.9, "speed": 40}}
    roads = scope_for({"user_id": "u1", "role": "ROADS_INFRASTRUCTURE", "department": "ROADS_INFRASTRUCTURE"})
    safety = scope_for({"user_id": "u2", "role": "PUBLIC_SAFETY", "department": "PUBLIC_SAFETY"})
    hidden = _redact({**row, "metadata": dict(row["metadata"])}, roads)
    shown = _redact({**row, "metadata": dict(row["metadata"])}, safety)
    assert "registration_number" not in hidden["metadata"] and hidden["metadata"]["plate_redacted"] is True
    assert hidden["metadata"]["speed"] == 40
    assert shown["metadata"]["registration_number"] == "TN01AB1234"


# --------------------------------------------------------------------------
# deduplication + evidence
# --------------------------------------------------------------------------

def test_repeated_detections_become_one_managed_event(env):
    con = sqlite3.connect(env.db)
    raw_events = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    ped_events = con.execute("SELECT COUNT(*) FROM events WHERE event_type='pedestrian_vehicle_risk'").fetchone()[0]
    con.close()
    cases = _cases(env("admin"))
    ped_cases = [c for c in cases if c["event_type"] == "pedestrian_vehicle_risk"]
    assert len(cases) < raw_events
    assert 0 < len(ped_cases) < ped_events
    assert sum(c["observation_count"] for c in cases) >= len(cases)


def test_evidence_comes_from_the_annotated_recording(env):
    roads = env("roads")
    pothole = next(c for c in _cases(roads) if c.get("class_name") == "pothole")
    detail = roads.get(f"/cases/{pothole['case_id']}").json()
    ev = detail["evidence"]
    assert ev["available"] and ev["bounding_box"]
    video = ev["video"]
    assert video and video["name"].startswith("annotated_")
    assert (ROOT / "outputs" / "runs" / video["name"]).exists()
    frame = roads.get(ev["frame_url"])
    assert frame.status_code == 200 and frame.headers["content-type"].startswith("image/jpeg")
    assert frame.content[:2] == b"\xff\xd8"
    ranged = roads.get(video["url"], headers={"Range": "bytes=0-1023"})
    assert ranged.status_code == 206 and len(ranged.content) == 1024


# --------------------------------------------------------------------------
# workflow
# --------------------------------------------------------------------------

def test_full_lifecycle_mobile_to_web(env):
    roads = env("roads", "mobile")
    field = env("field", "mobile")
    central = env("admin", "web")
    cid = next(c["case_id"] for c in _cases(roads) if c["status"] == "DETECTED")

    assert _move(roads, cid, "verify").json()["status"] == "VERIFIED"
    people = roads.get(f"/cases/{cid}/assignees").json()
    officer = next(p for p in people if p["role"] == "FIELD_OFFICER")
    r = _move(roads, cid, "assign", assignee_user_id=officer["user_id"], note="patch today")
    assert r.status_code == 200 and r.json()["status"] == "ASSIGNED"

    inbox = field.get("/notifications").json()["items"]
    assert any(n["case_id"] == cid and n["kind"] == "assignment" for n in inbox)
    assert cid in {c["case_id"] for c in _cases(field, assigned="me")}

    assert _move(field, cid, "start").json()["status"] == "IN_PROGRESS"
    done = _move(field, cid, "resolve", note="filled and compacted").json()
    assert done["status"] == "RESOLVED" and done["resolution_note"] == "filled and compacted"

    seen = central.get(f"/cases/{cid}").json()
    assert seen["status"] == "RESOLVED"
    assert [a["action"] for a in seen["activity"]][-4:] == ["VERIFIED", "ASSIGNED", "IN_PROGRESS", "RESOLVED"]
    assert cid in {c["case_id"] for c in _cases(central, status="RESOLVED")}


def test_steps_cannot_be_skipped(env):
    roads = env("roads")
    cid = next(c["case_id"] for c in _cases(roads) if c["status"] == "DETECTED")
    r = _move(roads, cid, "resolve")
    assert r.status_code == 409
    assert roads.get(f"/cases/{cid}").json()["status"] == "DETECTED"


def test_stale_update_is_rejected(env):
    a, b = env("roads"), env("admin")
    cid = next(c["case_id"] for c in _cases(a) if c["status"] == "DETECTED")
    assert b.post(f"/cases/{cid}/transition", json={"action": "verify", "expected_status": "DETECTED"}).status_code == 200
    late = a.post(f"/cases/{cid}/transition", json={"action": "verify", "expected_status": "DETECTED"})
    assert late.status_code == 409


def test_field_officer_cannot_work_someone_elses_task(env):
    roads, field = env("roads"), env("field")
    cid = next(c["case_id"] for c in _cases(roads) if c["status"] == "DETECTED")
    _move(roads, cid, "verify")
    engineer = next(p for p in roads.get(f"/cases/{cid}/assignees").json() if p["role"] != "FIELD_OFFICER")
    assert _move(roads, cid, "assign", assignee_user_id=engineer["user_id"]).status_code == 200
    assert _move(field, cid, "start").status_code in (403, 409)


def test_original_endpoints_still_answer_for_central_command(env):
    central = env("admin")
    for path in ("/health", "/events", "/road-defects", "/traffic", "/incidents", "/hit-and-run", "/anpr",
                 "/vehicles", "/issues", "/analytics", "/heatmap", "/runs", "/videos"):
        assert central.get(path).status_code == 200, path
