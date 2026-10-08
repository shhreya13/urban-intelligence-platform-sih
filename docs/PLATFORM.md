# Multi-department platform

This document describes the multi-department layer added on top of the
existing pipeline, backend and dashboard: Central Command, four department
dashboards, the mobile Command & Field app, shared login and role-based
access, and the event workflow

    DETECTED -> VERIFIED -> ASSIGNED -> IN_PROGRESS -> RESOLVED

Nothing was rebuilt. Every original page, endpoint, module and test is
still there; the new layer sits beside them and reuses the same database.

Contents: 1 Quick start, 2 Demo accounts, 3 What was built, 4 How RBAC
works, 5 The event workflow and deduplication, 6 Evidence, 7 How web and
mobile share data, 8 API endpoints, 9 Database schema, 10 Files added and
changed, 11 Configuration, 12 Tests, 13 Assumptions and limitations.

---

## 1. Quick start

No model is needed to demo the platform. The pipeline's own shipped output
(`outputs/runs/events.jsonl`, the annotated recordings in `outputs/runs/`
and the crops in `outputs/evidence/`) is loaded as-is.

```bash
python -m pip install -r requirements.txt

./demo.sh --dashboard          # Linux / macOS
demo.bat --dashboard           # Windows
```

That imports the shipped events, builds the managed events, and starts the
server with the demo password `UrbanIntel@2026`.

| App | Address |
|---|---|
| Web (Central Command + departments) | http://127.0.0.1:8000/ |
| Mobile Command & Field app | http://127.0.0.1:8000/mobile/ |
| API docs (Swagger) | http://127.0.0.1:8000/docs |

The same thing by hand:

```bash
export UIP_AUTH__DEMO_PASSWORD='UrbanIntel@2026'     # Windows: set UIP_AUTH__DEMO_PASSWORD=UrbanIntel@2026
python scripts/import_events.py --sync               # safe to repeat: same counts every time
python run_backend.py
```

**On a phone.** Start the server with `python run_backend.py --host 0.0.0.0`,
put the phone on the same Wi-Fi and open `http://<computer's LAN IP>:8000/mobile/`.
In the browser menu choose "Add to Home screen" to install it as an app
(it has a manifest, icons and a service worker). The "Nearby" screen uses
GPS, which browsers only allow over HTTPS or on `localhost`; over plain
HTTP on the LAN everything else works and Nearby says location is
unavailable.

**The original pipeline demo is unchanged:** `./demo.sh`, `./demo.sh --multi-bus`
and `python run_pipeline.py` still run the models on a video. When the
backend starts afterwards it turns the new detections into managed events
automatically.

---

## 2. Demo accounts

| Account | Role | Lands on |
|---|---|---|
| admin@urbanintel.local | Central Command | Command dashboard (everything) |
| traffic@urbanintel.local | Traffic Department | Traffic home |
| roads@urbanintel.local | Roads & Infrastructure | Roads home |
| safety@urbanintel.local | Public Safety | Safety home |
| transport@urbanintel.local | Transport Authority | Fleet home |
| field@urbanintel.local | Field Officer (Roads & Infrastructure) | My tasks |

The username (the part before `@`) also works as the login.

Password: whatever `UIP_AUTH__DEMO_PASSWORD` is set to when the database is
first created (`demo.sh` / `demo.bat` use `UrbanIntel@2026`). If the variable
is not set, a random password is generated and written to
`outputs/demo_credentials.txt`. **No password appears in code or config.**
Demo accounts are only created in an empty database and only while
`auth.seed_demo_users` is true; turn that off for any real deployment and
create accounts with `scripts/manage_users.py` or the Users page.

---

## 3. What was built

### Web dashboard (`frontend/index.html`)

One login, then each role gets its own sidebar, home page and data. The
original ten pages (Dashboard, Live, Map, Incidents, Hit-and-run, ANPR,
Traffic, Road assets, Issues, Analytics) are unchanged in behaviour and
remain Central Command's; department users see the subset their role
allows.

New pages:

| Page | Who |
|---|---|
| Event Management (filter by status, department, type, assignee; lifecycle actions) | everyone except Transport acts; Transport reads |
| Evidence gallery (frames, crops, recordings of visible events) | all but Transport |
| Notifications | everyone |
| Fleet (buses, trail map, bus detail) | Central, Transport |
| Users & Departments (create, deactivate, reset password) | Central, and department users granted user management |
| Traffic / Roads / Safety / Transport / Field home | that role |

Every event opens in a right-hand drawer: the facts, evidence tabs
(annotated frame, evidence video, detection crop, incident frames, clip),
the five-step lifecycle bar, the action buttons the viewer is allowed to
press, assign and resolve forms, and the activity log. Original pages link
into it: an issue on Issues or Road assets, or an incident on Incidents,
has an "Open managed event" link.

The GIS map shows only the layers the role may use (buses, potholes,
damaged roads, waterlogging, traffic, incidents, ANPR, road assets,
persistent issues) with the role's defaults switched on.

### Mobile app (`mobile/`)

A mobile-first single-page app served by the same backend, installable as
a PWA. The login screen is exactly as specified (URBAN INTELLIGENCE /
COMMAND & FIELD APP / Email-Username / Password / LOGIN / Forgot password).

Bottom navigation, five tabs per role, 63 px minimum touch height, alert
badge on the Alerts tab:

| Role | Tabs |
|---|---|
| Central Command | Command, Map, Events, Alerts, Me |
| Traffic | Traffic, Map, Tasks, Alerts, Me |
| Roads & Infrastructure | Roads, Nearby, Issues, Alerts, Me |
| Public Safety | Safety, Map, Cases, Alerts, Me |
| Transport Authority | Fleet, Routes, Map, Alerts, Me |
| Field Officer | Tasks, Nearby, Events, Alerts, Me |

The event screen shows category, title, ID, type, location, bus, camera,
confidence, detection time, sightings and assignee; a large status label
with the five-step progress bar; big VERIFY / ASSIGN / START WORK / MARK
RESOLVED buttons (disabled with a reason when the viewer may not press
them); assign and resolve as bottom sheets; and a full-screen evidence
viewer for the annotated frame, the evidence video (with JUMP TO EVENT
FRAME) and the crop.

---

## 4. How RBAC works

All rules live in one file, `backend/rbac.py`. The browser never decides
what data a user gets: every endpoint filters on the server, and both apps
read the user's permissions from `GET /auth/me` only to decide which
navigation to draw.

**Role vs department.** A user has a role (six values) and a department
(five values). For five roles they are the same. A Field Officer belongs to
one of the four operating departments.

**Visibility: which event types a department may read.**

| Department | Sees |
|---|---|
| Central Command | everything |
| Traffic | traffic observation / congestion / bottleneck, rash driving, collisions |
| Roads & Infrastructure | road defects |
| Public Safety | collisions, hit-and-run, abnormal movement, pedestrian risk, rash driving, ANPR |
| Transport Authority | traffic types and safety incidents except pedestrian risk, read-only, plates redacted |

**Ownership: which department acts.** Road defects belong to Roads;
congestion, bottlenecks and rash driving to Traffic; collisions,
hit-and-run, abnormal movement and pedestrian risk to Public Safety.
Healthy infrastructure (zebra crossing present, divider present, sign
present, road repaired) is an asset inventory, never an event.

**Managed events.** Central Command sees and may act on every event. A
department sees events it owns plus events of types it may read; it may act
only on events it owns. A Field Officer sees their department's events and
anything assigned to them, and may start or resolve only what is assigned
to them. Transport can read but never act.

**Plates.** Registration numbers are shown only to Central Command and
Public Safety. For everyone else they are stripped from `/events`
responses (and `metadata.plate_redacted` is set), and `/anpr` and ANPR
evidence return 403.

**Evidence files.** `/evidence/<folder>/...` maps the folder to an event
type (`road_defect`, `pedestrian_risk`, ...) and applies the same visibility
rule. `/videos/<name>` is open to Central Command; others may stream a
recording only if it is the linked evidence of an event they can see.

**URL tampering.** Changing the page in the address bar only changes what
the page asks for; the server still refuses. Web: a Roads user typing
`#users` is sent back to Roads home with "Not available for your role".
Mobile: typing `#/routes` gives "That screen belongs to another
department". API: another department's page endpoint returns 403, and
another department's event ID returns 404 (the ID's existence is not
confirmed).

**Accounts and sessions.**

* Passwords are hashed with PBKDF2-HMAC-SHA256, 600,000 iterations, 16-byte
  random salt (`pbkdf2_sha256$<iterations>$<salt>$<hash>`). Plain passwords
  are never stored or logged.
* A login creates a random session token. Only its hash is stored. It is
  sent back as an `HttpOnly`, `SameSite=Lax` cookie (`Secure` on HTTPS),
  and is also returned in the JSON so a native app could send it as
  `Authorization: Bearer <token>`.
* Web sessions last 12 h, mobile sessions 168 h (configurable).
* 5 failed logins for the same account from the same address lock it for
  60 s.
* Deactivating a user, changing their role or department, or resetting
  their password signs them out everywhere immediately.
* The pipeline's `POST /events` ingest accepts loopback calls, a Central
  Command session, or a shared `X-Ingest-Token` (`auth.ingest_token`) for a
  bus posting over the network.

---

## 5. The event workflow and deduplication

**One real-world problem, one managed event.** The pipeline's intelligence
engine already folds every sighting of the same pothole (same class, same
place, any bus) into one persistent issue (`POT-001`). A managed event
(`EV-1007`) is created from that issue, so four frames of one pothole give
one event, and a second bus seeing it later adds a sighting instead of a
new event.

Short-lived types (pedestrian risk, abnormal movement, rash driving,
collisions, congestion) get one more rule: issues of the same type from the
same bus and camera, in the same recording, within 20 s of video time and
60 m, are one event. The engine's windows are wall-clock time, which on a
slow CPU replay covers a fraction of a second of video; this is what turned
one pedestrian situation into many issues.

On the shipped data: **149 detections -> 31 engine issues -> 24 managed
events** (EV-1001 to EV-1024). The 83 pedestrian-risk detections became 17
events; the pothole sightings became 2. Ownership: 18 Public Safety, 6 Roads
& Infrastructure (4 damaged road, 2 pothole). Traffic has none because the
shipped run contains only traffic observations, no congestion or
bottleneck, so its pages correctly show empty states.

**Lifecycle.**

| Action | From -> to | Who |
|---|---|---|
| Verify | DETECTED -> VERIFIED | anyone in the owning department (field officers included), Central |
| Assign / reassign | VERIFIED -> ASSIGNED | owning department, Central; the assignee must belong to the owning department; a field officer may only assign to themselves |
| Start work | ASSIGNED -> IN_PROGRESS | owning department staff, Central; a field officer only if it is assigned to them |
| Mark resolved | IN_PROGRESS -> RESOLVED | same as Start work |

* Steps cannot be skipped (409 with an explanation).
* Every request carries the status the user was looking at; if someone else
  changed it first the second request gets 409 "changed to X before your
  update" and the screen refreshes, so two people can never both "verify".
* Every step is written to the activity log with who, when, from, to and
  the note.
* If a bus sees the same problem again after it was marked RESOLVED, the
  event re-opens and the department and assignee are notified.
* The workflow tables are separate from the pipeline's tables, so a new
  pipeline run can never overwrite a human decision.

**Notifications.** A new event notifies the owning department; critical
ones also notify Central Command; an event with a reported route delay
notifies Transport; an assignment notifies the assignee personally; a
resolution by a field officer or Central Command notifies the owning
department, and resolving a critical or hit-and-run event also notifies
Central Command; a forgotten-password request notifies Central Command.
Each user has their own read state.

---

## 6. Evidence

**Rule kept: the supplied annotated video is the inference output.** Nothing
in the platform runs a model, re-detects, or draws a box. The frames shown
are cut from `outputs/runs/annotated_<run_id>.mp4`, which the pipeline
wrote with its own boxes and labels at inference time.

* **New runs** link exactly: the orchestrator now stamps
  `metadata.run_id` on every event.
* **Older events** (like the shipped ones) carry no run id. The recording
  index narrows the candidates by physical consistency (frame number / fps
  must equal the event's video time, the frame must exist, the box must fit
  inside it, the run's time window must contain the event) and, if more
  than one recording still fits, compares the event's own saved crop with
  the same region of each candidate frame. Below `min_match_score` (0.5)
  the event is left unlinked rather than guessed, and only its crop is
  shown. On the shipped data, of the 29 detections the index was asked to
  link, 25 matched by crop comparison (lowest accepted score 0.504), 1
  linked on consistency alone and 3 had no qualifying recording. In the
  end 1 of the 24 managed events has no recording and shows only its crop.
* The frame endpoint extracts that exact frame once and caches it in
  `outputs/cache/frames/`.
* The video opens a few seconds before the event and "Jump to event frame"
  seeks to the middle of the event frame (`(frame + 0.5) / fps`) so it lands
  on it, not on the boundary before it. Video is served with HTTP Range
  support, so phones can seek without downloading the whole file.

Example: EV-1007 (pothole) links to `annotated_7baca60ac0b4.mp4` frame 135
with match score 0.925; the frame shows the pipeline's own `pothole 0.50`
box.

---

## 7. How web and mobile share data

There is one backend, one SQLite database and one API. Both apps are served
from the same origin (`/` and `/mobile/`), log in through the same
`/auth/login`, get the same session cookie and call the same endpoints
(`/cases`, `/cases/{id}/transition`, `/notifications`, `/map/layers`,
`/fleet`, ...). Neither app stores data of its own; there is nothing to
sync between them.

Live updates: both apps poll `GET /sync/state` (web every 8 s, mobile every
10 s). It returns a token that changes whenever any event, status,
assignment or notification changes, plus the unread count. When the token
changes the current screen reloads quietly; it never interrupts a form
being filled in or a video playing. So a pothole resolved on a phone shows
as RESOLVED on Central Command's screen within a few seconds without a
refresh.

The mobile service worker caches only the app shell (HTML, icons,
manifest). API responses, evidence frames and video are never cached, so a
phone never shows stale status or keeps evidence after logout.

---

## 8. API endpoints

**Every endpoint except `/health`, `/auth/login` and `/auth/forgot-password`
now requires a session.** Anonymous calls get 401.

### Original endpoints (same paths and response shapes, now scoped by role)

| Endpoint | Change |
|---|---|
| `GET /health` | unchanged, public |
| `POST /events`, `POST /events/batch` | pipeline ingest; loopback, Central session or `X-Ingest-Token` |
| `GET /events`, `GET /events/{id}` | filtered to visible types; plates redacted where not allowed |
| `GET /road-defects` | Roads, Central |
| `GET /traffic` | Traffic, Transport, Central |
| `GET /incidents`, `GET /incidents/{id}` | Safety, Traffic (collisions, rash driving), Transport (read-only), Central; `/incidents/{id}` now also returns `case_id` |
| `GET /hit-and-run` | Safety, Transport (read-only), Central |
| `GET /anpr`, `GET /vehicles` | Safety, Central only (registration numbers) |
| `GET /vehicles/{track_id}` | scoped; plate only for Safety and Central |
| `GET /issues`, `GET /issues/{id}` | issues of visible types only |
| `GET /analytics`, `GET /heatmap` | computed over visible types only |
| `GET /runs` | any signed-in user (run metadata only, no detections) |
| `GET /videos` | Central only (Live Analysis page) |
| `GET /videos/{name}` | Central, or linked evidence of a visible event; Range requests (206) |
| `GET /evidence/{path}` | folder mapped to event type, same visibility rule |
| `GET /`, `/app/` | web dashboard (unchanged path) |

### New endpoints

| Endpoint | Purpose |
|---|---|
| `POST /auth/login` | `{login, password, client: "web"|"mobile"}` -> user, permissions, session cookie + token |
| `POST /auth/logout` | end this session |
| `GET /auth/me` | current user and permissions (pages, visible types, map layers, home) |
| `POST /auth/forgot-password` | sends Central Command a "password reset requested" notification; the reply never reveals whether the account exists; rate-limited |
| `POST /auth/change-password` | change own password (needs the current one); signs out every session, including this one |
| `GET /users`, `POST /users` | list / create users (user managers) |
| `PATCH /users/{id}` | rename, activate/deactivate, change role or department |
| `POST /users/{id}/reset-password` | set a new password |
| `GET /departments`, `GET /roles` | reference lists with counts |
| `GET /cases` | managed events, filtered by status, open_only, department, type, class, bus, assigned (me/unassigned), severity, text, distance (`near=lat,lon`) |
| `GET /cases/summary` | counts by status, department, type, class, severity; my tasks |
| `GET /cases/lookup?issue_id=` or `?event_id=` | which managed event an issue or detection belongs to |
| `GET /cases/{id}` | full detail: evidence, allowed actions (with reasons), activity log, observations |
| `GET /cases/{id}/assignees` | people in the owning department with their open task counts |
| `POST /cases/{id}/transition` | `{action, expected_status, assignee_user_id?, note?}` |
| `GET /cases/{id}/frame.jpg` | the annotated frame at the event, from the pipeline's recording |
| `GET /notifications`, `GET /notifications/unread-count`, `POST /notifications/read` | per-user inbox |
| `GET /fleet`, `GET /fleet/routes`, `GET /fleet/od`, `GET /fleet/{bus_id}`, `PATCH /fleet/{bus_id}` | buses, activity, trails, route coverage (Transport, Central) |
| `GET /map/layers` | role-allowed GIS layers as markers and lines |
| `GET /sync/state` | change token + unread count for live refresh |
| `GET /mobile/` | mobile app (static) |

Full request and response schemas are in Swagger at `/docs`.

---

## 9. Database schema

**The six original tables (`events`, `issues`, `issue_observations`,
`vehicles`, `segments`, `runs`) are unchanged**: no column added, removed
or retyped. Existing databases keep working; the new tables are created
with `CREATE TABLE IF NOT EXISTS` on start-up.

| New table | Holds |
|---|---|
| `users` | user_id, name, email (unique), username (unique), password_hash, role, department, is_active, can_manage_users, created/updated/last_login timestamps, created_by |
| `sessions` | token_hash (key), user_id, client (web/mobile), created_at, expires_at, last_seen_at, user_agent |
| `cases` | one row per managed event: case_id (EV-1001...), type, class, title, owner_department, status, severity, priority, confidence, buses, bus_count, observation_count, location, road_segment, primary event and issue, linked recording, first/last seen, assignee and who/when for every step, resolution_note, reopened_count, version |
| `case_issues` | which engine issues make up each managed event, with the observation count and buses last seen |
| `case_activity` | the audit log: when, actor, actor department, action, from/to status, note |
| `notifications` | department-wide or personal alerts: severity, kind, title, body, case_id |
| `notification_reads` | per-user read state |
| `evidence_links` | event -> annotated recording, frame index, fps, link method and score (cache) |
| `fleet` | bus_id, registration, route, depot, service status, cameras, notes |

Login throttling is held in memory, not in the database.

---

## 10. Files added and changed

### Added

| File | What |
|---|---|
| `backend/rbac.py` | roles, departments, visibility, ownership, pages, map layers, the `Scope` object |
| `backend/auth.py` | users, password hashing, sessions, throttling, demo seeding, request authentication |
| `backend/workflow.py` | managed events, grouping, lifecycle, activity log, notifications, sync from issues |
| `backend/recordings.py` | links events to the annotated recording and frame; frame extraction |
| `backend/fleet.py` | fleet registry, bus activity, trails, route coverage |
| `backend/platform_routes.py` | all new API routes |
| `mobile/index.html`, `manifest.webmanifest`, `sw.js`, `icon.svg`, `icon-192.png`, `icon-512.png` | mobile Command & Field app |
| `scripts/import_events.py` | loads a shipped `events.jsonl` into the database, no model |
| `scripts/manage_users.py` | command-line user management |
| `tests/test_platform_rbac_workflow.py` | 28 tests for auth, RBAC, dedup, evidence, workflow |
| `docs/PLATFORM.md`, `docs/ACCEPTANCE_TEST.md` | this document and the 20-step acceptance test |

### Changed

| File | Change |
|---|---|
| `backend/app.py` | login required, every original endpoint scoped by role, new stores and routes mounted, `/mobile/` served, `case_id` on incidents |
| `backend/db.py` | small query helpers (`query`, `query_one`, `execute`, `executescript`, `transaction`); `insert_event` takes `run_id` from metadata. Schema unchanged |
| `frontend/index.html` | login, role identity bar, per-role sidebars and homes, event drawer, new pages, role-aware map, live refresh; one pre-existing selector bug fixed (see below) |
| `pipeline/orchestrator.py` | stamps `metadata.run_id` on each event. No detection logic touched |
| `core/sinks.py` | HTTP sink can send `X-Ingest-Token` |
| `config/platform.yaml` | new `auth`, `workflow`, `fleet` sections |
| `run_backend.py` | prints the mobile address and LAN hint |
| `demo.sh`, `demo.bat` | new `--dashboard` mode; demo password default; `demo.sh` made executable |
| `README.md`, `docs/DEMO.md`, `docs/ARCHITECTURE.md`, `docs/EVENT_SCHEMA.md` | platform sections added |

Pre-existing bug fixed in passing: on the original Road assets page the
filter-button handler was bound with `querySelectorAll("[data-r]")`, which
also matched the sidebar links (they carry the same `data-r` attribute), so
after opening Road assets a sidebar click set the road filter to the page
name instead of simply navigating. It is now scoped to the page content.

---

## 11. Configuration

All in `config/platform.yaml`, each overridable by environment variable
(`UIP_<SECTION>__<KEY>`).

| Key | Default | Meaning |
|---|---|---|
| `auth.enabled` | true | false = no login, everyone is Central Command (local debugging only) |
| `auth.seed_demo_users` | true | create the six demo accounts in an empty database |
| `auth.demo_password` | null | set via `UIP_AUTH__DEMO_PASSWORD`; random if unset |
| `auth.pbkdf2_iterations` | 600000 | password hashing cost |
| `auth.session_ttl_hours` / `mobile_session_ttl_hours` | 12 / 168 | session lifetimes |
| `auth.max_failed_logins` / `lockout_seconds` | 5 / 60 | login throttling |
| `auth.ingest_token` | null | shared secret for buses posting events over the network |
| `workflow.first_case_number` | 1001 | EV- numbering |
| `workflow.group_window_s` / `group_radius_m` | 20 / 60 | extra grouping for short-lived types |
| `workflow.clip_before_s` / `clip_after_s` | 3 / 4 | evidence video window |
| `workflow.min_match_score` | 0.5 | minimum score to link an older event to a recording |
| `fleet.active_window_minutes` | 30 | a bus is "reporting" if it sent data this recently |
| `fleet.buses` | BUS-01..03 on R-01 | fleet registry seed |

---

## 12. Tests

```bash
python -m pytest tests -q
```

* `tests/test_platform_rbac_workflow.py`: 28 tests, all pass. Each test runs
  on its own copy of a database rebuilt from the shipped output, in a
  temporary folder, so `outputs/` is never touched.
* Whole suite: 90 passed, 1 skipped (the model smoke test skips when
  `ultralytics` is not installed), 2 failed.
* The 2 failures (`test_incident_engine.py::test_case5...` and
  `::test_case6...`) were already failing in the project as received,
  before any of this work, and are unrelated (ANPR thresholds in the
  incident engine). They were left alone.

Manual and browser checks done during development (Playwright, desktop
1440x900 and phone 390x844, all six roles): every page and tab renders with
no script errors; URL guards redirect; the full lifecycle run on the phone
(Roads verifies and assigns, Field Officer starts and resolves) appears as
RESOLVED with the complete activity log on Central Command's web screen;
the evidence frame and video play in Chrome, and "Jump to event frame"
lands inside the event frame.

---

## 13. Assumptions and limitations

* **Demo data.** Everything shown comes from one real run of the pipeline
  (5 recordings, 149 detections, BUS-01, 17 Sep 2026). Traffic has no
  events because that run contains only traffic observations. Fleet shows
  BUS-02 and BUS-03 as "never reported" and BUS-01 as idle, because no bus
  is live. Nothing was invented to fill these gaps.
* **No plate reads in the shipped run**, so plate redaction is covered by a
  unit test on a synthetic row rather than by the demo data.
* **One managed event has no linked recording** (3 of the older
  detections had none that passed the consistency checks); it shows only
  its saved crop. New runs link exactly by `run_id`.
* **Forgot password** notifies Central Command and tells the user they
  will be given a new password; it does not send e-mail (no mail server is
  configured).
  Central Command resets the password from the Users page or
  `manage_users.py`.
* **Live refresh is polling**, not WebSockets: changes appear within
  8 to 10 s. Fine for a few hundred users on one server.
* **SQLite, one server process.** Matches the existing project. For many
  concurrent writers or several servers, move to PostgreSQL; the stores use
  plain SQL through the `Database` helpers.
* **The mobile app is a PWA, not a store app.** It installs to the home
  screen and runs full-screen. A native wrapper (Capacitor, React Native)
  could reuse the same API with the Bearer token.
* **Leaflet maps load from a CDN.** Without internet the dashboards fall
  back to a built-in SVG map with the same markers and click-through.
* **GPS on the phone needs HTTPS** (or localhost). For field use, put the
  server behind HTTPS (a reverse proxy such as Caddy or nginx).
* **Browser video** needs H.264 support (all mainstream phone and desktop
  browsers have it).
* **Security for production:** set `auth.seed_demo_users: false`, serve
  over HTTPS, set `auth.ingest_token` if buses post over a network, and
  keep `outputs/` (database, evidence) off public storage.
