# AI Urban Intelligence Platform

Turning public transport buses into mobile urban sensing units.

One bus camera feed goes in. Road defects, traffic congestion, rash driving,
collisions, hit-and-run and the registration numbers of offending vehicles
come out — deduplicated across buses, tracked over time, scored by priority
and routed to the right city department.

Five team modules, previously separate repos, run here as **one application**
on a normal Windows or Linux machine, CPU or GPU. Three AI models total:
vehicle detection + tracking, road infrastructure, and the plate pipeline.
Incident detection, hit-and-run reasoning, traffic analytics, persistence and
priority are all logic on top of those, not extra models.

> **Phase note.** This is the integration phase. Nothing here is
> Jetson-specific and nothing is TensorRT/ONNX/FP16-optimised — that is a
> deliberately separate later phase. The architecture is hardware-independent
> so that phase changes deployment, not design.

---

## 1. Quick start

```bash
# 1. dependencies
python -m pip install -r requirements.txt

# 2. does this machine have everything?
python scripts/check_setup.py

# 3. the whole thing, one command
./demo.sh                 # Linux / macOS
demo.bat                  # Windows
```

`demo.sh` / `demo.bat` runs the video through the full pipeline, writes the
database and evidence, starts the backend and opens the dashboard at
<http://127.0.0.1:8000>.

No models on this machine, or no GPU? The multi-department platform runs on
the pipeline output already in `outputs/`, with nothing re-detected:

```bash
./demo.sh --dashboard      # Linux / macOS
demo.bat --dashboard       # Windows
```

That loads `outputs/runs/events.jsonl`, builds the managed events and starts
both apps: the web dashboard at <http://127.0.0.1:8000/> and the mobile
Command & Field app at <http://127.0.0.1:8000/mobile/>. Sign in as
`admin@urbanintel.local` (or `traffic@`, `roads@`, `safety@`, `transport@`,
`field@`) with the password `UrbanIntel@2026`. See
[docs/PLATFORM.md](docs/PLATFORM.md).

Run the pieces separately if you prefer:

```bash
python run_pipeline.py --source data/videos/street.mp4
python run_backend.py
# then open http://127.0.0.1:8000
```

---

## 2. What is where

```
urban-intelligence-platform/
├── config/platform.yaml     ← EVERY tunable value and path lives here
├── core/                    ← the common interface (event, module base, config…)
├── modules/
│   ├── perception/          ← the four models, each loaded ONCE per run
│   ├── road_detection/      ← MODULE A/B  Person 1
│   ├── traffic/             ← MODULE C    Person 2
│   ├── incident/            ← MODULE D/E  Person 4  (collision · hit-and-run)
│   ├── anpr/                ← MODULE F    Person 5  (event-triggered only)
│   └── intelligence/        ← MODULE G    Person 6
├── pipeline/orchestrator.py ← the only file that knows the module order
├── backend/                 ← FastAPI + SQLite, auth, RBAC, event workflow
├── frontend/index.html      ← the web dashboard (Central Command + departments)
├── mobile/                  ← the mobile Command & Field app (PWA)
├── vendor/                  ← EVERY TEAM MEMBER'S ORIGINAL REPO, UNMODIFIED
├── models/                  ← the .pt weights the pipeline loads
├── data/videos/             ← input clips
└── outputs/                 ← db, events.jsonl, evidence images, annotated mp4
```

**`vendor/` is read-only by convention.** Team members keep committing to
their own repos; integration only ever loads from there. The wrappers in
`modules/` are adapters, not rewrites — see
[docs/MODULES.md](docs/MODULES.md) for what each one reuses, line by line.

---

## 3. The pipeline

```
                          VIDEO INPUT  (file / webcam / stream)
                                  │
                          VIDEO PROCESSOR
                                  │
                    ┌──────────────┬─────────────┬──────────────┐
                    ▼              ▼             ▼              ▼
             VEHICLE MODEL   ROAD DEFECT   WATERLOGGING     PLATE MODEL
             + BoT-SORT         MODEL         MODEL       (event-triggered
             (Person 2)       (Person 1)   (single-class)    only, below)
                    └──────────────┴─────────────┘
                                  ▼
                      FrameContext  (one shared read)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
            ROAD MODULE                    TRAFFIC ENGINE
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                          INCIDENT ENGINE  (Person 4)
                    collision · rash · HIT-AND-RUN
                                  │  requires_identification?
                                  ▼
                              ANPR  (Person 5)   ← only on incidents
                                  │
                                  ▼
                        COMMON EVENT FORMAT
                                  ▼
                   INTELLIGENCE ENGINE  (Person 6)
              dedup · multi-bus · persistence · repair
              verification · priority · department routing
                                  ▼
                    SQLite  +  events.jsonl  +  evidence
                                  ▼
                       BACKEND API  →  DASHBOARD
```

Each model runs **at most once per frame**. Both analysis engines need
vehicle tracks; only one vehicle model is loaded. There are exactly four
models in the system: vehicle, road defect, waterlogging, and the plate
pipeline. The road defect and waterlogging detections land in the same
shared list, so the road module, annotator, and intelligence engine treat a
waterlogging hit exactly like any other road defect. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 4. The rules the code keeps

| Rule | Where it is enforced |
|---|---|
| Modules never import each other | they only exchange `core.event.Event` |
| One failed module never kills the run | `core/module_base.py` — `safe_process()` catches, logs with the module name, and continues |
| A missing model degrades, never crashes | `ModuleUnavailable` → module marked UNAVAILABLE, pipeline runs without it |
| ANPR never runs per frame | `ANPRModule.process()` returns `[]` by design; only `identify()` runs, and only on an incident |
| No hardcoded paths or thresholds | everything reads `config/platform.yaml` |
| No dummy detections | if a model is missing the module says so in the log and is skipped |

---

## 5. Common event format

Every module produces objects of exactly this shape, and nothing else
crosses a module boundary:

```json
{
  "event_id": "…", "event_type": "road_defect|traffic_congestion|hit_and_run|…",
  "source_module": "road_detection|traffic|incident|anpr|intelligence",
  "bus_id": "BUS-01", "camera_id": "CAM-FRONT",
  "timestamp": "2026-09-09T04:52:43.775+00:00",
  "gps": {"latitude": 13.08, "longitude": 80.27, "speed_kmh": null},
  "confidence": 0.91, "severity": "HIGH",
  "track_id": 24, "class_name": "car", "bounding_box": [x1, y1, x2, y2],
  "evidence_image": "hit_and_run/BUS-01_000232_07fd3a.jpg",
  "evidence_video": null,
  "frame_number": 232, "video_time_s": 7.7, "road_segment": "SEG-13.08490-80.27351",
  "issue_id": "HNR-001", "priority_score": 77.1,
  "department": "Traffic Police", "duplicate_of": null,
  "metadata": { }
}
```

No module is forced to fill a field it does not have — everything except
`event_type` and `source_module` has a default. Module-specific detail lives
in `metadata`, so the schema never grows a per-module column.
Full reference: [docs/EVENT_SCHEMA.md](docs/EVENT_SCHEMA.md).

---

## 6. Configuration

`config/platform.yaml` is the single source of truth. Three ways to override,
in increasing priority:

```bash
# 1. edit the yaml
# 2. environment (double underscore = nesting)
UIP_VIDEO__SOURCE=data/videos/other.mp4 python run_pipeline.py
# 3. command line
python run_pipeline.py --source data/videos/other.mp4 --bus-id BUS-02 --device 0
```

Useful flags:

```bash
--device 0                  # run on the GPU (nothing else changes)
--resize-width 960          # big speed-up on CPU
--display                   # live annotated window, q to quit
--only traffic incident     # run a subset of modules
--disable anpr              # switch one off
--gps 13.0827,80.2707       # fixed position instead of the simulated route
--max-frames 300            # short test run
```

---

## 7. Backend and dashboard

```bash
python run_backend.py                 # http://127.0.0.1:8000
python run_backend.py --host 0.0.0.0 --port 8080
```

| Endpoint | Returns |
|---|---|
| `POST /events`, `POST /events/batch` | ingest (used when `backend.mode: http`) |
| `GET /events`, `GET /events/{id}` | the raw event stream, filterable |
| `GET /road-defects` `GET /traffic` `GET /incidents` | per-module views |
| `GET /incidents/{id}` `GET /hit-and-run` `GET /anpr` | investigation detail, alerts, plate reads |
| `GET /vehicles`, `GET /vehicles/{track_id}` | identified vehicles + plates |
| `GET /issues`, `GET /issues/{issue_id}` | persistent issues, priority, department |
| `GET /analytics`, `GET /heatmap` | dashboard rollups, GIS-ready segments |
| `GET /runs` | run history |
| `GET /evidence/{path}` | evidence images |
| `GET /docs` | interactive OpenAPI |

The pipeline writes to SQLite directly by default. Set
`backend.mode: http` to make it POST to a running backend instead — same
events, different transport, no module changes.

**Every endpoint above now needs a session** (`POST /auth/login`), and each
one returns only what the signed-in role may see. The multi-department layer
adds the managed-event, user, notification, fleet, map and sync endpoints on
top, listed in [docs/PLATFORM.md](docs/PLATFORM.md) section 8. Paths and
response shapes of the original endpoints did not change.

---

## 8. Tests

```bash
python -m pytest tests -q
```

Covers the event schema, each module in isolation against real weights, the
incident geometry, the intelligence engine's dedup/persistence/priority, and
the guarantee that ANPR never fires per frame. `tests/test_platform_rbac_workflow.py`
adds 28 tests for login, role-based access, deduplication, evidence linking
and the event workflow; they build their own database in a temporary folder
and never touch `outputs/`.

---

## 9. Demo procedure

```bash
python scripts/run_demo.py                       # default clip
python scripts/run_demo.py --source my_clip.mp4
python scripts/run_demo.py --multi-bus           # 3 buses on one road
```

`--multi-bus` replays the clip as BUS-01/02/03. This is the one to show:
the same defect seen by three buses becomes **one** issue with
`observations=3, buses=3`, which is what Module 6 exists to do.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `check_setup.py` fails on a package | `pip install -r requirements.txt` |
| Module logged as `UNAVAILABLE` | its `.pt` file is missing — the path is in the log line; drop the weights in `models/` |
| No incidents on footage that has one | check the log for `[HIT_RUN]` lines: they name the state each vehicle reached. `TRACK_LOST` means the tracker dropped the id; `COLLISION_ONLY` means the vehicle did not travel `departure_distance_threshold` px |
| Hit-and-run never fires in a demo | that is the design — run `--demo-incident` to exercise the chain with a clearly-labelled synthetic trigger |
| Very slow on CPU | `--resize-width 960`, or `perception.road_defect.frame_stride: 10` |
| `cv2` import error on a headless server | `pip install opencv-python-headless` |
| Dashboard empty | run the pipeline first — the dashboard reads the database the pipeline writes |

---

## 11. Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — current + integrated architecture, data flow, design decisions
* [docs/MODULES.md](docs/MODULES.md) — what each owner shipped, what the adapter reuses, how to extend it
* [docs/EVENT_SCHEMA.md](docs/EVENT_SCHEMA.md) — the common event format, field by field
* [docs/DEMO.md](docs/DEMO.md) — the three demo modes and a five-minute walkthrough
* [docs/PLATFORM.md](docs/PLATFORM.md) : the multi-department layer, roles, workflow, mobile app, API, schema
* [docs/ACCEPTANCE_TEST.md](docs/ACCEPTANCE_TEST.md) : 20-step acceptance run, from login to a pothole resolved on a phone
* [docs/INTEGRATION_REPORT.md](docs/INTEGRATION_REPORT.md) — the first integration write-up (partly superseded)


---

## 12. What changed in the incident upgrade

* **Pedestrian detection removed entirely** — model, module, perception,
  events, config, API, database column, dashboard page and tests. Nothing
  pedestrian-related remains connected to the pipeline.
* **Hit-and-run rebuilt on Person 4's detectors** (`vendor/person4_incident/`)
  with an explicit state machine. The old logic reported 2 hit-and-runs on
  266 frames of crash-free traffic; the same clip now reports none. See the
  header of `modules/incident/module.py` for the four defects that caused it.
* **A vanished track is never a hit-and-run.** `track_loss_timeout` frames of
  absence resolves to `TRACK_LOST` and emits nothing.
* **New event types**: `possible_collision` and `abnormal_movement` — the
  system now says "possible" when the evidence is partial instead of
  asserting a collision.
* **ANPR publishes `UNKNOWN`** rather than a low-confidence guess
  (`min_plate_confidence`).
* **New command-center dashboard** with ten pages, a GIS map, an incident
  register, a hit-and-run investigation view with state timeline and
  before/impact/after evidence, and a live video page whose event feed is
  synchronised to the annotated recording.
* **`--demo-incident`** injects one clearly-labelled synthetic collision so
  the full chain can be demonstrated on footage with no real crash. Every
  event it produces carries `synthetic: true` and shows a SYNTHETIC badge.
  It is off by default and it is never a detection.
