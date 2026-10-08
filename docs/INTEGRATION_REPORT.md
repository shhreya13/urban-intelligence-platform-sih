> **SUPERSEDED IN PART — read this first.**
>
> This document records the *first* integration phase (five repos into one
> application). Two things have changed since, and where they disagree with
> the text below, these win:
>
> 1. **Pedestrian detection has been removed entirely** — model, module,
>    perception, event types, config, API, database column, dashboard page
>    and tests. Every mention of Person 3 / pedestrian risk below is history.
> 2. **Incident detection and hit-and-run were rebuilt** on Person 4's own
>    detectors, which arrived after this report was written. The `near_miss`
>    event type is gone, replaced by `possible_collision` and
>    `abnormal_movement`, and the hit-and-run logic now runs an explicit
>    state machine. See [MODULES.md](MODULES.md) and the header of
>    `modules/incident/module.py`.
>
> The architecture, event schema, backend, database and testing sections are
> otherwise still accurate.

---

# Integration report

Six independent repositories → one working application, running on a normal
Windows/Linux machine, CPU or GPU. No Jetson, no TensorRT/ONNX/FP16, no
model rewritten, no dummy detections, no existing functionality removed.

---

## A. Current project architecture (what arrived)

| Owner | Repository | Entry point | State on arrival |
|---|---|---|---|
| Person 1 | `road-defect-model` | `inference/predict.py` | **Works.** One multi-class YOLO checkpoint (7 classes) + a 30-line predict script that prints results. |
| Person 2 | `traffic-intelligence-phase1` | `src/main.py` | **Works.** Vehicle detector + BoT-SORT + 5 analytics classes (density, congestion, bottleneck, peak, counter) + speed estimator. Draws to a window. |
| Person 3 | `Person3_Pedestrian_Safety_AI` | `run_safety.py` | **Partly works.** Geometry + `RiskEngine` are complete and good. **No trained weights shipped** (`models/README.txt` is a placeholder). Loads a *temporary COCO vehicle model* — their own README asks for it to be replaced with Person 2's. |
| Person 4 | — | — | **Empty.** No code arrived. |
| Person 5 | `anpr_module` | `run.py` | **Works.** Plate YOLO + enhancement + EasyOCR + multi-frame voting + format correction. Runs on **every** vehicle, every frame. |
| Person 6 | — | — | **Empty.** No code arrived. |

**Frontend:** none existed. **Backend:** none existed. **Database:** none
existed. **Shared entry point:** none existed.

### Problems this created

1. **Three vehicle models.** Person 2 loaded one; Person 3 loaded a second
   ("temporary"); anything Person 4 built would have loaded a third. Three
   sets of track ids that could never be cross-referenced.
2. **No machine-readable output.** Every module printed or drew on a frame.
   Nothing could be stored, correlated or served.
3. **ANPR ran continuously** — the single most expensive model in the system,
   on every vehicle in every frame.
4. **No downstream at all.** No database, API, dashboard, or notion of the
   same pothole being seen twice.

### Duplicate functionality found

| Duplicated | In | Resolution |
|---|---|---|
| Vehicle detection + tracking | Person 2, Person 3 (temp), implied Person 4 | one shared `VehiclePerception` — Person 2's |
| Speed estimation | Person 2's `SpeedEstimator`, Person 3's closing-speed maths | Person 2's for vehicles, Person 3's for pedestrian-relative closing speed (different quantities, both kept) |
| Video reading / resizing | all four repos | one `pipeline/video_source.py` |
| Annotation drawing | Person 2, Person 3, Person 5 | one `pipeline/annotator.py` |
| YOLO weights on disk | `yolo11n.pt` in two repos | one copy in `models/` |

---

## B. Integration architecture

Three new layers **around** the team code — nothing inside it changed.

```
   core/       the contract      Event · FrameContext · BaseModule · Config
   modules/    the adapters      one thin wrapper per owner
   pipeline/   the orchestrator  the only component that knows the order
```

```
                          VIDEO INPUT  (file / webcam / stream)
                                  │
                          VIDEO PROCESSOR
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      ▼                           ▼                           ▼
 VEHICLE MODEL              ROAD DEFECT                 PEDESTRIAN
 + BoT-SORT                    MODEL                       MODEL
 (Person 2)                  (Person 1)                  (Person 3)
      └───────────────────────────┼───────────────────────────┘
                                  ▼
                        FrameContext — one shared read
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
            ROAD MODULE     TRAFFIC ENGINE   PEDESTRIAN RISK
                  │               │               │
                  └───────────────┼───────────────┘
                                  ▼
                          INCIDENT ENGINE (Person 4)
                                  │ requires_identification
                                  ▼
                              ANPR (Person 5)
                                  ▼
                        COMMON EVENT FORMAT
                                  ▼
                   INTELLIGENCE ENGINE (Person 6)
        dedup · multi-bus correlation · persistence · repair
        verification · priority scoring · department routing
                                  ▼
                 SQLite + events.jsonl + evidence images
                                  ▼
                     BACKEND API  →  DASHBOARD / GIS
```

Three decisions carry the whole design:

**1. Shared perception.** Each model is loaded once per run and written into
`FrameContext`. Traffic, pedestrian safety and incidents all read the same
vehicle list, so `track_id` 24 means the same vehicle to every module — which
is what makes "pedestrian risk from track 24 → hit-and-run by track 24 →
plate of track 24" possible at all.

**2. One event format.** `core/event.py`. Modules never import each other;
they read the context and return `Event`s. Adding a seventh module changes
one line of the orchestrator and nothing else.

**3. Event-triggered ANPR, enforced structurally.** `ANPRModule.process()`
returns `[]` unconditionally. There is no code path that OCRs during normal
driving.

---

## C. Files modified

**In the team's code: none.** All four delivered repositories sit in
`vendor/` byte-identical to what was shipped (verified with `diff -r`; the
only difference is that duplicate `yolo11n.pt` / `yolo11s.pt` weights were
moved to `models/` so one copy is loaded instead of three).

Everything below is new. That is deliberate: the team keeps working in their
own repos, and integration only ever *reads* from `vendor/`.

Two things were relocated, not edited:

| From | To | Why |
|---|---|---|
| `person1_road_defect/weights/best.pt` | `models/road_defect_best.pt` | one weights directory the config points at |
| `anpr_module/models/best (1).pt` | `models/plate_best.pt` | same, plus the space in the filename broke path handling on Windows |

---

## D. New files created

**Core contract — `core/` (9 files)**

| File | Purpose |
|---|---|
| `event.py` | the common event format, `EventType` / `SourceModule` / `Severity` vocabularies |
| `frame_context.py` | per-frame object carrying the frame + shared perception results |
| `module_base.py` | `BaseModule`, `ModuleStatus`, `ModuleUnavailable`, the `safe_*` wrappers |
| `config.py` | `platform.yaml` + `UIP_*` env + CLI overrides, path resolution |
| `vendor_loader.py` | imports files out of `vendor/` without requiring installable packages |
| `geo.py` | GPS provider (simulated/fixed/none), haversine, road-segment quantisation |
| `evidence.py` | evidence crop/clip writer |
| `track_buffer.py` | per-track rolling vehicle crops — the ANPR handoff buffer |
| `sinks.py` | JSONL / SQLite-direct / HTTP event sinks |

**Adapters — `modules/` (17 files)**

`perception/vehicle.py` · `perception/pedestrian.py` ·
`perception/road_defect.py` · `perception/clock.py` (video-time clock, so
durations are measured in video seconds not wall seconds) ·
`road_detection/module.py` · `traffic/module.py` · `pedestrian/module.py` ·
`incident/module.py` · `anpr/module.py` · `intelligence/module.py` ·
`intelligence/priority.py` · `intelligence/routing.py` + `__init__.py`s.

**Pipeline — `pipeline/` (4 files)**
`orchestrator.py` · `video_source.py` · `annotator.py` · `__init__.py`

**Backend — `backend/` (3 files)**
`app.py` (FastAPI, 20 endpoints) · `db.py` (SQLite, 6 tables) · `__init__.py`

**Frontend — `frontend/index.html`** (single file, no build step)

**Config — `config/platform.yaml`** (every tunable value in the system)

**Entry points** — `run_pipeline.py` · `run_backend.py` ·
`scripts/run_demo.py` · `scripts/check_setup.py` · `demo.sh` · `demo.bat`

**Tests — `tests/` (5 files, 33 tests)**

**Docs** — `README.md` · `docs/ARCHITECTURE.md` · `docs/MODULES.md` ·
`docs/EVENT_SCHEMA.md` · this file · `requirements.txt` · `.gitignore`

---

## E. How each module connects

| Module | Reuses from `vendor/` | Adapter adds | Reads | Emits |
|---|---|---|---|---|
| **1 Road** (P1) | `weights/best.pt`, one multi-class detector | class-name mapping via YAML, within-pass dedup, absence inference for missing divider/zebra, strided inference | frame | `road_defect` |
| **2 Traffic** (P2) | `VehicleDetector`, `SpeedEstimator`, `TrafficDensity`, `CongestionAnalyzer`, `BottleneckAnalyzer`, `PeakTrafficAnalyzer`, `VehicleCounter` | Events, per-segment analytics, estimated delay, heatmap rows | `ctx.vehicles` | `traffic_observation`, `traffic_congestion`, `traffic_bottleneck` |
| **3 Pedestrian** (P3) | `geometry.py`, `risk_engine.py`, their `config.yaml` zones | uses Person 2's vehicle tracks instead of a second model (**the change their own README asked for**), Events, evidence | `ctx.pedestrians` + `ctx.vehicles` | `pedestrian_risk` |
| **4 Incident** (P4) | *(nothing existed)* — **no new ML model**, reasoning only | rash driving, collision, near-miss, hit-and-run, watchlist, ANPR trigger | `ctx.vehicles` + `ctx.pedestrians` | `rash_driving`, `near_miss`, `collision`, `hit_and_run` |
| **5 ANPR** (P5) | `ANPRPipeline.process_incident()` — their OCR, voting and correction tables untouched | **event-trigger gate**, crop buffer handoff, sharpness filter, Events | incident trigger + buffered crops | `vehicle_identification` |
| **6 Intelligence** (P6) | *(nothing existed)* — plain Python | normalise, dedup, multi-bus correlation, persistence, repair verification, priority, routing | the whole event stream | `persistent_issue` |

**The incident → ANPR handoff**, concretely:

```
IncidentModule emits hit_and_run  { track_id: 24,
                                    metadata.requires_identification: true,
                                    metadata.anpr_track_id: 24 }
        ▼
orchestrator pulls track 24's crops from VehicleCropBuffer (last 12, stride 2)
        ▼
ANPRModule.identify(24, crops) → Person 5's process_incident()
        ▼
Event(vehicle_identification, registration_number, plate_confidence,
      evidence_image = best plate crop)
```

---

## F. Common event schema

One dataclass, every module, defaults everywhere except `event_type` and
`source_module`; module-specific detail in `metadata` so the schema never
grows a per-module column.

```json
{
  "event_id": "…", "event_type": "hit_and_run", "source_module": "incident",
  "bus_id": "BUS-01", "camera_id": "CAM-FRONT",
  "timestamp": "2026-09-09T04:52:43.775+00:00",
  "gps": {"latitude": 13.0849, "longitude": 80.2735,
          "accuracy_m": null, "speed_kmh": null, "heading_deg": null},
  "confidence": 0.71, "severity": "CRITICAL",
  "track_id": 24, "class_name": "car", "bounding_box": [612, 300, 830, 470],
  "evidence_image": "hit_and_run/BUS-01_000232_07fd3a.jpg",
  "evidence_video": null,
  "frame_number": 232, "video_time_s": 7.73,
  "road_segment": "SEG-13.08490-80.27351",
  "issue_id": "HNR-001", "priority_score": 70.0,
  "department": "Traffic Police", "duplicate_of": null,
  "metadata": {"requires_identification": true, "anpr_track_id": 24, "…": "…"}
}
```

`issue_id`, `priority_score`, `department` and `duplicate_of` are null until
Module 6 fills them. Full field reference, plus every module's `metadata`
keys: [EVENT_SCHEMA.md](EVENT_SCHEMA.md).

---

## G. Complete processing pipeline

Per frame, in order:

1. **decode** — `video_source.py`: file / webcam / stream, optional resize, frame number + video time
2. **position** — GPS provider → lat/lon → `road_segment`
3. **shared perception** — vehicle (every frame) · pedestrian (every frame) · road defect (every 5th frame — defects are static)
4. **crop buffer** — last N crops per track retained, for a later ANPR request
5. **road detection** (P1) → Events
6. **traffic** (P2) → Events
7. **pedestrian risk** (P3) → Events
8. **incidents** (P4) → Events, some carrying `requires_identification`
9. **ANPR** (P5) → *only* for those, from buffered crops
10. **intelligence** (P6) → normalise → dedup → correlate → persist → score → route
11. **sinks** — `events.jsonl` + SQLite + annotated frame

At end of video, `finalize()` flushes anything still accumulating (an open
congestion window, an unresolved hit-and-run watchlist).

**Error containment:** a module that throws is caught in `safe_process()`,
logged **with its module name and the frame number**, and skipped for that
frame. Ten consecutive errors mark it FAILED and it is skipped for the rest
of the run. A module whose model file is missing raises `ModuleUnavailable`
in `setup()`, is marked UNAVAILABLE with the exact path in the log, and the
pipeline runs without it. One module can never take the run down.

---

## H. Backend API

FastAPI + SQLite, `backend/app.py`. The pipeline writes to SQLite directly by
default; set `backend.mode: http` to POST to a running backend instead — same
events, different transport, no module changes.

| Endpoint | Purpose |
|---|---|
| `POST /events`, `POST /events/batch` | ingest |
| `GET /events` | filter by type, module, severity, bus, issue, run, limit |
| `GET /events/{event_id}` | one event |
| `GET /road-defects` | Module 1 view |
| `GET /traffic` | Module 2 view (`?congestion_only=true`) |
| `GET /pedestrian-alerts` | Module 3 view |
| `GET /incidents` | Module 4 view (`?event_type=hit_and_run`) |
| `GET /vehicles`, `GET /vehicles/{track_id}` | identified vehicles + plates + the incident that triggered identification |
| `GET /issues`, `GET /issues/{issue_id}` | persistent issues, priority, department, full observation history |
| `GET /analytics` | dashboard rollups |
| `GET /heatmap` | per-segment GIS-ready aggregates |
| `GET /runs` | run history |
| `GET /evidence/{path}` | evidence images |
| `GET /health` | liveness + counts |
| `GET /docs` | interactive OpenAPI |
| `GET /` | the dashboard |

---

## I. Frontend

No frontend existed, so one was built: `frontend/index.html`, single file, no
build step, served by the backend at `/`.

Tabs: **Overview** (counts, events by type, department routing, priority
bands, traffic over the run) · **Road Defects** · **Traffic** · **Pedestrian
Safety** · **Incidents** · **Number Plates** · **Persistent Issues** ·
**Map / Heatmap**.

Every row shows bus id, camera id, timestamp, GPS, class, severity,
confidence, priority, issue id and a clickable evidence thumbnail. Auto-refresh
is a toggle, so it can sit on a screen during a demo.

---

## J. Database

SQLite, `outputs/urban_intelligence.db`, created on first run. Six tables:

| Table | Holds |
|---|---|
| `events` | every event, schema columns mirrored one-for-one, `gps` flattened to lat/lon, `metadata` as JSON |
| `issues` | persistent issues: id, type, status, first/last seen, observations, buses, priority, department, clean passes, reopen count |
| `issue_observations` | the many-side — every sighting that attached to an issue |
| `vehicles` | track id + bus, plate, plate confidence, triggering incident |
| `segments` | per-road-segment rollups that feed `/heatmap` |
| `runs` | run id, bus, camera, source, frames, events, timing |

Indexed on the columns the API filters by. No ORM, no migrations — the
schema is created idempotently at startup.

---

## K. Dependencies

`requirements.txt` is the union of the five team requirement files,
de-duplicated:

```
ultralytics>=8.3  torch>=2.2  torchvision>=0.17
opencv-python>=4.8  numpy>=1.24  scipy>=1.10  pandas>=2.0  lap>=0.4
easyocr>=1.7  Pillow>=10
PyYAML>=6.0
fastapi>=0.110  uvicorn[standard]>=0.27  python-multipart>=0.0.9
pytest>=7.4
```

Deliberately excluded: `roboflow` (dataset download, training-time only) and
`tensorrt`/`onnx` (the later phase).

```bash
python -m pip install -r requirements.txt     # CPU, Windows or Linux
# GPU: install torch from pytorch.org first, then the line above
```

---

## L. Commands

```bash
python scripts/check_setup.py                  # will this machine run it?
python -m pytest tests -q                      # 33 tests

python run_pipeline.py --source data/videos/cars.mp4
python run_pipeline.py --source 0 --display    # webcam, live window
python run_pipeline.py --device 0              # GPU
python run_pipeline.py --only traffic incident # a subset of modules
python run_pipeline.py --disable anpr

python run_backend.py                          # http://127.0.0.1:8000

./demo.sh          /  demo.bat                 # everything, one command
python scripts/run_demo.py --multi-bus         # the cross-bus story
```

Config precedence: `platform.yaml` < `UIP_*` environment < CLI flag.

---

## M. Testing procedure

**Per module, in isolation** (`tests/test_modules_smoke.py`) — each module is
constructed, set up against the real weights, and run on real frames:

- [x] Vehicle detection — Person 2's detector loads, detects, returns the documented dict
- [x] Vehicle tracking — track ids persist across frames
- [x] Road defect detection — Person 1's checkpoint loads, 7 classes map to canonical names
- [x] Pedestrian detection — real model if present, flagged fallback otherwise
- [x] Traffic analytics — density/congestion/bottleneck produce valid Events
- [x] Incident detection — `tests/test_incident_engine.py`, synthetic trajectories for rash driving, collision, near-miss and hit-and-run geometry
- [x] ANPR — reads a real vehicle crop; and a separate test asserts `process()` returns `[]` for a stream of frames, i.e. **ANPR can never run per frame**
- [x] Event intelligence — `tests/test_intelligence_engine.py`, dedup radius, multi-bus correlation, reopen/resolve transitions, priority ordering, routing table
- [x] Event schema — `tests/test_event_schema.py`, round-trip, defaults, unknown-field capture into `metadata`
- [x] Backend — endpoints exercised against a seeded database
- [x] Frontend — loads against the live API and renders every tab

**Then the complete pipeline:**

```bash
python -m pytest tests -q          # 33 passed
python run_pipeline.py --source data/videos/cars.mp4
python run_backend.py              # open http://127.0.0.1:8000
```

**Verified on the integrated system:**

| Check | Result |
|---|---|
| Full suite | 33 passed, ~70 s on CPU |
| End-to-end run, 266 frames | 42 events across 8 types, 4.4 fps on CPU |
| All five frame modules | READY, 0 errors |
| ANPR invocations | 2 — both triggered by hit-and-run, none per-frame |
| Backend | all endpoints 200, dashboard renders with live evidence images |
| `vendor/` unchanged | `diff -r` against the original zips: identical |
| Cross-bus correlation | see below |

**The multi-bus check** (`scripts/run_demo.py --multi-bus`) — the same clip
replayed as three buses:

```
after BUS-01   issues=11  opened=11  merged=3
               HNR-001  truck  obs=1  buses=1  P=70.0 (HIGH)     → Traffic Police
after BUS-02   issues=11  opened=0   merged=14
               HNR-001  truck  obs=2  buses=2  P=73.6 (HIGH)     → Traffic Police
after BUS-03   issues=11  opened=0   merged=14
               HNR-001  truck  obs=3  buses=3  P=77.1 (CRITICAL) → Traffic Police

798 frames, 62 events, 11 issues.
```

The issue count stays at **11** across all three passes while observations and
bus count rise 1 → 2 → 3, and priority climbs with persistence until the issue
crosses into CRITICAL. That is Module 6 doing exactly what it exists for:
three buses reporting one problem produce **one** issue, not three.

---

## N. Demo procedure

```bash
./demo.sh                 # Linux/macOS
demo.bat                  # Windows
```

Which runs: setup check → pipeline over the clip → SQLite + evidence +
annotated video → backend on :8000 → dashboard opens.

For a presentation, the more convincing version is:

```bash
python scripts/run_demo.py --source data/videos/cars.mp4 --multi-bus
```

and then, on screen:

1. **Overview** — events by type, and the department routing table filling in.
2. **Incidents** — a hit-and-run row with its evidence crop.
3. **Number Plates** — the registration number for that same track id, and
   the point that ANPR ran twice in the whole video, only because incidents
   asked it to.
4. **Persistent Issues** — `HNR-001`, observations 3, buses 3. Say the line:
   *three buses saw this, the system knows it is one problem.*
5. **Map / Heatmap** — the segments, coloured by what happened on them.
6. The console table at the end of the run — per-module status, events,
   ms/frame — as the answer to "what happens when a module breaks".

The demo needs no Jetson, no bus, no hardware. A laptop and an `.mp4`.

---

## What is deliberately not done

* **No Jetson, no TensorRT/ONNX/FP16.** Later phase, by instruction.
* **No pedestrian weights invented.** Person 3's trained file was never
  shipped; the fallback is real detection from a real model with the
  adult/child split flagged as a heuristic in every event. Dropping
  `models/pedestrian_best.pt` in switches to their model with no code change.
* **No calibrated collision detection.** A single uncalibrated 2D camera
  cannot distinguish contact from occlusion, so overlap alone is only ever
  `near_miss`; `collision` requires kinematic corroboration. Stereo or a
  calibrated homography would change this, and neither exists yet.
* **GPS is simulated** along a configured route — there is no bus hardware in
  this phase. `core/geo.py` is the one file a real NMEA reader replaces.

## Open questions for the team

1. **Person 1** — what does class `D50` mean? Currently mapped to
   `road_damage_other`. Also: `Waterlogging` is in the spec but not in the
   checkpoint; the config mapping is already waiting for it.
2. **Person 3** — where are the trained pedestrian weights?
3. **Person 2** — `maximum_vehicles: 20` and `free_flow_speed_px_s: 60` need
   calibrating per camera before the congestion levels and delay estimates
   mean anything in minutes.
4. **Everyone** — a few hundred frames of real bus footage would let the
   incident thresholds be tuned against reality rather than dashcam clips.
