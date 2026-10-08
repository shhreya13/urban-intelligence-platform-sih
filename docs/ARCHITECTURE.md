# Architecture

## 1. Where the project started

Six people, six independent repositories, no shared vocabulary:

| Owner | What arrived | Entry point | Its own idea of "a detection" |
|---|---|---|---|
| Person 1 | `road-defect-model` — YOLO checkpoint + a 30-line `predict.py` | `inference/predict.py` | ultralytics `Results` object, printed |
| Person 2 | `traffic-intelligence-phase1` — detector, BoT-SORT, 5 analytics classes | `src/main.py` | `dict(bbox, class_name, confidence, track_id, speed, …)` |
| Person 4 | `incident_detection` — collision, hit-and-run, rash driving, motion analysis | `test_full_pipeline.py` | `dict(incident_type, vehicle_track_id, confidence, …)` |
| Person 5 | `anpr_module` — plate YOLO + EasyOCR + multi-frame voting | `run.py` | `dict(track_id, plate, confidence)` |
| Person 6 | *(nothing — the slot was empty)* | — | — |

Four independent problems followed from that:

1. **Two vehicle models.** Person 2 loaded one and Person 4 vendored a
   second (YOLOv8n + its own BoT-SORT) so their module could run standalone.
2. **No common output.** Every module printed or drew; nothing was
   machine-readable, so nothing could be correlated.
3. **ANPR was per-frame.** `run.py` OCR'd every vehicle it saw — the single
   most expensive thing in the system, run continuously.
4. **No downstream.** No database, no API, no dashboard, no notion of the
   same pothole being seen twice.

## 2. What integration changed

Nothing inside `vendor/`. Every original repo sits there unmodified and still
runs standalone. Integration is three new layers around them:

```
   core/       the contract       Event · FrameContext · BaseModule · Config
   modules/    the adapters       one thin wrapper per owner
   pipeline/   the orchestrator   the only file that knows the order
```

### 2.1 The contract — `core/`

| File | Job |
|---|---|
| `event.py` | **The** common event format. One dataclass, every module. |
| `frame_context.py` | The per-frame object: the frame + bus/camera/GPS/time + the **shared** vehicle list. |
| `module_base.py` | `setup / process / finalize / close`, wrapped in `safe_*` so one module's exception is logged and skipped, never fatal. |
| `config.py` | `platform.yaml` + `UIP_*` env + CLI overrides, merged. Every path resolved against the project root, so the same config works on Windows and Linux. |
| `vendor_loader.py` | Imports a file out of `vendor/` by path, without requiring the team repos to be installable packages. |
| `geo.py` | GPS provider (simulated / fixed / none), haversine, road-segment quantisation. |
| `evidence.py` | Writes evidence crops under `outputs/evidence/<event_type>/`. |
| `track_buffer.py` | Per-track rolling vehicle crops — this is what ANPR reads after an incident. |
| `sinks.py` | Where events go: JSONL, SQLite direct, or HTTP POST to the backend. |

### 2.2 Shared perception — `modules/perception/`

The single most important integration decision: **each model is loaded once
per run and read by everyone.**

```
frame ──► VehiclePerception       (Person 2's VehicleDetector + BoT-SORT + SpeedEstimator)
      ──► RoadDefectPerception    (Person 1's ROAD_DEFECT_MODEL, strided)
      ──► WaterloggingPerception  (dedicated single-class model, strided)
                     │
                     ▼
   FrameContext.vehicles  /  FrameContext.shared["road_defects"]
                     │
     read by Traffic (P2) · Incidents + hit-and-run (P4) · road_detection (P1)
```

`RoadDefectPerception` and `WaterloggingPerception` both write into the same
`shared["road_defects"]` list (the second one appends rather than
overwrites), so the road module, the annotator, and the intelligence engine
never need to know a detection came from a separate checkpoint — a
`waterlogging` hit is just another road-defect class in that list.

Consequences: one vehicle model instead of three; consistent `track_id`s
across modules — the id Person 4 flags for hit-and-run is the same id
Person 3 saw approaching a child, and the same id ANPR reads a plate for;
and the per-frame cost is bounded no matter how many modules are enabled.

### 2.3 The orchestrator — `pipeline/orchestrator.py`

Per frame, in this order:

```
 1. video_source        decode, optional resize, frame number + video time
 2. GPS provider        position for this frame, and its road segment
 3. shared perception   vehicle (every frame)
                        road defect (every Nth frame — defects are static)
 4. crop buffer         keep the last N crops per track, for later ANPR
 5. road_detection      Person 1 → Events
 6. traffic             Person 2 → Events
 8. incident            Person 4 → Events, some carrying
                                   metadata.requires_identification
 9. ANPR                Person 5 → only for those events, using the buffered crops
10. intelligence        Person 6 → normalise, dedup, correlate, persist,
                                   score, route
11. sinks               JSONL + SQLite (+ annotated video frame)
```

At end of video, `finalize()` lets a module flush what it was still
accumulating (an open congestion window, a hit-and-run watchlist).

**Modules never call each other.** They read `FrameContext`, they return
`List[Event]`. The orchestrator is the only component that knows there is
more than one module.

### 2.4 Event-triggered ANPR

This is the flow the spec asked for, and it is enforced structurally rather
than by convention: `ANPRModule.process()` returns `[]` unconditionally.
There is no code path that OCRs a frame during normal driving.

```
normal driving
    vehicle detected → tracked → crops buffered → (nothing else)

incident
    IncidentModule emits e.g. hit_and_run with
        metadata.requires_identification = {track_id: 24, reason: "hit_and_run"}
              ▼
    orchestrator pulls track 24's buffered crops from VehicleCropBuffer
              ▼
    ANPRModule.identify(24, crops)
        → Person 5's ANPRPipeline.process_incident()
        → plate detect → crop → enhance → EasyOCR → multi-frame vote
              ▼
    Event(vehicle_identification, registration_number, plate_confidence,
          best plate image saved to evidence/)
```

Only events whose type is in `modules.anpr.trigger_event_types` and whose
severity is at least `min_trigger_severity` get there, and
`max_identifications` caps a long run.

### 2.5 Intelligence — turning events into issues

The event stream is noisy by nature: one pothole is 40 frames, three buses,
five passes. Module 6 collapses that into issues.

```
event ─► normalise ─► is there an OPEN issue of this type
                      within gps_radius_m of this point?
                        │yes                        │no
                        ▼                           ▼
              attach as an observation        open a new issue
              (obs++, buses++, last_seen)      POT-001 / PED-003 / HNR-002
                        │                           │
                        └──────────┬────────────────┘
                                   ▼
                    priority = severity · persistence ·
                               traffic impact · safety risk
                                   ▼
                    department routing → Roads/PWD, Drainage,
                                         Traffic, Traffic Police, Transport
```

Repair verification runs on the same structure in reverse: a bus that passes
through a segment and does *not* see the defect scores a **clean pass**;
`resolve_after_clean_passes` clean passes close the issue as RESOLVED; a
later sighting flips it to REOPENED with the count preserved.

Issue state lives in SQLite, not in memory, so correlation works **across
runs**: process bus 1's footage today and bus 2's tomorrow, and the second
run attaches to the issues the first one opened.

## 3. Failure behaviour

The design assumption is that on a real bus, something is always broken.

| Failure | What happens |
|---|---|
| A model file is missing | `setup()` raises `ModuleUnavailable` → module marked UNAVAILABLE, logged with the exact path, pipeline runs without it |
| A module throws on one frame | caught in `safe_process`, logged with the module name and frame number, that frame's events are dropped, everything else continues |
| A module throws repeatedly | after 10 consecutive errors it is marked FAILED and skipped for the rest of the run, so it cannot spam the log |
| A module is switched off in config | marked DISABLED, never constructed |
| The database is unreachable | JSONL sink still receives everything |

Every run ends with a per-module table — status, events emitted, errors,
ms/frame — so "which module failed" is never a guess.

## 4. The multi-department platform layer

The pipeline ends at the common event format. Everything below that point is
the platform layer, added without changing how detection works.

```
        INTELLIGENCE ENGINE  (dedup, persistence, priority, routing)
                                  |
                         events + issues in SQLite
                                  |
                          WORKFLOW  (backend/workflow.py)
              one real-world problem -> one managed event EV-xxxx
              DETECTED -> VERIFIED -> ASSIGNED -> IN_PROGRESS -> RESOLVED
              activity log, notifications, re-open on re-detection
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
   RBAC (backend/rbac.py)   EVIDENCE INDEX          FLEET (backend/fleet.py)
   roles, ownership,        (backend/recordings.py)  buses, trails,
   visibility, pages,       links an event to the    route coverage
   map layers               pipeline's own
        |                   annotated recording
        |                         |
        +-------------------------+
                                  |
                    ONE API  (backend/app.py + platform_routes.py)
                    every response filtered by the caller's scope
                                  |
                  +---------------+---------------+
                  |                               |
          WEB  frontend/index.html         MOBILE  mobile/index.html
          Central Command + 4              Command & Field app (PWA),
          department dashboards            per-role bottom navigation
```

Design points that matter later:

* **One rules file.** `backend/rbac.py` holds every role, ownership and
  visibility rule. Adding a department is a table entry, not a rewrite.
* **The server decides.** Both apps read permissions only to draw their
  navigation; every endpoint filters again on the server, so a tampered URL
  changes nothing.
* **Human decisions live in their own tables.** `cases`, `case_activity` and
  the rest are separate from `events` and `issues`, so a new pipeline run can
  add sightings but can never overwrite what a person did.
* **Evidence is never regenerated.** The platform extracts frames from the
  recordings the pipeline already annotated. No model is loaded by the
  backend.
* **The mobile app is the same API.** One backend, one database, one session
  cookie, no sync layer between the apps.

Details in [PLATFORM.md](PLATFORM.md).

---

## 5. What the Jetson phase will change

Deliberately isolated, so that phase is additive:

* `perception/*` — swap `.pt` for `.engine`. The adapters already take the
  model path from config; nothing above them changes.
* `platform.device` — already the only place the device is named.
* `frame_stride` per model — already config, already how road defect runs.
* `backend.mode: http` — already supported, for a bus that ships events to a
  central server rather than holding its own SQLite.

Nothing in `core/`, `modules/*/module.py`, `backend/` or `frontend/` needs to
know that a Jetson exists.
