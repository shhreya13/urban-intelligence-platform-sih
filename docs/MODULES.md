# The modules

One rule ran through all of this: **nothing in `vendor/` was edited.** Each
team member's repository sits there exactly as they published it and still
runs standalone. What follows is what each adapter in `modules/` reuses, what
it adds, and where to extend it.

Every module exposes the same interface (`core/module_base.py`):

```python
class MyModule(BaseModule):
    name  = "my_module"
    owner = "Person N"

    def setup(self)                 -> None                # load, or raise ModuleUnavailable
    def process(self, ctx)          -> List[Event]         # per frame
    def finalize(self)              -> List[Event]         # end of video flush
    def close(self)                 -> None
```

---

## Module 1 — Road / infrastructure detection · Person 1

**Files** `modules/perception/road_defect.py`, `modules/road_detection/module.py`

**Reused unmodified** — `vendor/person1_road_defect/weights/best.pt`, one
multi-class detector, exactly as the spec required ("do not create separate
models for every road defect"). Copied to `models/road_defect_best.pt`.

**Class mapping.** Their checkpoint's raw names are mapped to the platform
vocabulary in `config/platform.yaml → road_defect_classes`:

```
Pothole → pothole            Damaged road → damaged_road      D50 → road_damage_other
Repair  → road_repair        Divider      → divider           Zebra Crossing → zebra_crossing
Traffic Sign → traffic_signboard
```

When Person 1 retrains with `Waterlogging`, **only that YAML block changes**
— the mapping already contains the entry, waiting.

**What the adapter adds**

* Per-detection Events with class, confidence, bbox, timestamp, frame number
  and GPS — the exact contract in the spec.
* Within-pass deduplication: a pothole visible for 40 frames is one event,
  not 40 (`emit_cooldown_s`). Cross-*bus* dedup is Module 6's job, not this
  one's.
* **Missing infrastructure by absence.** No detector can see a thing that
  isn't there, so instead of inventing a model the module keeps a
  per-road-segment baseline: once a segment has been observed to contain a
  zebra crossing or divider on several passes, a later pass with none raises
  `missing_zebra_crossing` / `missing_divider`. Switch off with
  `infrastructure_absence.enabled: false`.
* Strided inference (`frame_stride: 5`) — defects are static, so there is no
  reason to run this model on every frame.

> **Open question for Person 1:** class `D50` is an RDD-style code. It is
> currently mapped to `road_damage_other`; confirm the intended meaning and
> the YAML line can be corrected in one edit.

---

## Module 2 — Traffic intelligence · Person 2

**Files** `modules/perception/vehicle.py`, `modules/traffic/module.py`

**Reused unmodified**, loaded straight out of `vendor/person2_traffic/src/`:

| Their file | Their class | Used for |
|---|---|---|
| `detection/vehicle_detector.py` | `VehicleDetector` | YOLO + BoT-SORT — **the** vehicle source for the whole platform |
| `analytics/speed_estimator.py` | `SpeedEstimator` | per-track speed |
| `analytics/traffic_density.py` | `TrafficDensity` | density %, LOW/MEDIUM/HIGH/CRITICAL |
| `analytics/congestion.py` | `CongestionAnalyzer` | congestion state + duration |
| `analytics/bottleneck.py` | `BottleneckAnalyzer` | zone grid, bottleneck score |
| `analytics/peak_traffic.py` | `PeakTrafficAnalyzer` | session peak |
| `analytics/vehicle_counter.py` | `VehicleCounter` | line-crossing counts |

No second ML model was introduced — as the spec required.

**What the adapter adds** — only the parts their standalone `main.py` drew on
screen instead of exporting: Events, per-road-segment route analytics,
estimated delay (from `free_flow_speed_px_s`), and heatmap-ready rows.

**Events**: `traffic_observation` (a heartbeat every
`observation_interval_s`, this is what feeds the heatmap),
`traffic_congestion` (with `congestion_duration`, `estimated_delay`),
`traffic_bottleneck` (with `bottleneck_score`).

**Calibration.** `maximum_vehicles` is "what counts as 100 % density for this
camera" and `free_flow_speed_px_s` converts pixel speed into a delay in
minutes. Both are per-camera and both are in the config, not the code.

---

---

## Module 3 — Pedestrian safety / vulnerable pedestrians · Person 3

**Files** `modules/perception/pedestrian.py`, `modules/pedestrian/module.py`,
`modules/pedestrian/crossing_logic.py`

**Requirement (SIH26124).** "Identify vulnerable pedestrian situations such
as school children crossing roads."

### PERSON ≠ PEDESTRIAN

The earlier integration labelled every detected `person` inside a fixed,
hand-drawn image rectangle (`zones.crossing_zone`) as a pedestrian and scored
it against nearby vehicles. People on the pavement, at a bus stop or inside a
bus could raise pedestrian alerts, and the "crossing" was never a real one.

The perception adapter now publishes detections as `ctx.shared["persons"]`.
Module 3 is the only writer of `ctx.shared["pedestrians"]`, and a person gets
there only when **all** of these hold for the same tracker id over several
consecutive frames:

| # | Condition | How it is decided |
|---|---|---|
| 1 | person detected and tracked | Person 3 perception + BoT-SORT id |
| 2 | a zebra crossing is **detected** | Person 1's ROAD_DEFECT_MODEL class `zebra_crossing`, seen on ≥ `min_detections` strided passes; no image rectangle is ever assumed to be a crossing |
| 3 | the person's **feet** are on it | bottom 20% of the person box ≥ 50% inside the crossing box, foot point inside it |
| 3b | not on/in a vehicle | feet inside a vehicle box above its ground line = passenger or rider, rejected |
| 4 | vehicles present **and moving** | net frame-to-frame displacement of vehicle track ids over ~0.5 s, scale-aware threshold; confidence, presence and the upstream `speed` field are not used as a substitute |
| 5 | crossing / occupying the carriageway, confirmed over time | `confirm_frames` consecutive frames with 1-4 all true |

Then, and only then, one `pedestrian_vehicle_risk` event with
`metadata.event_label = "VULNERABLE_PEDESTRIAN_EVENT"` is emitted.

**One crossing, one event.** `CrossingStateMachine` keeps a per-track phase
(`IDLE → CANDIDATE → CONFIRMED`). Staying on the crossing never re-emits.
The episode closes after `exit_frames` consecutive frames off the crossing or
`track_lost_frames` undetected; a later crossing may raise a new event.
Stepping back on within `reentry_cooldown_frames` of a close resumes the same
crossing. A tracker id switch mid-crossing (lost track, new id at the same
spot) inherits the confirmed crossing instead of re-alerting. Frame counts
scale from the video fps unless set.

**Bus camera motion.** On a moving bus, parked cars slide across the image
too. The crossing is painted on the road, so its own image velocity
(least-squares over recent detections, with standard error) is the camera's
motion at road level. Vehicle displacement is measured relative to it, and a
vehicle must out-move the estimate by more than that estimate's own
uncertainty. Until the crossing has enough detections to estimate this, no
vehicle is judged to be moving. Switch off with
`vehicle_motion.ego_motion_compensation: false`.

**UI roles** on each person: `person` (grey, `PERSON`), `crossing` (cyan,
`PERSON / CROSSING`: on a detected crossing, rule not met), `pedestrian`
(red, `PEDESTRIAN` + `VULNERABLE PEDESTRIAN EVENT`). Tracked crossings are
drawn as a white outline.

**Children.** The shipped detector is COCO `person` only, so events say
`pedestrian_category: "vulnerable pedestrian"` and `age_group: null`. Only a
dedicated checkpoint emitting class `child` yields `age_group: "child"`. The
platform never claims "school child".

**Reused unmodified** from `vendor/person3_pedestrian_safety/src/`:
`risk_engine.py` → `RiskEngine` (position history, closing speed, additive
score, LOW..CRITICAL). It now sets the **severity** of a confirmed event
(floor `severity_floor: HIGH`); it no longer decides whether an event exists.
`geometry.py` is no longer needed (bbox geometry is in `crossing_logic.py`).

**Known limitations, stated plainly.**
* The road model is a detector, so the crossing is an axis-aligned box. On a
  strongly diagonal crossing the box includes some kerb at its corners.
  `min_foot_overlap` and the foot-point test limit this; a segmentation
  model would remove it.
* Moving vehicles are judged by image-plane displacement near the crossing,
  not physical speed. `min_displacement_ratio` is relative to each vehicle's
  own box height so it transfers across resolutions.
* Ego-motion compensation is a translation model anchored on the crossing.
  Strong perspective change very close to the crossing can make it less
  accurate; its uncertainty margin errs toward *no* event.
* A car moving at exactly the bus's speed right next to the camera is
  handled correctly only when the crossing is visible (it is the reference).
* No child-aware checkpoint ships. Train one with
  `vendor/person3_pedestrian_safety/train.py` and point
  `perception.pedestrian.model` at it; nothing else changes.

**Tests** `tests/test_vulnerable_pedestrian.py` (the 12 required scenarios
plus bus-passenger, pavement-behind-crossing, single-frame, id-switch,
noisy-detector and moving-bus cases).

---

## Module 4 — Incident detection and hit-and-run · Person 4

**File** `modules/incident/module.py`

**Reused unmodified**, from `vendor/person4_incident/`:

| Their file | Their class | Used for |
|---|---|---|
| `motion_analysis.py` | `MotionAnalyzer` | per-track speed, acceleration, lateral movement |
| `rash_driving.py` | `RashDrivingDetector` | rule-based, requires 2+ corroborating signals |
| `collision_detection.py` | `CollisionDetector` | multi-signal collision confirmation |
| `hit_and_run.py` | `HitAndRunDetector` | post-collision escape monitoring |

**No new ML model.** Everything is reasoning over Person 2's vehicle boxes,
BoT-SORT track ids and speeds.

### Why the previous hit-and-run was replaced

The old implementation emitted **2 hit-and-runs on 266 frames of ordinary
dashcam traffic containing no crash**. Four concrete defects:

1. **Disappearance was treated as escape.** `gone = track not in frame` after
   ~10 frames *raised* the confidence and still emitted, so an id switch or a
   vehicle leaving the camera view produced a hit-and-run.
2. **"Did not stop" was used as "fled".** The only test was
   `min_speed_after < 4 px/s` — every moving vehicle passes it.
3. **Overlap alone started the watch.** Two boxes overlapping with a speed
   wobble counted as a corroborated collision.
4. **The flee threshold was inert.** `flee_speed_ratio` was only ever used as
   a fallback value, never compared against anything.

Person 4's detector fixes all four, and its own findings report documents the
false positives it was hardened against: near-field occluders, co-moving
traffic, and YOLO box jitter / blinking ids.

### The state machine

```
NORMAL
  → SUSPICIOUS_INTERACTION      loose detector flagged the pair
  → POSSIBLE_COLLISION          loose confirmation
  → POST_COLLISION_MONITORING   strict confirmation, track locked,
                                incident point recorded
  → VEHICLE_LEAVING             consistent movement away from that point
  → HIT_AND_RUN_CONFIRMED       sustained departure, ANPR requested

Terminal states that emit NO hit-and-run:
  COLLISION_ONLY   the vehicle stayed near the incident point
  TRACK_LOST       the track vanished without a departure run
```

Every transition is logged (`[HIT_RUN] track 31 …`) and stored on the event as
`metadata.timeline`, which is what the dashboard's investigation view renders.

### Two confirmation tiers

The same algorithm runs twice with different confirmation bars. The strict
tier emits `collision`; the loose tier emits `possible_collision` for pairs
the strict tier did not confirm. Every false-positive guard applies to both,
so "possible" means *partial evidence*, not *no guards*.

### Track-loss handling

`track_loss_timeout` (default 30 frames) is how long a locked track may be
absent. Exceeding it resolves the case to `TRACK_LOST` and emits nothing —
this is the single most important rule in the module and it has its own test.

### Evidence

On a confirmed collision the module saves the frame before (from a rolling
ring buffer), the impact frame, the vehicle crop, then captures N frames
after and writes a short clip. Paths land in `metadata.evidence_frames`.

---

## Module 5 — ANPR / number plate · Person 5

**File** `modules/anpr/module.py`

**Reused unmodified** — `vendor/person5_anpr/anpr_pipeline.py`. Their plate
YOLO, their image enhancement, their EasyOCR call, their multi-frame voting,
their format-correction tables. The adapter calls exactly one method:

```python
ANPRPipeline.process_incident(track_id, vehicle_frames)
```

**ANPR never runs per frame.** `ANPRModule.process()` returns `[]`
unconditionally — there is no code path that OCRs during normal driving.
The module is reached only through `identify()`, which the orchestrator calls
when an incident event carries `requires_identification`.

```
incident → track_id → buffered crops → plate detect → plate crop
        → OCR → multi-frame vote → registration_number + confidence
        → best plate image saved under outputs/evidence/
```

A low-confidence read is published as `UNKNOWN`, never as a guessed
number (`min_plate_confidence`).

Gates in config: `trigger_event_types`, `min_trigger_severity`,
`max_frames_to_ocr` (how many buffered crops to spend), `min_sharpness`
(blurry crops are skipped before OCR), `max_identifications` (a safety valve
for a long video).

This is also the compute budget the Jetson phase depends on: the most
expensive model in the system runs a handful of times per hour, not 30 times
per second.

---

## Module 6 — Central intelligence / event engine · Person 6

**Files** `modules/intelligence/module.py`, `priority.py`, `routing.py`

Person 6's slot arrived empty; this is plain Python, no Nano-specific
anything, as the spec asked for this phase. Seven responsibilities:

**1. Normalisation** — fill `road_segment`, clamp severity, canonicalise
class names, guarantee every field the backend expects exists.

**2. Deduplication** — the same defect on 40 frames of one pass is one
observation.

**3. Multi-bus correlation** — BUS-01, BUS-02 and BUS-03 reporting a pothole
at GPS A all attach to `POT-001`. Matching is by issue type + GPS proximity
(`dedup.gps_radius_m`, default 35 m) + road segment + time window.

**4. Persistent issues** — `POT-001`, `first_seen`, `last_seen`,
`observations`, `buses`. Issue ids are typed and readable: `POT`, `DMG`,
`WTR`, `COL`, `PCL`, `HNR`, `RSH`, `ABN`, `CNG`, `INF`.

**5. Repair verification** — a bus that passes through the segment and does
*not* see the defect scores a **clean pass**; `resolve_after_clean_passes`
of them closes the issue as RESOLVED; a later sighting flips it to REOPENED
with the history preserved.

**6. Priority scoring** — the four spec inputs, each normalised to 0..1 and
combined with configured weights:

```
priority = 0.35·severity + 0.25·persistence + 0.20·traffic_impact + 0.20·safety_risk
```

Every component is returned alongside the score, so the dashboard can explain
*why* something is priority 82 instead of just asserting it.

**7. Department routing** — data, not code (`routing.py`, overridable in
config under `departments:`):

| | |
|---|---|
| Pothole, damaged road, missing divider | Roads/PWD |
| Waterlogging | Drainage |
| Congestion, bottleneck, zebra crossing, signboard | Traffic |
| Rash driving, collision, hit-and-run, abnormal movement | Traffic Police |
| Bus delay | Transport |

**State lives in SQLite**, not in memory — so correlation works across
pipeline runs. Process bus 1's footage today and bus 2's tomorrow; the second
run attaches to the issues the first one opened.

---

## Adding a seventh module

1. `modules/my_module/module.py`, subclass `BaseModule`, set `name` and `owner`.
2. Raise `ModuleUnavailable` in `setup()` if a dependency is missing — never crash.
3. Return `Event` objects from `process()`; never import another module.
4. Add a `modules.my_module:` block to `config/platform.yaml`.
5. Register it in `pipeline/orchestrator.py` → `self.frame_modules`.
6. Add a smoke test to `tests/test_modules_smoke.py`.

If your module needs a model that already exists in `perception/`, read it
from the `FrameContext` — do not load a second copy.
