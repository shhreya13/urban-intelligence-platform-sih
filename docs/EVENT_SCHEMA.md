# The common event format

`core/event.py` defines one dataclass. Every module produces it, the database
stores it, the API serves it, the dashboard renders it. **Nothing else
crosses a module boundary** — modules do not import each other, they exchange
`Event` objects through the orchestrator.

Two rules shape it:

* A module is never forced to fill a field it does not have. Everything
  except `event_type` and `source_module` has a default.
* Anything module-specific lives in `metadata`, so the schema never grows a
  per-module column.

---

## Fields

| Field | Type | Filled by | Meaning |
|---|---|---|---|
| `event_id` | str | auto | uuid4 hex |
| `event_type` | str | **required** | see the vocabulary below |
| `source_module` | str | **required** | `road_detection` · `traffic` · `incident` · `anpr` · `intelligence` |
| `bus_id` | str? | orchestrator | e.g. `BUS-01` |
| `camera_id` | str? | orchestrator | e.g. `CAM-FRONT` |
| `timestamp` | str | auto | ISO-8601 UTC, millisecond precision |
| `gps` | object? | orchestrator | `{latitude, longitude, accuracy_m, speed_kmh, heading_deg}` |
| `confidence` | float | module | 0.0 – 1.0 |
| `severity` | str | module | `INFO` `LOW` `MEDIUM` `HIGH` `CRITICAL` |
| `track_id` | int? | module | BoT-SORT id — **the same id across all modules** |
| `class_name` | str? | module | canonical class, e.g. `pothole`, `car`, `truck_car` (a collision pair) |
| `bounding_box` | [int]? | module | `[x1, y1, x2, y2]` in pixels |
| `evidence_image` | str? | module | path relative to `outputs/evidence/` |
| `evidence_video` | str? | module | clip path, when one was saved |
| `frame_number` | int? | orchestrator | |
| `video_time_s` | float? | orchestrator | seconds into the clip |
| `road_segment` | str? | orchestrator | quantised segment id, e.g. `SEG-13.08490-80.27351` |
| `issue_id` | str? | **intelligence** | `POT-001` — null before Module 6 |
| `priority_score` | float? | **intelligence** | 0 – 100 |
| `department` | str? | **intelligence** | `Roads/PWD`, `Drainage`, `Traffic`, `Traffic Police`, `Transport` |
| `duplicate_of` | str? | **intelligence** | the event this one duplicates |
| `metadata` | object | module | everything module-specific |

`with_context(ctx)` stamps bus / camera / GPS / frame fields from the
`FrameContext` **without overwriting anything the module set deliberately**,
so a module can always override a default.

---

## Event types

| `event_type` | Source | Emitted when |
|---|---|---|
| `road_defect` | road_detection | a defect or infrastructure item is detected (or inferred missing) |
| `traffic_observation` | traffic | heartbeat every `observation_interval_s` — feeds heatmap and analytics |
| `traffic_congestion` | traffic | density/speed thresholds held for `minimum_duration_seconds` |
| `traffic_bottleneck` | traffic | a zone in the grid sustains bottleneck conditions |
| `pedestrian_vehicle_risk` | pedestrian_safety | VULNERABLE PEDESTRIAN EVENT: a tracked person on a **detected** zebra crossing while vehicles are **moving**, temporally confirmed; one event per continuous crossing |
| `rash_driving` | incident | speed spike, direction change, lane change or weaving |
| `possible_collision` | incident | partial contact evidence — reported as possible, not confirmed |
| `collision` | incident | transient overlap from a separated state, corroborated by approach or impact motion |
| `abnormal_movement` | incident | two suspicious motion signals, below the rash-driving bar |
| `hit_and_run` | incident | a collision whose offending track does not slow down |
| `vehicle_identification` | anpr | a plate was read for an incident track |
| `persistent_issue` | intelligence | an issue was opened, updated, reopened or resolved |

---

## Per-module `metadata`

Only the module-specific parts are listed; every event also has the common
fields above.

### `road_defect` — Person 1

```json
"metadata": {
  "raw_class_name": "Damaged road",
  "model": "road_defect_best.pt",
  "area_px": 18420,
  "relative_area": 0.031
}
```

An *inferred* absence (missing divider / missing zebra crossing) instead
carries:

```json
"metadata": {
  "inferred": true,
  "detector": "baseline",
  "reason": "segment SEG-13.08490-80.27351 showed a zebra crossing on 4 earlier passes, none on this one",
  "baseline_observations": 4
}
```

### `traffic_congestion` / `traffic_observation` — Person 2

```json
{
  "event_type": "traffic_congestion",
  "severity": "MEDIUM",
  "metadata": {
    "traffic_level": "HIGH",
    "vehicle_count": 60,
    "density_percentage": 71.2,
    "average_speed_px_s": 22.4,
    "moving_vehicles": 38,
    "slow_vehicles": 22,
    "congestion_status": "CONGESTED",
    "congestion_duration": 600,
    "bottleneck_status": "BOTTLENECK",
    "bottleneck_score": 0.87,
    "bottleneck_zone": "r1c2",
    "estimated_delay": 8,
    "counts_by_class": {"car": 41, "motorcycle": 12, "bus": 3, "truck": 4},
    "counted_by_class": {"car": 88, "motorcycle": 31},
    "counted_total": 137,
    "unique_vehicles": 164,
    "peak_vehicles": 71,
    "road_segment": "SEG-13.08490-80.27351"
  }
}
```

`road_segment` on these rows is what `GET /heatmap` aggregates.

### `pedestrian_vehicle_risk` (Person 3, vulnerable pedestrian event)

```json
{
  "event_type": "pedestrian_vehicle_risk",
  "source_module": "pedestrian_safety",
  "severity": "HIGH",
  "confidence": 0.82,
  "track_id": 7,
  "class_name": "person",
  "bounding_box": [600, 330, 640, 470],
  "timestamp": "2026-09-17T09:12:44.120+00:00",
  "gps": {"latitude": 13.0827, "longitude": 80.2707},
  "frame_number": 18,
  "evidence_image": "vulnerable_pedestrian/BUS-01_000018_ab12cd.jpg",
  "metadata": {
    "event_label": "VULNERABLE_PEDESTRIAN_EVENT",
    "pedestrian_category": "vulnerable pedestrian",
    "age_group": null,
    "age_classification": "not available (detector only provides 'person')",
    "person_track_id": 7,
    "vehicle_track_ids": [11, 12],
    "moving_vehicles": [{"track_id": 11, "class_name": "car", "displacement_px": 60.0, "...": "..."}],
    "zebra_crossing_detected": true,
    "zebra_crossing_bbox": [200, 400, 1080, 520],
    "zebra_crossing_confidence": 0.82,
    "zebra_crossing_source": "ROAD_DEFECT_MODEL:zebra_crossing",
    "person_on_zebra_crossing": true,
    "person_on_vehicle": false,
    "vehicle_moving": true,
    "ego_motion_compensated": true,
    "crossing_confirmed": true,
    "crossing_motion": "occupying_crossing",
    "candidate_start_frame": 8,
    "confirmed_frame": 18,
    "confirmation_frames": 10,
    "concurrent_person_track_ids_on_crossing": [7],
    "confidence_components": {"person_mean": 0.88, "zebra_crossing": 0.82, "moving_vehicle_max": 0.9},
    "vehicle_track_id": 11,
    "distance_px": 80.0,
    "closing_speed_px_per_frame": 0.4,
    "approaching": false,
    "risk_score": 25.0,
    "risk_level": "MEDIUM"
  }
}
```

Emitted **once per continuous crossing** and only when every condition in
`docs/MODULES.md` → Module 3 holds. `confidence` is the weakest link of
person, crossing and moving-vehicle detection confidence. `risk_score` /
`risk_level` are Person 3's original additive score, now used only to raise
severity above the `severity_floor` (HIGH). `pedestrian_category` is never
"school child": `age_group` is `"child"` only when a child-aware detector
emits class `child`.

### `hit_and_run` / `collision` / `possible_collision` — Person 4

```json
{
  "event_type": "hit_and_run",
  "track_id": 31,
  "class_name": "truck",
  "confidence": 0.72,
  "severity": "CRITICAL",
  "metadata": {
    "incident_id": "INC-000004",
    "incident_type": "hit_and_run",
    "status": "suspected",
    "state": "HIT_AND_RUN_CONFIRMED",
    "vehicle_track_id": 31,
    "other_track_id": 18,
    "collision_event_id": "3b90…",
    "collision_frame_id": 24,
    "incident_location": {"latitude": 13.0876, "longitude": 80.2764},
    "escape_distance_px": 232.35,
    "escape_speed_px_s": 232.35,
    "escape_direction": "right",
    "progress": 0.69,
    "timeline": [
      {"state": "SUSPICIOUS_INTERACTION",    "frame": 20, "video_time_s": 0.67, "note": "…"},
      {"state": "POSSIBLE_COLLISION",        "frame": 24, "video_time_s": 0.80, "note": "…"},
      {"state": "POST_COLLISION_MONITORING", "frame": 24, "video_time_s": 0.80, "note": "…"},
      {"state": "VEHICLE_LEAVING",           "frame": 33, "video_time_s": 1.10, "note": "…"},
      {"state": "HIT_AND_RUN_CONFIRMED",     "frame": 54, "video_time_s": 1.80, "note": "…"}
    ],
    "evidence_frames": {
      "frame_before":   "collision/BUS-01_000021_ab12.jpg",
      "incident_frame": "collision/BUS-01_000024_cd34.jpg",
      "frame_after":    "hit_and_run/BUS-01_000048_ef56.jpg",
      "vehicle_crop":   "hit_and_run/BUS-01_000054_78ab.jpg",
      "clip":           "hit_and_run/BUS-01_000048_90cd.mp4"
    },
    "detector": "person4.HitAndRunDetector",
    "requires_identification": true,
    "anpr_track_id": 31
  }
}
```

A `collision` carries the contact measurements instead:

```json
"metadata": {
  "status": "confirmed",
  "confirmation_tier": "strict",
  "vehicle_track_ids": [31, 18],
  "iou": 0.52, "approach": 1.0, "motion_change": 0.29,
  "impact_frame": 24,
  "detector": "person4.CollisionDetector",
  "requires_identification": false
}
```

`metadata.requires_identification` (a bool) plus `anpr_track_id` is the
**only** thing that makes ANPR run. A confirmed collision on its own does
**not** set it — only a hit-and-run, or a `HIGH`-severity `rash_driving`
event when `rash.trigger_anpr_on_high` is on.

A `synthetic: true` key means the record came from `--demo-incident` or
`scripts/demo_hit_and_run.py`. It is a demonstration, not a detection, and
the dashboard badges it as such.

### `vehicle_identification` — Person 5

```json
{
  "event_type": "vehicle_identification",
  "track_id": 24,
  "confidence": 0.94,
  "metadata": {
    "registration_number": "TN12AB1234",
    "plate_confidence": 0.94,
    "plate_valid_format": true,
    "vehicle_track_id": 24,
    "frames_used": 5,
    "votes": {"TN12AB1234": 4, "TN12A81234": 1},
    "raw_candidates": ["TN12AB1234", "TN12A81234"],
    "engine": "easyocr",
    "triggered_by_event_id": "9c218d70476a…",
    "triggered_by_event_type": "hit_and_run"
  }
}
```

### `persistent_issue` — Person 6

```json
{
  "event_type": "persistent_issue",
  "issue_id": "POT-001",
  "priority_score": 78.4,
  "department": "Roads/PWD",
  "metadata": {
    "issue_state": "OPENED",         // OPENED | UPDATED | REOPENED | RESOLVED
    "issue_id": "POT-001",
    "issue_type": "pothole",
    "first_seen": "2026-09-09T04:52:43.775+00:00",
    "last_seen":  "2026-09-11T08:14:02.110+00:00",
    "observations": 12,
    "buses": ["BUS-01", "BUS-02", "BUS-03", "BUS-07"],
    "bus_count": 4,
    "clean_passes": 0,
    "reopened_count": 1,
    "repair_observed": false,
    "department": "Roads/PWD",
    "priority_score": 78.4,
    "priority_band": "HIGH",
    "triggering_event_id": "9c218d70476a…"
  }
}
```

---

## Producing an event from a module

```python
from core.event import EventType, Severity, SourceModule, make_event

def process(self, ctx):
    return [
        make_event(
            EventType.ROAD_DEFECT,
            SourceModule.ROAD,
            class_name="pothole",
            confidence=0.87,
            severity=Severity.HIGH,
            bounding_box=[x1, y1, x2, y2],
            metadata={"raw_class": "Pothole", "area_px": 18420},
        )
    ]
```

`bus_id`, `camera_id`, `gps`, `frame_number`, `video_time_s` and
`road_segment` are stamped by the orchestrator — do not set them unless you
mean to override.

---

## Storage

`Event.to_dict()` is written to `outputs/runs/events.jsonl` (one JSON object
per line, append-only) and to the SQLite `events` table, whose columns mirror
the schema one-for-one with `gps` flattened to `latitude` / `longitude` and
`metadata` stored as a JSON blob. Consequence: `sqlite3` and `jq` both work
on the output with no extra tooling.

---

## Managed events (the platform layer)

The schema above is the pipeline's output and is unchanged. The
multi-department platform never edits it: it builds a separate row, the
**managed event** (`EV-1001`, `EV-1002`, ...), on top.

```
detections (events)  ->  issues (intelligence engine)  ->  cases (workflow)
   149 rows                    31 rows                       24 rows
   "what a camera saw"     "one real-world problem"      "one item of work"
```

A managed event carries the identity of the problem (type, class, title,
owning department, severity, priority, location, buses, sighting count,
first and last seen) plus the human part: status, who verified it, who it is
assigned to, when work started, the resolution note, and how many times it
re-opened. `case_activity` records every step with actor, time, from-status,
to-status and note. `case_issues` records which engine issues a managed
event is made of.

The six original tables (`events`, `issues`, `issue_observations`,
`vehicles`, `segments`, `runs`) are untouched: no column added, removed or
retyped. The platform tables (`users`, `sessions`, `cases`, `case_issues`,
`case_activity`, `notifications`, `notification_reads`, `evidence_links`,
`fleet`) are created with `CREATE TABLE IF NOT EXISTS` when the backend
starts, so an existing database keeps working.

The one field the platform asks the pipeline for is `metadata.run_id`,
stamped by the orchestrator so an event can be tied to the annotated
recording it came from. Events written before that (the shipped run) are
matched to a recording by physical consistency and a crop comparison
instead, and left unlinked rather than guessed when nothing scores well
enough.

Field-by-field tables are in [PLATFORM.md](PLATFORM.md) section 9.
