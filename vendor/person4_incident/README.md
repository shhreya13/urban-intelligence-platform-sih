# Incident Detection + Tracking (Person 4)

COLLISION → HIT-AND-RUN → ANPR chain for the **Urban Intelligence Platform**.

This is a self-contained module. It vendors its own copy of the shared
`ai/vehicle` detector + BoT-SORT tracker (YOLOv8n) so it can run standalone
on this machine.

```
incident_detection/
├── ai/vehicle/                 # vendored YOLO + BoT-SORT tracker
├── anpr_module/
│   ├── anpr_pipeline.py        # ANPR plugin (plate reading on demand)
│   └── models/                 # drop "best (1).pt" plate model here (optional)
├── collision_detection.py      # CollisionDetector
├── hit_and_run.py              # HitAndRunDetector
├── motion_analysis.py          # MotionAnalyzer (speed/accel/lateral)
├── rash_driving.py             # RashDrivingDetector
├── test_collision.py           # collision-only diagnostic (real video)
├── test_collision_synthetic.py # deterministic collision unit tests
├── test_hit_and_run_synthetic.py
├── test_full_pipeline.py       # full chain runner + final report
└── yolov8n.pt
```

## Quick start

```bash
# Synthetic unit tests (fast, no video)
python test_collision_synthetic.py
python test_hit_and_run_synthetic.py

# Collision-only scan of a real video
python test_collision.py --video "<path>" --quiet

# Full chain: COLLISION -> HIT-AND-RUN -> ANPR -> report
python test_full_pipeline.py --video "<path>" --quiet
python test_full_pipeline.py --video "<path>" --demo-hitandrun   # integration demo
```

Outputs:

- `outputs/final_incident_report.json` — all incidents in the platform's
  report schema (`rash_driving_incidents`, `collision_incidents`,
  `hit_and_run_incidents`, `anpr_results`).
- `outputs/incident_evidence/` — cropped vehicle images handed to ANPR.

## Default thresholds (tunable in each class constructor)

| Component | Key defaults |
|---|---|
| Collision | `confirm_peak=0.62`, `confirm_sustained=0.50`, `min_persist_frames=2`, `confirm_min_iou=0.08`, `max_episode_frames=30`, `separated_threshold=0.06`, `max_vehicle_fraction=0.42`, `approach_min=0.35`, `motion_min=0.30` |
| Hit-and-run | `min_escape_frames=12`, `min_escape_distance_px=70`, `escape_ratio=1.5`, `escape_progress=0.5`, `require_consecutive=3`, `max_missing_frames=30` |
| ANPR | no thresholds (see Findings report for optional components) |

## Detection philosophy

- **Collision** = real box overlap that is **transient** (bounded episode) and
  **preceded by a separated state** (the boxes came together). Co-moving
  traffic, side-by-side creep, box jitter, near-field occluders and blinking
  track IDs are all rejected (see the findings report and the synthetic unit
  tests).
- **Hit-and-run** = a confirmed collision after which the vehicle **moves
  away from the incident point** by > 1.5× its own size over consecutive
  frames. Track **disappearance alone is never** an escape.
- **ANPR** = plugin called **once per confirmed hit-and-run**; it never
  invents a plate — without OCR it returns `license_plate: null` and only
  saves the evidence crop.

## Environment notes

- Python 3.11, `ultralytics`, OpenCV, NumPy. No GPU required.
- `easyocr` and the plate-detection model `best (1).pt` are **optional**; the
  module works (honestly, without text) when they are absent.