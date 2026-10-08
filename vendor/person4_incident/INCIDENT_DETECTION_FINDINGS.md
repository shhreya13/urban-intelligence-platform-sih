# Findings & Limitations — Incident Detection (Person 4)

Chain: **COLLISION → HIT-AND-RUN → ANPR**

## Validation summary

| Test | Result |
|---|---|
| `test_collision_synthetic.py` (rear-end, hit-and-run scrape, congestion, near-miss, jitter) | 5/5 PASS |
| `test_hit_and_run_synthetic.py` (escape / stop / vanish) | 3/3 PASS |
| Real video A `WhatsApp Video 2026-09-08 at 2.17.47 PM.mp4` (767 fr, 59.98 fps, gridlock) | 0 collisions, 0 hit-and-run |
| Real video B `WhatsApp Video 2026-09-08 at 3.35.45 PM.mp4` (563 fr, 29.84 fps, through-traffic) | 0 collisions, 0 hit-and-run |
| `test_full_pipeline.py --demo-hitandrun` on video B | full chain exercised: collision → escape (157 px @ 88 px/s, conf 0.65) → ANPR called → evidence saved |

## What the detector was hardened against

During development, video A produced three apparent "collisions". All were
traced to **false positives**, and the detector was hardened until they
disappeared without breaking genuine cases:

1. **Near-field occluder** — a close-up box spanning 31% of the frame
   (camera-adjacent vehicle) overlapped everything. → Guard: boxes larger than
   `max_vehicle_fraction` of the frame are ignored as partners.
2. **Co-moving traffic** — two side-by-side cars images the same speed for 40+
   frames with constant overlap (a "collision" would need to separate).
   → Confirmation requires a **transient overlap episode** and a **separated
   (non-overlapping) state earlier in the window**.
3. **YOLO box jitter / blinking IDs** — a detection that popped into existence
   already overlapping another car, plus vertical box oscillation that faked a
   huge motion spike. → Motion is computed on the **bottom-centre** (feet)
   point with a 3-frame moving average, and uses a **scale-free relative decel**
   ratio (object's own recent speed) so it transfers across videos of very
   different sizes/framerates (video A cars crawl, video B cars are fast).

Near-misses (fast cars, zero overlap) are deliberately not collisions —
video B's highest-scoring pairs all had `iou=0.0` with high motion and were
rejected correctly.

## Why the real videos produce zero incidents

Both available clips are incident-free footage. Video A is a congested
junctions scene; video B is fast through-traffic with pedestrians. Neither
contains an actual collision, so the honest output is an empty
collision/hit-and-run/anpr section (identical to the platform's "nothing
happened" state). The chain's positive behaviour is proven deterministically
by the synthetic suites, and the `--demo-hitandrun` integration mode proves
the end-to-end wiring on real footage with a clearly-labelled synthetic
collision.

## Documented limitations

- **No ground-truth collision video on this machine** — the platform spec's
  1887-frame clip was not available here, so real-footage collision recall
  could not be measured; precision is bounded by the synthetic
  near-miss/congestion/jitter scenarios.
- **ANPR reads no text in this environment**: `easyocr` is not installed and
  the plate-detection model `anpr_module/models/best (1).pt` is absent. The
  plugin degrades honestly (`license_plate: null`, method `plate_crop_only`)
  and still saves the plate candidate crop. To enable text reading:
  1. place the plate model at `anpr_module/models/best (1).pt`,
  2. `pip install easyocr`, re-run the pipeline (no code changes needed).
- **GPS is null** — the chain has no GPS feed; every incident record carries
  `"gps": null` rather than a fabricated coordinate.
- **Track identity churn** — BoT-SORT IDs can change under occlusion/RE-ID in
  crowd scenes; the escape monitor follows the locked ID only and therefore
  may miss a hit-and-run if the offender's ID is lost mid-flight. Confirming
  escape on disappearance alone was intentionally NOT implemented to avoid
  false positives.
- **Constant-speed scrapes into much larger objects** (e.g. a car brushing a
  bus) produce low geometric overlap and no speed change, so they can stay
  below confirmation. High-impact collisions (hard stop / burst / strong
  approach) are the reliably confirmed class.

## Files produced

- `collision_detection.py`, `hit_and_run.py`, `anpr_module/anpr_pipeline.py`
- `test_collision_synthetic.py`, `test_hit_and_run_synthetic.py`,
  `test_collision.py`, `test_full_pipeline.py`
- `outputs/final_incident_report.json` (current copy: video B demo run)
- `outputs/incident_evidence/vehicle_3.jpg` (ANPR evidence crop)