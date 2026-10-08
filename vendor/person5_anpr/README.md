# ANPR Module (Person 5 — Number Plate Recognition)

Event-triggered license plate detection + OCR module for the bus
urban-sensing project. This module is **not** meant to run
continuously on the Nano — it is called once per flagged incident
(e.g. a hit-and-run track_id from Person 4's incident detector).

## Quickest way to test (no code editing)

1. Put your trained plate-detector weights (`best.pt` or similar) into `models/`
2. Put your test video(s) into `videos/`
3. `pip install -r requirements.txt`
4. `python run.py`
5. Results land in `outputs/`: an annotated video per input, plus a
   `<name>_results.txt` log of every plate reading and a summary of
   distinct strings seen.

This is for quick testing/demoing on a full video. For the real
integration with Person 4's incident detector, see below.

## Pipeline

```
Incident detected (Person 4)
    -> Vehicle Track ID + buffered vehicle crops
    -> [THIS MODULE]
    -> Plate detector (YOLO)
    -> Plate crop
    -> Image enhancement (multi-variant: color / CLAHE / Otsu)
    -> OCR (EasyOCR)
    -> Multi-frame voting
    -> Plate-format validation + character correction
    -> Confidence score
    -> Registration number + confidence
```

## Integration point

The only function the rest of the project needs to call:

```python
from anpr_module import ANPRPipeline

pipeline = ANPRPipeline(plate_model_path="path/to/best.pt", gpu=True)

result = pipeline.process_incident(
    track_id=42,
    vehicle_frames=vehicle_crop_list   # from Person 4's tracker buffer
)
# result = {
#     "plate": "TN12AB1234",
#     "confidence": 0.94,
#     "valid": True,
#     "track_id": 42,
#     "frames_used": 4,
#     "votes": {...},
#     "raw_candidates": [...]
# }
```

**Call this only when Person 4's incident logic flags a track_id** —
never per-frame during normal driving. That's what keeps ANPR off the
Nano's continuous compute budget.

## Files

- `anpr_pipeline.py` — the module itself (`PlateDetector`, `PlateOCR`,
  `vote_best_plate`, `correct_and_validate`, `ANPRPipeline`)
- `tests/run_full_video_anpr.py` — runs detection+OCR over an entire
  video file frame-by-frame (no incident simulation), prints every
  reading + a summary. Useful for evaluating a whole clip / raw
  detector+OCR quality.
- `tests/test_anpr_video.py` — simulates an incident by buffering a
  short time window from a video and calling `process_incident()`
  directly, closer to how Person 4/6 will actually invoke this module.

## Setup

```bash
pip install -r requirements.txt
```

Update `PLATE_MODEL_PATH` in the test scripts (and wherever
`ANPRPipeline(...)` is constructed) to point at the trained YOLO
weights (`best.pt`).

## ⚠️ Known open issue (as of last test run)

Running `tests/run_full_video_anpr.py` against the current test video
produced no stable plate reading — OCR output was highly fragmented
(dozens of distinct 1-6 character strings per run, mostly low
confidence, no single string dominating). A recurring `L6D...`-style
cluster suggests the detector IS finding roughly the right region, but
the crop is too low-resolution/blurry for character-level OCR once
upscaled.

**Before debugging further, test against footage where the vehicle
and plate are closer to / more square-on to the camera.** If the same
fragmentation happens on cleaner footage, the next things to check
are: whether the detector's bounding box is tight/accurate (visualize
a few crops), and whether the training dataset's image
resolution/distance matches your actual camera's real-world
distance-from-plate.

Do not assume a full retrain is needed yet — the detector (dataset)
and the OCR (pretrained EasyOCR, no dataset of its own) are separate
failure points, and this symptom pattern points more toward
crop resolution than either.
