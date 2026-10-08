# Person 3 — Pedestrian Safety AI

A complete prototype pipeline for the SIH pedestrian-safety module:

**Roboflow dataset → YOLO training → BoT-SORT tracking → road/crossing zones → pedestrian/vehicle proximity → approach detection → risk score → JSON events → Jetson/TensorRT export**

## Scope

Continuously running:
- Pedestrian/child detector (`person`, `child`)
- Vehicle detector (reuse Person 2's model if available; otherwise use a COCO YOLO model for development)

Logic layer:
- Tracking
- Road-zone and crossing-zone membership
- Pedestrian ↔ vehicle distance in image space
- Closing-distance / approach detection
- Risk score: LOW / MEDIUM / HIGH / CRITICAL
- Event JSON output for Person 4

Event output example:

```json
{
  "event": "pedestrian_vehicle_risk",
  "pedestrian_id": "P17",
  "vehicle_id": "V42",
  "pedestrian_type": "child",
  "distance_px": 58.2,
  "approaching": true,
  "crossing": true,
  "risk_level": "HIGH",
  "risk_score": 78,
  "timestamp_s": 12.47
}
```

## 1. Install

Development PC:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Jetson:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Roboflow dataset

This project is designed around **Human Dataset v2** because it is large and includes a `child` class among human-related classes. The public project currently lists about 13.7k images and CC BY 4.0 licensing.

Create a Roboflow API key and run:

```bash
python scripts/download_roboflow.py
python scripts/normalize_dataset.py
```

The downloader defaults to:
- workspace: `human-v2`
- project: `human-dataset-v2`
- version: `6`
- format: YOLOv8

If you use another Roboflow dataset, change those values in `config.yaml`.

## 3. Train the pedestrian model

```bash
python train.py
```

The default is `yolov8n.pt` because the final target is an old Jetson Nano. You can change the base model later.

Expected custom classes:

```text
0 person
1 child
```

## 4. Run the safety pipeline on your supplied video

Example:

```bash
python run_safety.py ^
  --source "WhatsApp Video 2026-09-08 at 10.15.09 PM.mp4" ^
  --ped-model "runs/detect/person3_pedestrian/weights/best.pt" ^
  --vehicle-model "yolov8n.pt" ^
  --output "outputs/pedestrian_safety.mp4"
```

Linux:

```bash
python run_safety.py \
  --source "WhatsApp Video 2026-09-08 at 10.15.09 PM.mp4" \
  --ped-model "runs/detect/person3_pedestrian/weights/best.pt" \
  --vehicle-model "yolov8n.pt" \
  --output "outputs/pedestrian_safety.mp4"
```

**For the final SIH system, replace the temporary COCO vehicle model with Person 2's vehicle model.**

## 5. Zones

The sample `config.yaml` contains normalized polygons. They are only starting points for the supplied vertical video.

You should tune these polygons using a frame from your actual bus camera.

- `road_zone`: area considered road
- `crossing_zone`: zebra-crossing area

Coordinates are normalized from 0 to 1:

```text
x = pixel_x / image_width
y = pixel_y / image_height
```

## 6. Risk logic

The prototype intentionally does NOT train a separate "danger model".

Risk is computed from:

```text
distance
+ closing distance
+ relative approach
+ road/crossing position
+ vehicle speed proxy
+ child multiplier
```

This makes the system lightweight and explainable.

### Important limitation

A 2D bounding-box distance is **pixel distance**, not meters. Do not present it as a physical distance until you calibrate the camera/road plane.

For the SIH demo, use the risk engine comparatively. For a stronger version, add homography/camera calibration and estimate distance in meters.

## 7. Jetson Nano deployment

Development:

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT
   ↓
FP16
   ↓
Jetson Nano
```

Export after you have a trained model:

```bash
python export_tensorrt.py --model runs/detect/person3_pedestrian/weights/best.pt
```

Start with FP16. Benchmark before choosing INT8.

Also use:
- frame skipping
- smaller inference resolution (e.g. 416 or 512)
- one inference stream per model
- GPU video decode where available
- event throttling
- no ANPR unless an incident requires it

## 8. Integration contract for Person 4

Person 3 should expose:

```text
pedestrian_id
vehicle_id
pedestrian_type
distance_px
approaching
crossing
in_road
risk_score
risk_level
timestamp
```

Person 4 can consume these events to reason about:
- near miss
- pedestrian collision
- child/vehicle dangerous interaction
- incident evidence

## 9. Suggested team integration

Person 2 → vehicle detections/tracks/speed

Person 3 → pedestrian safety events

Person 4 → incident reasoning

Person 6 → Nano optimization and scheduling

Do not duplicate Person 2's vehicle model in the final deployment unless there is a specific accuracy/latency reason.
