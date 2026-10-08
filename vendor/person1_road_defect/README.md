# Road Defect Detection Model

YOLO (Ultralytics) object detection model fine-tuned to detect road defects
(potholes / cracks) from images and video frames.

- Base model: `best_pothole_crack.pt` (fine-tuned further, see `training/args.yaml`)
- Input size: 640x640
- Epochs: 50
- Weights: `weights/best.pt` (best validation checkpoint)

> Note: exact class names come from the training `data.yaml`, which wasn't
> included in this export — update this section once confirmed.

## Setup

```bash
git clone <repo-url>
cd road-defect-model
pip install -r requirements.txt
```

## Run inference

```bash
python inference/predict.py path/to/image.jpg
python inference/predict.py path/to/video.mp4
python inference/predict.py 0          # webcam
```

Or directly in code:

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model("image.jpg")
results[0].show()          # display
results[0].save()          # save annotated output
```

`results[0].boxes` gives you class ids, confidences, and bounding boxes —
see `inference/predict.py` for a parsing example.

## Repo layout

```
weights/
    best.pt          # trained model weights
inference/
    predict.py        # inference wrapper
training/
    args.yaml          # training config used to produce this model
    results.csv        # per-epoch metrics
```
