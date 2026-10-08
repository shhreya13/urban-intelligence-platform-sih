"""
Minimal inference wrapper for the road defect detection model.
Usage:
    python predict.py path/to/image_or_video_or_folder
    python predict.py 0                # webcam
"""

import sys
from ultralytics import YOLO

WEIGHTS_PATH = "weights/best.pt"


def load_model(weights_path: str = WEIGHTS_PATH) -> YOLO:
    return YOLO(weights_path)


def predict(source, weights_path: str = WEIGHTS_PATH, conf: float = 0.25):
    model = load_model(weights_path)
    results = model.predict(source=source, conf=conf, save=True)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image/video/folder path or webcam index>")
        sys.exit(1)

    src = sys.argv[1]
    results = predict(src)

    for r in results:
        print(f"\n{r.path}")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  class={r.names[cls_id]}  conf={conf:.2f}  xyxy={box.xyxy[0].tolist()}")
