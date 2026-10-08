import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--half", action="store_true")
    args = p.parse_args()

    model = YOLO(args.model)
    exported = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        device=0,
        workspace=2,
    )
    print(f"TensorRT engine: {exported}")

if __name__ == "__main__":
    main()
