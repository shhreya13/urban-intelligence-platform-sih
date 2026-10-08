from pathlib import Path
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_cfg = cfg["training"]

    if args.data:
        data_yaml = args.data
    else:
        matches = list(Path(".").glob("*/person_child.yaml"))
        if not matches:
            raise SystemExit("No person_child.yaml found. Run the Roboflow download + normalization steps first.")
        data_yaml = str(matches[0])

    model_name = args.model or train_cfg["base_model"]
    model = YOLO(model_name)

    model.train(
        data=data_yaml,
        epochs=int(train_cfg["epochs"]),
        imgsz=int(train_cfg["imgsz"]),
        batch=int(train_cfg["batch"]),
        project=train_cfg["project"],
        name=train_cfg["name"],
        pretrained=True,
        plots=True,
    )

if __name__ == "__main__":
    main()
