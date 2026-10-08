import argparse
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    from roboflow import Roboflow

    cfg = yaml.safe_load(Path(args.config).read_text())
    rf_cfg = cfg["roboflow"]

    api_key = input("Enter your Roboflow API key (input is hidden by your terminal if supported): ").strip()
    if not api_key:
        raise SystemExit("Roboflow API key is required.")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(rf_cfg["workspace"]).project(rf_cfg["project"])
    dataset = project.version(int(rf_cfg["version"])).download(rf_cfg["format"])

    print(f"Dataset downloaded to: {dataset.location}")

if __name__ == "__main__":
    main()
