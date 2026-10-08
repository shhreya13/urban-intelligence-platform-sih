"""
Normalize a downloaded Roboflow YOLO dataset to two classes:

0 = person
1 = child

Roboflow's Human Dataset v2 contains several human-related labels.
We map common human labels to `person`, `child` stays `child`, and
unrelated classes are ignored.
"""

from pathlib import Path
import shutil
import yaml

PERSON_NAMES = {
    "person", "Person", "human", "Human", "pedestrian", "Pedestrian",
    "people", "People", "persons", "aperson", "Pessoa"
}
CHILD_NAMES = {"child", "Child"}

def load_names(data_yaml):
    data = yaml.safe_load(Path(data_yaml).read_text())
    names = data["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)

def normalize_split(root, split, old_names):
    label_dir = root / split / "labels"
    image_dir = root / split / "images"
    if not label_dir.exists():
        return

    removed = 0
    kept = 0

    for label_file in label_dir.glob("*.txt"):
        new_lines = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            if old_id < 0 or old_id >= len(old_names):
                continue
            name = old_names[old_id]
            if name in CHILD_NAMES:
                new_id = 1
            elif name in PERSON_NAMES:
                new_id = 0
            else:
                continue
            new_lines.append(" ".join([str(new_id)] + parts[1:]))

        if new_lines:
            label_file.write_text("\n".join(new_lines) + "\n")
            kept += 1
        else:
            # Keep images without target objects out of the object detector training set.
            img_candidates = list(image_dir.glob(label_file.stem + ".*"))
            for img in img_candidates:
                img.unlink(missing_ok=True)
            label_file.unlink(missing_ok=True)
            removed += 1

    print(f"{split}: kept {kept}, removed empty {removed}")

def main():
    candidates = list(Path(".").glob("*/data.yaml"))
    if not candidates:
        raise SystemExit("No downloaded Roboflow dataset found. Run download_roboflow.py first.")

    data_yaml = candidates[0]
    root = data_yaml.parent
    old_names = load_names(data_yaml)

    for split in ("train", "valid", "test"):
        normalize_split(root, split, old_names)

    new_yaml = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "person", 1: "child"},
    }
    (root / "person_child.yaml").write_text(yaml.safe_dump(new_yaml, sort_keys=False))
    print(f"Wrote {root / 'person_child.yaml'}")

if __name__ == "__main__":
    main()
