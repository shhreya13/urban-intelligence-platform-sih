#!/usr/bin/env python3
"""
Pre-flight check -- run this first on any new machine.

    python scripts/check_setup.py

It answers one question per line: "will this module actually run here?"
Nothing is loaded twice and no video is processed, so it finishes in a
few seconds.  Exit code 0 means the full pipeline can run; 1 means at
least one REQUIRED piece is missing (optional pieces only warn).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import load_config          # noqa: E402

OK = "  OK  "
WARN = " WARN "
FAIL = " FAIL "

_failures = 0
_warnings = 0


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<34} {detail}")


def need(label: str, ok: bool, detail: str = "", optional: bool = False) -> bool:
    global _failures, _warnings
    if ok:
        line(OK, label, detail)
    elif optional:
        _warnings += 1
        line(WARN, label, detail)
    else:
        _failures += 1
        line(FAIL, label, detail)
    return ok


def check_package(name: str, import_as: str | None = None, optional: bool = False) -> None:
    try:
        mod = importlib.import_module(import_as or name)
        version = getattr(mod, "__version__", "")
        need(f"python package: {name}", True, str(version))
    except Exception as exc:                                     # noqa: BLE001
        need(f"python package: {name}", False, f"{type(exc).__name__}: {exc}", optional)


def check_file(label: str, path: Path | None, optional: bool = False) -> None:
    if path is None:
        need(label, False, "not configured", optional)
        return
    exists = path.exists()
    size = f"{path.stat().st_size / 1e6:.1f} MB" if exists else f"missing: {path}"
    need(label, exists, size, optional)


def main() -> int:
    print("=" * 78)
    print("AI URBAN INTELLIGENCE PLATFORM -- setup check")
    print("=" * 78)

    print("\n-- dependencies ------------------------------------------------")
    for pkg in ("cv2", "numpy", "yaml", "torch", "ultralytics"):
        check_package(pkg)
    check_package("easyocr", optional=True)       # ANPR only
    check_package("fastapi", optional=True)       # backend only
    check_package("uvicorn", optional=True)
    check_package("lap", optional=True)           # BoT-SORT assignment solver

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detail = torch.cuda.get_device_name(0) if device == "cuda" else "CPU only (fine for this phase)"
        line(OK, "torch device", f"{device} -- {detail}")
    except Exception:                                            # noqa: BLE001
        pass

    config = load_config()

    print("\n-- model weights -----------------------------------------------")
    check_file("Person 2  vehicle model",
               config.path("perception.vehicle.model"))
    check_file("Person 1  ROAD_DEFECT_MODEL",
               config.path("perception.road_defect.model"))
    check_file("Person 5  plate model",
               config.path("modules.anpr.model"), optional=True)

    print("\n-- team source (vendor/, untouched originals) -------------------")
    for label, rel in (
        ("Person 1  road defect repo", "person1_road_defect"),
        ("Person 2  traffic repo", "person2_traffic"),
        ("Person 4  incident repo", "person4_incident"),
        ("Person 5  anpr repo", "person5_anpr"),
    ):
        need(label, (ROOT / "vendor" / rel).is_dir(), f"vendor/{rel}")

    print("\n-- video input --------------------------------------------------")
    source = config.get("video.source")
    if str(source).isdigit():
        line(OK, "video source", f"webcam index {source}")
    else:
        check_file("video source", config.resolve(source), optional=True)
    videos = sorted((ROOT / "data" / "videos").glob("*.mp4"))
    need("clips in data/videos", bool(videos),
         ", ".join(v.name for v in videos) or "drop an .mp4 in data/videos/",
         optional=True)

    print("\n-- writable paths -----------------------------------------------")
    config.ensure_dirs()
    for label, key in (
        ("database", "backend.database"),
        ("evidence dir", "backend.evidence_dir"),
        ("events jsonl", "output.events_jsonl"),
    ):
        path = config.path(key)
        parent = path.parent if path else None
        need(label, bool(parent and parent.is_dir()), str(path))

    print("\n" + "=" * 78)
    if _failures:
        print(f"{_failures} required check(s) failed -- fix those before running the pipeline.")
        print("Usually:   pip install -r requirements.txt")
        return 1
    if _warnings:
        print(f"Ready.  {_warnings} optional item(s) missing -- those modules will be")
        print("skipped or fall back automatically, the pipeline still runs.")
    else:
        print("Ready.  Everything the full pipeline needs is present.")
    print("Next:   python run_pipeline.py --source data/videos/<clip>.mp4")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
