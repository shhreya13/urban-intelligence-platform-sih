#!/usr/bin/env python3
"""
Entry point: run the integrated pipeline over a video (or webcam).

    python run_pipeline.py
    python run_pipeline.py --source data/videos/bus_front.mp4 --bus-id BUS-02
    python run_pipeline.py --source 0 --display
    python run_pipeline.py --max-frames 300 --resize-width 960

Every flag simply overrides the matching key in config/platform.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import load_config                              # noqa: E402
from pipeline.orchestrator import Pipeline                       # noqa: E402


def build_overrides(args) -> dict:
    overrides: dict = {"platform": {}, "video": {}, "modules": {}, "backend": {}}

    if args.source is not None:
        overrides["video"]["source"] = args.source
    if args.max_frames is not None:
        overrides["video"]["max_frames"] = args.max_frames
    if args.start_frame is not None:
        overrides["video"]["start_frame"] = args.start_frame
    if args.resize_width is not None:
        overrides["video"]["resize_width"] = args.resize_width
    if args.display:
        overrides["video"]["display"] = True
    if args.no_video:
        overrides["video"]["save_annotated"] = False
    if args.output is not None:
        overrides["video"]["annotated_output"] = args.output

    if args.bus_id:
        overrides["platform"]["bus_id"] = args.bus_id
    if args.camera_id:
        overrides["platform"]["camera_id"] = args.camera_id
    if args.device:
        overrides["platform"]["device"] = args.device
    if args.log_level:
        overrides["platform"]["log_level"] = args.log_level

    if args.no_db:
        overrides["backend"]["enabled"] = False
    if args.database:
        overrides["backend"]["database"] = args.database

    for name in (args.disable or []):
        overrides["modules"].setdefault(name, {})["enabled"] = False
    for name in (args.only or []):
        pass  # handled below

    if args.only:
        every = [
            "road_detection", "traffic",
            "incident", "anpr", "intelligence",
        ]
        for name in every:
            overrides["modules"].setdefault(name, {})["enabled"] = name in args.only

    if args.demo_incident:
        overrides["modules"].setdefault("incident", {})["demo"] = {
            "inject_synthetic_collision": True,
            "at_frame": args.demo_at_frame or 0,
        }

    if args.gps:
        latitude, longitude = [float(v) for v in args.gps.split(",")]
        overrides["gps"] = {"mode": "fixed", "start": [latitude, longitude]}

    return {k: v for k, v in overrides.items() if v}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Urban Intelligence Platform -- integrated pipeline",
    )
    parser.add_argument("--config", default=None, help="path to platform.yaml")
    parser.add_argument("--source", default=None, help="video file, webcam index or stream URL")
    parser.add_argument("--bus-id", default=None)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--device", default=None, help="cpu | 0 | cuda:0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--output", default=None, help="annotated video path")
    parser.add_argument("--display", action="store_true", help="show a live window")
    parser.add_argument("--no-video", action="store_true", help="skip the annotated video")
    parser.add_argument("--no-db", action="store_true", help="JSONL only, no database")
    parser.add_argument("--database", default=None)
    parser.add_argument("--gps", default=None, help="fixed 'lat,lon' instead of the simulated route")
    parser.add_argument(
        "--disable", nargs="*", default=None,
        help="module names to switch off (road_detection traffic incident anpr intelligence)",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="run ONLY these modules",
    )
    parser.add_argument(
        "--demo-incident", action="store_true",
        help="inject ONE clearly-labelled synthetic collision so the "
             "collision -> hit-and-run -> ANPR chain can be demonstrated on "
             "footage with no real crash (marked synthetic everywhere)",
    )
    parser.add_argument("--demo-at-frame", type=int, default=None,
                        help="frame to inject the demo collision at")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    config = load_config(args.config, overrides=build_overrides(args))
    pipeline = Pipeline(config=config)
    summary = pipeline.run()
    return 0 if summary["frames_processed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
