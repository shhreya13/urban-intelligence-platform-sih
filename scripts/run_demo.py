#!/usr/bin/env python3
"""
The one-command demo.

    python scripts/run_demo.py
    python scripts/run_demo.py --source data/videos/my_clip.mp4
    python scripts/run_demo.py --multi-bus        # 3 buses over the same road

What it does, end to end, on a normal laptop:

    video file
      -> shared perception (vehicle + tracker, road defect)
      -> traffic / incident modules
      -> ANPR, only where an incident asked for it
      -> intelligence engine (dedup, persistence, priority, routing)
      -> SQLite + events.jsonl + annotated.mp4 + evidence images
      -> backend API on :8000
      -> dashboard opens in your browser

--multi-bus replays the clip as three different buses driving the same
route, which is what makes cross-bus deduplication and persistent issue
tracking visible: the same pothole seen by BUS-01/02/03 becomes ONE
issue POT-00x with observations=3, buses=3.
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import load_config                              # noqa: E402
from pipeline.orchestrator import Pipeline                       # noqa: E402


def banner(text: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {text}")
    print("=" * 74)


def pick_source(explicit: str | None) -> str:
    if explicit:
        return explicit
    videos = sorted((ROOT / "data" / "videos").glob("*.mp4"))
    if not videos:
        print("No video found.  Put a clip in data/videos/ or pass --source PATH.")
        raise SystemExit(1)
    return str(videos[0])


def run_one(source: str, bus_id: str, args, annotate: bool) -> dict:
    overrides = {
        "platform": {"bus_id": bus_id},
        "video": {
            "source": source,
            "save_annotated": annotate and not args.no_video,
            "annotated_output": f"outputs/runs/annotated_{bus_id}.mp4",
        },
    }
    if args.max_frames:
        overrides["video"]["max_frames"] = args.max_frames
    if args.resize_width:
        overrides["video"]["resize_width"] = args.resize_width
    if args.device:
        overrides["platform"]["device"] = args.device

    pipeline = Pipeline(config=load_config(overrides=overrides))
    return pipeline.run()


def main() -> int:
    parser = argparse.ArgumentParser(description="Urban Intelligence Platform demo")
    parser.add_argument("--source", default=None, help="video file (default: first clip in data/videos)")
    parser.add_argument("--multi-bus", action="store_true",
                        help="replay as BUS-01/02/03 to show cross-bus dedup + persistence")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=960,
                        help="downscale for CPU speed (0 = native resolution)")
    parser.add_argument("--device", default=None, help="cpu | 0 | cuda:0")
    parser.add_argument("--no-video", action="store_true", help="skip the annotated .mp4")
    parser.add_argument("--no-serve", action="store_true", help="process only, do not start the backend")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    source = pick_source(args.source)
    buses = ["BUS-01", "BUS-02", "BUS-03"] if args.multi_bus else ["BUS-01"]

    banner("STEP 1/3  --  AI pipeline")
    print(f"  video   : {source}")
    print(f"  buses   : {', '.join(buses)}")
    print("  modules : road_detection, traffic, incident, anpr, intelligence")

    summaries = []
    started = time.time()
    for index, bus_id in enumerate(buses):
        if len(buses) > 1:
            banner(f"  pass {index + 1}/{len(buses)}  --  {bus_id}")
        summaries.append(run_one(source, bus_id, args, annotate=(index == 0)))

    banner("STEP 2/3  --  what the run produced")
    total_events = sum(s.get("events_total", 0) for s in summaries)
    total_frames = sum(s.get("frames_processed", 0) for s in summaries)
    print(f"  frames processed : {total_frames}")
    print(f"  events emitted   : {total_events}")
    merged = {}
    for summary in summaries:
        for key, value in (summary.get("events_by_type") or {}).items():
            merged[key] = merged.get(key, 0) + value
    for key in sorted(merged):
        print(f"     {key:<26} {merged[key]}")
    print(f"  wall clock       : {time.time() - started:.1f}s")
    print("\n  artefacts:")
    for label, rel in (
        ("events   ", "outputs/runs/events.jsonl"),
        ("database ", "outputs/urban_intelligence.db"),
        ("evidence ", "outputs/evidence/"),
        ("video    ", f"outputs/runs/annotated_{buses[0]}.mp4"),
    ):
        print(f"     {label} {ROOT / rel}")

    if args.no_serve:
        print("\n  --no-serve: skipping the dashboard.  Start it later with:")
        print("     python run_backend.py")
        return 0

    config = load_config()
    host = config.get("backend.host", "127.0.0.1")
    port = args.port or int(config.get("backend.port", 8000))
    url = f"http://{host}:{port}/"

    banner("STEP 3/3  --  dashboard")
    print(f"  opening {url}   (Ctrl-C to stop)")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:                                        # noqa: BLE001
            pass

    try:
        import uvicorn
    except ImportError:
        print("  uvicorn is not installed:  pip install fastapi uvicorn")
        return 1
    uvicorn.run("backend.app:app", host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
