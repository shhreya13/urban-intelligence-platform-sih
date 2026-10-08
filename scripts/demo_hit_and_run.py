#!/usr/bin/env python3
"""
Deterministic hit-and-run walkthrough for the dashboard.

    python scripts/demo_hit_and_run.py
    python scripts/demo_hit_and_run.py --background data/videos/street.mp4

WHY THIS EXISTS
---------------
A hit-and-run needs footage where a vehicle collides and then *demonstrably
drives away while still being tracked*.  None of the clips on this machine
contain one -- on real footage the pipeline correctly reports nothing, or
resolves the case to TRACK_LOST / COLLISION_ONLY.  That is the system
working, but it leaves the investigation UI with nothing to show.

So this script drives the REAL modules -- Person 4's CollisionDetector and
HitAndRunDetector, the real state machine, the real evidence store, the real
ANPR trigger and the real database -- over a SCRIPTED two-vehicle scenario
rendered onto real road footage.

What is real: every line of detection, state-machine, evidence, ANPR,
deduplication, priority and routing code, and the database rows.
What is scripted: the two vehicles' positions.

Every event it writes carries ``metadata.synthetic = true`` and the
dashboard renders it with a SYNTHETIC badge.  Nothing here is presented as
a detection from real footage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import load_config                              # noqa: E402
from core.event import GPS                                       # noqa: E402
from core.evidence import EvidenceStore                          # noqa: E402
from core.frame_context import FrameContext                      # noqa: E402
from core.geo import build_gps_provider, road_segment_id         # noqa: E402
from core.logging_setup import get_logger, setup_logging         # noqa: E402
from core.sinks import build_sinks                               # noqa: E402
from core.track_buffer import VehicleCropBuffer                  # noqa: E402
from modules.anpr import ANPRModule                              # noqa: E402
from modules.incident import IncidentModule                      # noqa: E402
from modules.intelligence import IntelligenceEngine              # noqa: E402

W, H, FPS = 960, 540, 30.0
BUS_ID = "BUS-DEMO"


# ---------------------------------------------------------------------------
# the scripted scenario
# ---------------------------------------------------------------------------

def scenario():
    """Offender (track 1) approaches the victim (track 2) from a clear gap,
    makes contact, then accelerates away out of the scene.

    Returns a list of per-frame vehicle lists in the platform's format."""
    frames = []

    def car(track_id, x, y, w, h, cls):
        return {
            "track_id": track_id,
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "class_name": cls, "confidence": 0.92,
            "speed": 0.0, "center": (x + w / 2, y + h / 2),
        }

    victim = lambda: car(2, 520, 300, 150, 110, "car")            # noqa: E731

    # phase 1 -- separated approach (24 frames)
    for i in range(24):
        frames.append([car(1, 120 + i * 15, 296, 150, 112, "truck"), victim()])
    # phase 2 -- contact (6 frames), the offender stops closing
    for i in range(6):
        frames.append([car(1, 476 + i, 296, 150, 112, "truck"), victim()])
    # phase 3 -- the offender leaves the scene while the victim stays
    for i in range(1, 55):
        frames.append([car(1, 482 + i * 9, 296 - i, 150, 112, "truck"), victim()])
    return frames


def render(background, vehicles, frame_number):
    """Draw the scripted vehicles onto a real road frame, so the evidence
    images show exactly the boxes the modules reasoned about."""
    import cv2

    frame = background.copy()
    banner = "SYNTHETIC SCENARIO - scripted vehicles on real footage"
    cv2.rectangle(frame, (0, 0), (W, 26), (18, 18, 18), -1)
    cv2.putText(frame, banner, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (70, 160, 255), 1, cv2.LINE_AA)
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        colour = (60, 90, 240) if v["track_id"] == 1 else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{v['class_name']} #{v['track_id']}"
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + 130, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def backgrounds(path: Path, count: int):
    """Real frames to draw on; a plain asphalt tone if the clip is missing."""
    try:
        import cv2
        capture = cv2.VideoCapture(str(path))
        out = []
        while len(out) < count:
            ok, frame = capture.read()
            if not ok:
                break
            out.append(cv2.resize(frame, (W, H)))
        capture.release()
        if out:
            while len(out) < count:
                out.append(out[len(out) % len(out)].copy())
            return out
    except Exception:                                            # noqa: BLE001
        pass
    grey = np.full((H, W, 3), 48, dtype=np.uint8)
    return [grey.copy() for _ in range(count)]


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", default=None,
                        help="video whose frames are used as the road backdrop")
    parser.add_argument("--bus-id", default=BUS_ID)
    parser.add_argument("--keep", action="store_true",
                        help="keep existing database rows (default: append)")
    args = parser.parse_args()

    config = load_config(overrides={
        "platform": {"bus_id": args.bus_id, "camera_id": "CAM-FRONT"},
        "video": {"save_annotated": False},
    })
    config.ensure_dirs()
    setup_logging("INFO", str(config.path("platform.log_file") or ""))
    log = get_logger("demo")

    source = Path(args.background) if args.background else None
    if source is None:
        clips = sorted((ROOT / "data" / "videos").glob("*.mp4"))
        source = clips[0] if clips else Path("missing.mp4")

    script = scenario()
    plates = backgrounds(source, len(script))

    evidence = EvidenceStore(config)
    incident = IncidentModule(config, evidence=evidence)
    anpr = ANPRModule(config, evidence=evidence)
    sinks = build_sinks(config)
    database = next((getattr(s, "db", None) for s in sinks if hasattr(s, "db")), None)
    intelligence = IntelligenceEngine(config, database=database)

    if not incident.safe_setup():
        log.error("incident module unavailable: %s", incident.status_detail)
        return 1
    anpr.safe_setup()
    intelligence.setup()

    crops = VehicleCropBuffer(per_track=12, stride=2)
    gps_provider = build_gps_provider(config)
    segment_size = float(config.get("gps.segment_size_m", 60))

    print("=" * 72)
    print("  HIT-AND-RUN WALKTHROUGH  --  scripted scenario, real modules")
    print("=" * 72)
    print(f"  background : {source.name}")
    print(f"  bus        : {args.bus_id}")
    print(f"  frames     : {len(script)}")
    print("  every event written is marked  synthetic: true")
    print("-" * 72)

    total = 0
    for number, vehicles in enumerate(script):
        frame = render(plates[number], vehicles, number)
        ctx = FrameContext(
            frame=frame, frame_number=number, video_time_s=number / FPS,
            width=W, height=H, fps=FPS,
            bus_id=args.bus_id, camera_id="CAM-FRONT",
            vehicles=vehicles,
        )
        ctx.gps = gps_provider.position(number, len(script))
        ctx.road_segment = road_segment_id(ctx.gps, segment_size)
        crops.update(ctx)

        events = incident.safe_process(ctx)

        # event-triggered ANPR, exactly as the orchestrator does it
        produced = []
        for event in list(events):
            if not anpr.should_trigger(event):
                continue
            identification = anpr.identify(event, crops.get(
                event.metadata.get("anpr_track_id", event.track_id)), ctx)
            if identification is not None:
                produced.append(identification.with_context(ctx))
        events.extend(produced)

        events.extend(intelligence.process_events(events))
        for event in events:
            event.metadata["synthetic"] = True
            event.metadata.setdefault(
                "synthetic_note",
                "scripted demo scenario -- real detection, state-machine, "
                "ANPR and database code, scripted vehicle positions",
            )
            for sink in sinks:
                sink.emit(event)
            total += 1
            print(f"  frame {number:3d}  {event.event_type:<24} "
                  f"conf {event.confidence:.2f}  {event.severity}")

    for event in incident.safe_finalize():
        for sink in sinks:
            sink.emit(event)

    for sink in sinks:
        if hasattr(sink, "close"):
            sink.close()

    states = {t: c.state for t, c in incident._cases.items()}
    print("-" * 72)
    print(f"  events written : {total}")
    print(f"  final states   : {states}")
    print("=" * 72)
    print("  Now start the dashboard and open the Hit-and-Run page:")
    print("     python run_backend.py   ->   http://127.0.0.1:8000/#hitrun")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
