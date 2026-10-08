"""
test_full_pipeline.py
=====================

Person 4 - FULL chain test: COLLISION -> HIT-AND-RUN -> ANPR
(pipeline runner that also produces the final incident report).

Streams one video through YOLO + BoT-SORT, then feeds every frame into:

    1. MotionAnalyzer + RashDrivingDetector   (rash-driving incidents)
    2. CollisionDetector                      (collision incidents)
    3. HitAndRunDetector                      (hit-and-run incidents)
    4. ANPRPipeline                           (plate read on each confirmed
                                               hit-and-run vehicle)

Output: outputs/final_incident_report.json + evidence crops under
outputs/incident_evidence/.

Modes:
    --demo-hitandrun   inject ONE synthetic collision for the last visible
                       vehicle after N frames, exercising the whole
                       COLLISION -> HIT-AND-RUN -> ANPR chain end-to-end
                       on real footage. The synthetic nature is recorded in
                       the report (used for integration validation only).

Usage:
    python test_full_pipeline.py [--video <path>] [--max-frames N] [--quiet]
    python test_full_pipeline.py --demo-hitandrun
"""

import argparse
import datetime
import io
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from ai.vehicle import VehicleDetector, VehicleTracker          # noqa: E402
from collision_detection import CollisionDetector              # noqa: E402
from hit_and_run import HitAndRunDetector                       # noqa: E402
from motion_analysis import MotionAnalyzer                     # noqa: E402
from rash_driving import RashDrivingDetector                   # noqa: E402
from anpr_module.anpr_pipeline import ANPRPipeline             # noqa: E402

VIDEO_A = r"C:\Users\Lenovo\Downloads\WhatsApp Video 2026-09-08 at 2.17.47 PM.mp4"
VIDEO_B = r"C:\Users\Lenovo\Downloads\WhatsApp Video 2026-09-08 at 3.35.45 PM.mp4"
DEFAULT_VIDEO = VIDEO_A

MODEL_PATH = str(CURRENT_DIR / "yolov8n.pt")
OUTPUT_DIR = CURRENT_DIR / "outputs"
EVIDENCE_DIR = OUTPUT_DIR / "incident_evidence"

FRAME_BUF = 160          # how many recent frames to keep for ANPR crops
CROP_MAX = 10            # max candidate crops handed to ANPR per incident


def run_pipeline(video_path, max_frames=0, demo_hitandrun=False, quiet=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n" + "=" * 78)
    print("FULL INCIDENT CHAIN: COLLISION -> HIT-AND-RUN -> ANPR")
    print("=" * 78)
    print(f"Video    : {video_path}")
    print(f"Size     : {total} frames @ {fps:.2f} fps, {width}x{height}")

    detector = VehicleDetector(model_name=MODEL_PATH, confidence=0.4)
    tracker = VehicleTracker(detector)

    collision = CollisionDetector(fps=fps, debug=not quiet)
    hit_and_run = HitAndRunDetector(fps=fps, debug=not quiet)
    motion = MotionAnalyzer(fps=fps)
    rash = RashDrivingDetector()
    anpr = ANPRPipeline(debug=not quiet)

    rash_results = []
    collision_results = []
    hit_run_results = []
    anpr_results = []

    # Rolling history for crop collection
    frame_buf = deque(maxlen=FRAME_BUF)
    obj_buf = deque(maxlen=FRAME_BUF)

    # Per-track cooldown for duplicated rash flags
    rash_last = {}

    # Demo target selection: pick the car that is actually driving
    car_trails = {}
    body = None
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        if max_frames and frame_number > max_frames:
            break

        tracked_objects = tracker.process_frame(frame)

        # --- collision + hit-and-run chain ------------------------
        collision_events = collision.update(
            tracked_objects, frame_size=(width, height)
        )
        hit_events = hit_and_run.update(tracked_objects, collision_events)
        collision_results.extend(collision_events)
        hit_run_results.extend(hit_events)

        for ev in collision_events:
            print(_fmt_collision(ev))

        for ev in hit_events:
            print(_fmt_hit_and_run(ev))
            frames_for_anpr = _collect_vehicle_frames(
                frame_buf, obj_buf, ev["hit_vehicle_track_id"],
                ev.get("collision_frame_id"), ev["frame_id"],
            )
            anpr_result = anpr.process_incident(
                ev["hit_vehicle_track_id"],
                frames_for_anpr,
                evidence_dir=str(EVIDENCE_DIR),
            )
            anpr_results.append(anpr_result)
            anpr_result["incident_frame_id"] = ev["frame_id"]
            anpr_result["collision_track_id"] = ev.get("other_track_id")
            print(f"    -> ANPR: {anpr_result['plate_crop_path']} "
                  f"plate={anpr_result['license_plate']} "
                  f"method={anpr_result['method']}")

        # --- rash driving -----------------------------------------
        motion_data = motion.update(tracked_objects)
        for inc in rash.analyze(motion_data):
            tid = inc["vehicle_track_id"]
            if frame_number - rash_last.get(tid, -10 ** 9) < 90:
                continue
            rash_last[tid] = frame_number
            inc["frame_id"] = frame_number
            inc["timestamp_seconds"] = round(frame_number / fps, 3)
            rash_results.append(inc)

        # --- demo hit-and-run (synthetic, integration only) -------
        for o in tracked_objects:
            if o.object_type != "car":
                continue
            cx = (o.bbox[0] + o.bbox[2]) / 2.0
            cy = (o.bbox[1] + o.bbox[3]) / 2.0
            trail = car_trails.setdefault(int(o.track_id), deque(maxlen=20))
            trail.append((cx, cy))

        if demo_hitandrun and body is None and frame_number >= 40:
            best_id = None
            best_disp = 0.0
            for tid, trail in car_trails.items():
                if len(trail) < 5:
                    continue
                p0 = trail[0]
                p1 = trail[-1]
                disp = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
                if disp > best_disp:
                    best_disp = disp
                    best_id = tid
            if best_id is not None and best_disp > 20.0:
                body_cand = next(
                    (o for o in tracked_objects if int(o.track_id) == best_id),
                    None,
                )
                if body_cand is not None:
                    body = body_cand
                    other = next(
                        (o for o in tracked_objects if int(o.track_id) != best_id
                         and o.object_type == "car"),
                        None,
                    )
                    fake_coll = {
                        "incident_type": "collision",
                        "vehicle_track_id": int(body.track_id),
                        "vehicle_type": "car",
                        "other_track_id": int(other.track_id) if other else 0,
                        "other_type": "car",
                        "frame_id": frame_number,
                        "timestamp": getattr(body, "timestamp", None),
                        "timestamp_seconds": round(frame_number / fps, 3),
                        "iou": 0.3,
                        "distance": 20.0,
                        "approach": 0.9,
                        "motion_change": 0.8,
                        "containment": False,
                        "confidence": 0.7,
                        "synthetic_demo": True,
                    }
                    collision_results.append(fake_coll)
                    hit_and_run.update(tracked_objects, [fake_coll])
                    print(
                        f"  [DEMO] injected synthetic collision for moving car "
                        f"track {int(body.track_id)} at frame {frame_number}"
                    )

# --- keep rolling history for ANPR crops -------------------
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_buf.append((frame_number, buf.tobytes() if ok else None))
        obj_buf.append((
            frame_number,
            [(int(o.track_id), tuple(int(v) for v in o.bbox))
             for o in tracked_objects],
        ))

    cap.release()

    anpr.available()

    report = build_report(
        video_path=video_path,
        total_frames=total,
        fps=fps,
        width=width,
        height=height,
        frames_processed=frame_number,
        rash=rash_results,
        collisions=collision_results,
        hit_and_run=hit_run_results,
        anpr=anpr_results,
        demo_hitandrun=demo_hitandrun,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "final_incident_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    print("-" * 78)
    print(f"Rash-driving incidents : {len(rash_results)}")
    print(f"Collision incidents    : {len(collision_results)}")
    print(f"Hit-and-run incidents  : {len(hit_run_results)}")
    print(f"ANPR results           : {len(anpr_results)}")
    print(f"Report written to      : {report_path}")
    print("-" * 78)


def _fmt_collision(ev) -> str:
    return (
        f"  [COLLISION] frame {ev['frame_id']}: vehicle "
        f"{ev['vehicle_track_id']} ({ev['vehicle_type']}) <-> "
        f"{ev['other_track_id']} ({ev['other_type']}) "
        f"conf={ev['confidence']}"
    )


def _fmt_hit_and_run(ev) -> str:
    return (
        f"  [HIT-AND-RUN] track {ev['hit_vehicle_track_id']} "
        f"({ev['vehicle_type']}) escaped after collision at frame "
        f"{ev['collision_frame_id']} | distance "
        f"{ev['escape_distance_px']}px @ {ev['escape_speed_px_s']}px/s "
        f"({ev['escape_direction']}) conf={ev['confidence']}"
    )


def _collect_vehicle_frames(
    frame_buf, obj_buf, track_id, start_frame, end_frame,
) -> list:
    """Retrieve up to CROP_MAX cropped images of `track_id` visible between
    the collision and escape frames."""
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = 10 ** 9

    img_by_fid = {fid: data for fid, data in frame_buf}
    objs_by_fid = {fid: objs for fid, objs in obj_buf}

    frames = []
    for fid in sorted(set(img_by_fid) & set(objs_by_fid)):
        if not (start_frame <= fid <= end_frame):
            continue
        bbox = None
        for tid, bx in objs_by_fid[fid]:
            if tid == track_id:
                bbox = bx
                break
        data = img_by_fid[fid]
        if bbox is None or data is None:
            continue
        arr = np.frombuffer(bytes(data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        frames.append({
            "frame_id": fid,
            "timestamp": None,
            "bbox": bbox,
            "image": img,
        })
        if len(frames) >= CROP_MAX:
            break
    return frames


def build_report(video_path, total_frames, fps, width, height,
                 frames_processed, rash, collisions, hit_and_run,
                 anpr, demo_hitandrun):
    return {
        "pipeline": "incident_detection (Person 4): "
                    "collision -> hit-and-run -> anpr",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "video": {
            "path": video_path,
            "frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "frames_processed": frames_processed,
        },
        "notes": {
            "demo_hitandrun": demo_hitandrun,
            "demo_warning": (
                "A synthetic collision was injected for integration "
                "validation only; do not treat its HIT-AND-RUN/ANPR entries "
                "as real incidents."
                if demo_hitandrun else None
            ),
        },
        "rash_driving_incidents": rash,
        "collision_incidents": collisions,
        "hit_and_run_incidents": hit_and_run,
        "anpr_results": anpr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full COLLISION->HIT-AND-RUN->ANPR chain test"
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--demo-hitandrun", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        max_frames=args.max_frames,
        demo_hitandrun=args.demo_hitandrun,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()