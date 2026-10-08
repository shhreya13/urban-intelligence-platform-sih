"""
test_collision.py
=================

Person 4 - COLLISION-ONLY diagnostic test.

Intentionally does NOT run rash driving, hit-and-run, ANPR or the full
pipeline. It streams one video through YOLO + BoT-SORT and lets the
CollisionDetector evaluate every vehicle-pair in every frame.

Use it to tune thresholds before running the full chain.

Usage:
    python test_collision.py --video <path> [--max-frames N] [--quiet]
    python test_collision.py                        # uses default video
"""

import argparse
import sys
from pathlib import Path

import cv2

# ============================================================
# SELF-CONTAINED PATH SETUP
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from ai.vehicle import VehicleDetector, VehicleTracker          # noqa: E402
from collision_detection import CollisionDetector              # noqa: E402


# ============================================================
# DEFAULT VIDEOS AVAILABLE ON THIS MACHINE
# ============================================================

VIDEO_A = r"C:\Users\Lenovo\Downloads\WhatsApp Video 2026-09-08 at 2.17.47 PM.mp4"
VIDEO_B = r"C:\Users\Lenovo\Downloads\WhatsApp Video 2026-09-08 at 3.35.45 PM.mp4"
DEFAULT_VIDEO = VIDEO_A

MODEL_PATH = str(CURRENT_DIR / "yolov8n.pt")


# ============================================================
# RUN ONE VIDEO
# ============================================================

def run_video(video_path: str, max_frames: int = 0, debug: bool = True):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n" + "=" * 78)
    print("COLLISION-ONLY TEST")
    print("=" * 78)
    print(f"Video    : {video_path}")
    print(f"Size     : {total} frames @ {fps:.2f} fps, {width}x{height}")

    detector = VehicleDetector(model_name=MODEL_PATH, confidence=0.4)
    tracker = VehicleTracker(detector)

    collision = CollisionDetector(
        fps=fps,
        debug=debug,
    )

    frame_number = 0
    confirmed = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if max_frames and frame_number > max_frames:
            break

        tracked_objects = tracker.process_frame(frame)

        events = collision.update(tracked_objects, frame_size=(width, height))

        for event in events:
            confirmed.append(
                {"frame": frame_number, "event": event}
            )
            print("\n" + "=" * 78)
            print("[COLLISION CONFIRMED]")
            print(f"Frame: {event['frame_id']}")
            print(
                f"Vehicle Track ID: {event['vehicle_track_id']} "
                f"({event['vehicle_type']})"
            )
            print(
                f"Other Track ID  : {event['other_track_id']} "
                f"({event['other_type']})"
            )
            print(f"IoU: {event['iou']} | distance: {event['distance']}px")
            print(
                f"approach: {event['approach']} | "
                f"motion_change: {event['motion_change']}"
            )
            print(f"Confidence: {event['confidence']}")
            print("=" * 78 + "\n")

    cap.release()

    print("-" * 78)
    print(f"Frames processed : {frame_number}")
    print(f"Collisions       : {len(confirmed)}")
    print("-" * 78)

    if confirmed:
        for item in confirmed:
            e = item["event"]
            print(
                f"Confirmed @ frame {e['frame_id']}: "
                f"vehicle {e['vehicle_track_id']} ({e['vehicle_type']}) "
                f"<-> {e['other_track_id']} ({e['other_type']}) "
                f"conf={e['confidence']}"
            )

    print("\nTop potential-contact pairs (tuning reference):")
    for row in collision.summary(top_n=12):
        print(
            f"  pair={row['pair']} best_instant={row['best_instant']} "
            f"iou={row['best_iou']} geom={row['best_geom']} "
            f"approach={row['best_approach']} "
            f"motion={row['best_motion']} frames={row['frames']}"
        )
    print("-" * 78)

    return confirmed


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Collision-only diagnostic")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_video(
        video_path=args.video,
        max_frames=args.max_frames,
        debug=not args.quiet,
    )


if __name__ == "__main__":
    main()