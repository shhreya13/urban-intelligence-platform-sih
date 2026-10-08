"""
run.py

ANPR pipeline that:
  1. Tracks each vehicle's plate across frames (YOLO + ByteTrack), so
     readings from different cars are never mixed together.
  2. ALWAYS gives you a best guess per vehicle, even if no reading came
     out to exactly 10 characters. A 10-char, standard-format reading
     is preferred when ranking, but nothing is thrown away outright.
  3. Writes an annotated output video with a box + best-guess label
     drawn over every detected vehicle, so you can eyeball accuracy.

Outputs (in outputs/):
  {video_name}_plates.txt      -- per-vehicle top-3 guesses, text
  {video_name}_annotated.mp4   -- video with boxes + labels drawn in
"""

import os
import glob
import re
import cv2
from ultralytics import YOLO
import easyocr
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

FRAME_SKIP = 5
# How many frames (after skipping) to run detection on. Raise this (or set
# to None for "no limit") if you want the whole video analyzed instead of
# just the first chunk -- it'll just take longer on CPU.
MAX_FRAMES_TO_PROCESS = 300
MIN_CROP_HEIGHT = 38
TOP_N = 3
PLATE_LENGTH = 10  # standard Indian plate length -- used as a scoring preference, not a hard filter

PLATE_REGEX_STANDARD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")


def find_model_path():
    pt_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt model file found in {MODELS_DIR}/")
    return pt_files[0]


def find_videos():
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(os.path.join(VIDEOS_DIR, f"*{ext}")))
    if not videos:
        raise FileNotFoundError(f"No video files found in {VIDEOS_DIR}/")
    return videos


def enhance_variants(crop):
    up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [up, gray_clahe, otsu]


def read_plate(reader, crop):
    best_text, best_conf = "", 0.0
    for variant in enhance_variants(crop):
        results = reader.readtext(
            variant, detail=1, paragraph=False, allowlist=ALLOWED_CHARS
        )
        if not results:
            continue
        results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
        text = "".join(re.sub(r'[^A-Z0-9]', '', t[1].upper()) for t in results)
        avg_conf = sum(t[2] for t in results) / len(results)
        if text and avg_conf > best_conf:
            best_text, best_conf = text, avg_conf
    return best_text, best_conf


def score(item):
    """Higher is better. Prefers standard-format, 10-char, high-confidence
    readings, but any non-empty reading still gets scored (never rejected
    outright)."""
    text, data = item
    s = data["best_conf"]
    if data["valid"]:
        s += 0.5
    if len(text) == PLATE_LENGTH:
        s += 0.2
    return s


def process_video(model, reader, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open {video_path}, skipping.")
        return

    frame_num = 0
    processed_count = 0
    skipped_small = 0
    skipped_no_track = 0

    # track_id -> { text -> {"count", "best_conf", "valid", "best_frame", "crop_h"} }
    tracks = defaultdict(
        lambda: defaultdict(
            lambda: {"count": 0, "best_conf": 0.0, "valid": False, "best_frame": None, "crop_h": 0}
        )
    )
    track_order = []  # first-seen order, used for "Vehicle 1, 2, 3..." numbering

    # frame_num -> list of (track_id, x1, y1, x2, y2), for drawing the output video later
    frame_boxes = defaultdict(list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % FRAME_SKIP != 0:
            continue

        processed_count += 1
        if MAX_FRAMES_TO_PROCESS and processed_count > MAX_FRAMES_TO_PROCESS:
            break

        # model.track (instead of model.predict) assigns a persistent track ID
        # to each detection across frames via YOLO's built-in ByteTrack, which
        # is what lets us tell different vehicles apart.
        results = model.track(
            frame, conf=0.35, imgsz=640, verbose=False, device='cpu',
            persist=True, tracker="bytetrack.yaml"
        )

        for result in results:
            if result.boxes is None:
                continue
            box_ids = result.boxes.id
            for i, box in enumerate(result.boxes):
                if box_ids is None:
                    skipped_no_track += 1
                    continue
                track_id = int(box_ids[i].item())

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Register this vehicle and remember its box for the output video,
                # even if the crop turns out too small/unreadable below.
                if track_id not in tracks:
                    track_order.append(track_id)
                frame_boxes[frame_num].append((track_id, x1, y1, x2, y2))

                plate = frame[y1:y2, x1:x2]
                if plate.size == 0:
                    continue

                crop_h = plate.shape[0]
                if crop_h < MIN_CROP_HEIGHT:
                    skipped_small += 1
                    continue

                text, conf = read_plate(reader, plate)
                if not text:
                    continue

                is_valid = bool(PLATE_REGEX_STANDARD.match(text))

                entry = tracks[track_id][text]
                entry["count"] += 1
                if conf > entry["best_conf"]:
                    entry["best_conf"] = conf
                    entry["best_frame"] = frame_num
                    entry["crop_h"] = crop_h
                entry["valid"] = entry["valid"] or is_valid

    cap.release()

    print(f"\nDetected {len(track_order)} vehicle(s) in this video.")
    print(f"(Skipped {skipped_small} detections under {MIN_CROP_HEIGHT}px tall.)")
    print(f"(Skipped {skipped_no_track} detections the tracker couldn't link to a vehicle yet.)")

    if not track_order:
        print("No vehicles detected.")
        return

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]

    # ---- Build per-vehicle rankings and a display label for the video ----
    vehicle_labels = {}   # track_id -> label string drawn on the output video
    vehicle_ranked = {}   # track_id -> ranked list of (text, data), top TOP_N

    for car_num, track_id in enumerate(track_order, 1):
        candidates = tracks.get(track_id, {})
        if candidates:
            ranked = sorted(candidates.items(), key=score, reverse=True)[:TOP_N]
            best_text, best_data = ranked[0]
            note = "" if len(best_text) == PLATE_LENGTH and best_data["valid"] else " (uncertain)"
            vehicle_labels[track_id] = f"#{car_num}: {best_text}{note}"
        else:
            ranked = []
            vehicle_labels[track_id] = f"#{car_num}: unreadable"
        vehicle_ranked[track_id] = ranked

    # ---- Console + text report ----
    out_txt_path = os.path.join(OUTPUTS_DIR, f"{name}_plates.txt")
    with open(out_txt_path, "w") as f:
        for car_num, track_id in enumerate(track_order, 1):
            ranked = vehicle_ranked[track_id]
            total_seen = sum(d["count"] for _, d in tracks.get(track_id, {}).items())

            header = f"\n=== VEHICLE {car_num} (track #{track_id}, {total_seen} readings) ==="
            print(header)
            f.write(header + "\n")

            if not ranked:
                line = "  No readable plate text for this vehicle."
                print(line)
                f.write(line + "\n")
                continue

            for i, (text, data) in enumerate(ranked, 1):
                standard = len(text) == PLATE_LENGTH and data["valid"]
                line = (f"  {i}. '{text}'  conf={data['best_conf']*100:.0f}%  "
                        f"standard_format={standard}  seen={data['count']}x  "
                        f"best_frame={data['best_frame']}  crop_height={data['crop_h']}px")
                print(line)
                f.write(line.replace("'", "") + "\n")

    # ---- Annotated output video ----
    cap2 = cv2.VideoCapture(video_path)
    fps = cap2.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video_path = os.path.join(OUTPUTS_DIR, f"{name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    frame_num2 = 0
    while cap2.isOpened():
        ret, frame = cap2.read()
        if not ret:
            break
        frame_num2 += 1

        for (track_id, x1, y1, x2, y2) in frame_boxes.get(frame_num2, []):
            label = vehicle_labels.get(track_id, f"track {track_id}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 3, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        writer.write(frame)

    cap2.release()
    writer.release()

    print(f"\nSaved report to: {out_txt_path}")
    print(f"Saved annotated video to: {out_video_path}")
    if MAX_FRAMES_TO_PROCESS and (processed_count > MAX_FRAMES_TO_PROCESS):
        print(f"Note: only the first ~{MAX_FRAMES_TO_PROCESS * FRAME_SKIP} frames were analyzed "
              f"(MAX_FRAMES_TO_PROCESS={MAX_FRAMES_TO_PROCESS}). Frames after that play back "
              f"in the output video with no boxes. Raise MAX_FRAMES_TO_PROCESS or set it to "
              f"None to analyze the whole video.")


def main():
    model_path = find_model_path()
    videos = find_videos()

    print(f"Model: {model_path}")
    print(f"Found {len(videos)} video(s) to process.\n")

    model = YOLO(model_path)
    reader = easyocr.Reader(['en'], gpu=False)

    for video_path in videos:
        print(f"Processing: {video_path}")
        process_video(model, reader, video_path)
        print()


if __name__ == "__main__":
    main()