"""
run_full_video_anpr.py

Runs plate detection + OCR on every frame of your ACTUAL video —
no simulated windows, no fake data. Prints every reading as it's
found, writes an annotated output video, and prints a summary of
distinct plate strings seen (rough stand-in for multi-frame voting
until real vehicle tracking is wired in).
"""

import cv2
import re
from ultralytics import YOLO
import easyocr

PLATE_MODEL_PATH = "/content/drive/MyDrive/license_plate_project/training/plate_v1/weights/best.pt"
VIDEO_PATH = "/content/WhatsApp Video 2026-09-08 at 10.04.48 AM.mp4"   # <-- your real video
OUTPUT_PATH = "/content/plate_output.mp4"

plate_model = YOLO(PLATE_MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True)

ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def enhance_variants(crop):
    up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [up, gray_clahe, otsu]


def read_plate(crop):
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


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video at {VIDEO_PATH} — check the path.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_num = 0
seen_plates = {}
plates_detected_but_unread = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    results = plate_model.predict(frame, conf=0.35, imgsz=640, verbose=False)

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            pad_x = int((x2 - x1) * 0.08)
            pad_y = int((y2 - y1) * 0.15)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)

            plate = frame[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            text, conf = read_plate(plate)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if text:
                cv2.putText(
                    frame, f"{text} ({conf:.2f})", (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                print(f"Frame {frame_num}: '{text}'  conf={conf:.2f}  box=({x1},{y1},{x2},{y2})")
                seen_plates.setdefault(text, []).append(conf)
            else:
                plates_detected_but_unread += 1
                print(f"Frame {frame_num}: plate detected but OCR read nothing. crop shape={plate.shape}")

    out.write(frame)

cap.release()
out.release()

print("\n" + "=" * 50)
print(f"Total frames processed: {frame_num}")
print(f"Plates detected but unreadable by OCR: {plates_detected_but_unread}")
print("\n=== Distinct plate strings read across the whole video ===")
if seen_plates:
    for text, confs in sorted(seen_plates.items(), key=lambda kv: -len(kv[1])):
        print(f"  '{text}'  — seen {len(confs)}x, avg conf {sum(confs)/len(confs):.2f}")
else:
    print("  (none — OCR never produced readable text on this video)")
print(f"\nAnnotated video saved to: {OUTPUT_PATH}")
