"""
test_anpr_video.py

Standalone test for anpr_pipeline.py using a real video, before the
vehicle tracker / incident trigger exists.

It simulates an "incident buffer" by grabbing a short window of raw
frames from the video and feeding them straight into
ANPRPipeline.process_incident(). No vehicle cropping happens here —
the plate detector searches the full frame, which works, just less
efficiently than a real tracked vehicle crop would.
"""

import cv2
from anpr_pipeline import ANPRPipeline

VIDEO_PATH = "/content/your_video.avi"          # <-- your video file
PLATE_MODEL_PATH = "/content/drive/MyDrive/license_plate_project/training/plate_v1/weights/best.pt"

# Time window (seconds) where a vehicle + its plate is actually visible
# in your video. START AT WHATEVER PART SHOWS A CAR CLEARLY — see the
# frame-preview snippet below if you're not sure where that is.
START_SEC = 2.0
END_SEC = 5.0

# Don't grab literally every frame — every 3rd is plenty and keeps this
# fast on the Nano's OCR step later.
SAMPLE_EVERY_N_FRAMES = 3

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video at {VIDEO_PATH} — check the path.")

print(f"Video FPS: {fps}")

start_frame = int(START_SEC * fps)
end_frame = int(END_SEC * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

buffered_frames = []
frame_idx = start_frame

while frame_idx < end_frame:
    ret, frame = cap.read()
    if not ret:
        print(f"Video ended early at frame {frame_idx} (video may be shorter than END_SEC).")
        break
    if (frame_idx - start_frame) % SAMPLE_EVERY_N_FRAMES == 0:
        buffered_frames.append(frame)
    frame_idx += 1

cap.release()

print(f"Buffered {len(buffered_frames)} frames for the simulated incident.")

if not buffered_frames:
    raise RuntimeError(
        "No frames were buffered — check START_SEC/END_SEC are within "
        "the video's actual length."
    )

pipeline = ANPRPipeline(plate_model_path=PLATE_MODEL_PATH, gpu=True)

result = pipeline.process_incident(track_id=1, vehicle_frames=buffered_frames)

print("\n=== ANPR RESULT ===")
for k, v in result.items():
    print(f"{k}: {v}")


# ------------------------------------------------------------------
# If the result comes back empty ("No plate detected in any buffered
# frame"), uncomment this to preview one frame and confirm a plate is
# actually visible + roughly in-frame during your chosen window:
# ------------------------------------------------------------------
# import matplotlib.pyplot as plt
# preview_cap = cv2.VideoCapture(VIDEO_PATH)
# preview_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
# ret, preview_frame = preview_cap.read()
# preview_cap.release()
# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()
