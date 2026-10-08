"""
anpr_pipeline.py

Event-triggered ANPR (Automatic Number Plate Recognition) module.

Pipeline (matches the Person 5 module in the bus urban-sensing project):

    Incident detected (external — Person 4/6)
        -> Vehicle Track ID + buffered vehicle crops (external)
        -> [THIS MODULE STARTS HERE]
        -> Plate detector (YOLO)
        -> Plate crop
        -> Image enhancement
        -> OCR (EasyOCR)
        -> Multi-frame voting
        -> Plate-format validation
        -> Confidence score
        -> Registration number + confidence

DESIGN GOAL: ANPR must NOT run continuously on the Nano. The only public
entry point is `ANPRPipeline.process_incident(...)`, which is meant to be
called exactly once per flagged incident (e.g. a hit-and-run track_id),
not once per frame. Normal driving never reaches this code.

Usage:
    pipeline = ANPRPipeline(plate_model_path="best.pt")
    result = pipeline.process_incident(track_id=42, vehicle_frames=[...])
    print(result)
    # {'plate': 'TN12AB1234', 'confidence': 0.94, 'track_id': 42,
    #  'frames_used': 4, 'raw_candidates': [...]}
"""

import re
import cv2
import numpy as np
import easyocr
from collections import Counter
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

PLATE_DETECT_CONF = 0.35
PLATE_DETECT_IMGSZ = 640

# Only OCR the N sharpest frames out of the buffered incident window.
# This is the main Nano-optimization lever: OCR is far more expensive
# than the sharpness check, so skip OCR on blurry/duplicate frames.
MAX_FRAMES_TO_OCR = 5

# Minimum Laplacian variance to even attempt OCR on a plate crop.
# Crops below this are almost certainly too blurry to read; skip them
# entirely rather than wasting an OCR call.
MIN_SHARPNESS = 40.0

ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Standard Indian plate: 2 letters (state) + 1-2 digits (RTO) +
# 1-3 letters (series) + 4 digits (unique number). e.g. TN12AB1234
PLATE_REGEX_STANDARD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

# Newer BH-series format: DD BH DDDD LL. e.g. 22BH1234AB
PLATE_REGEX_BH = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Common OCR character confusions, keyed by the type we expect at that
# position ('L' = letter, 'D' = digit).
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
LETTER_TO_DIGIT = {v: k for k, v in DIGIT_TO_LETTER.items()}
# a couple of extra common ones that aren't clean inverses
LETTER_TO_DIGIT.update({"Q": "0", "D": "0"})


# ============================================================
# PLATE DETECTION
# ============================================================

class PlateDetector:
    """Wraps the fine-tuned YOLO plate-detector model. Input is a
    (already cropped) vehicle image, output is the best plate crop
    found in it, or None."""

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_plate_crop(self, vehicle_img):
        results = self.model.predict(
            vehicle_img,
            conf=PLATE_DETECT_CONF,
            imgsz=PLATE_DETECT_IMGSZ,
            verbose=False
        )

        best_box = None
        best_conf = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

        if best_box is None:
            return None

        h, w = vehicle_img.shape[:2]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())

        # pad so edge characters aren't clipped
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = vehicle_img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop, best_conf


# ============================================================
# FRAME QUALITY SCORING (Nano optimization)
# ============================================================

def sharpness_score(img):
    """Laplacian-variance blur metric. Higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def select_best_frames(plate_crops, max_frames=MAX_FRAMES_TO_OCR):
    """Given a list of plate crops (one per buffered frame), keep only
    the sharpest `max_frames`, dropping anything below MIN_SHARPNESS.
    This is what keeps OCR calls bounded on the Nano — instead of
    running OCR on every frame in the incident window, we only spend
    compute on the frames actually worth reading."""
    scored = [(crop, sharpness_score(crop)) for crop in plate_crops]
    scored = [(c, s) for c, s in scored if s >= MIN_SHARPNESS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:max_frames]]


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def enhance_variants(crop):
    """Produce a few preprocessed versions of a plate crop. Different
    variants win on different lighting/plate conditions, so we try a
    handful and let OCR confidence pick the winner rather than
    committing to one fixed preprocessing recipe."""
    up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [up, gray_clahe, otsu]


# ============================================================
# OCR
# ============================================================

class PlateOCR:
    def __init__(self, gpu=True):
        self.reader = easyocr.Reader(['en'], gpu=gpu)

    def read(self, plate_crop):
        """Run OCR across enhancement variants of one plate crop.
        Returns (text, confidence) — the best-scoring reading found,
        or ("", 0.0) if nothing usable was read."""
        best_text, best_conf = "", 0.0

        for variant in enhance_variants(plate_crop):
            ocr_results = self.reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist=ALLOWED_CHARS
            )
            if not ocr_results:
                continue

            # top-to-bottom then left-to-right, for 2-line plates
            ocr_results = sorted(
                ocr_results,
                key=lambda r: (r[0][0][1], r[0][0][0])
            )

            text = "".join(
                re.sub(r'[^A-Z0-9]', '', t[1].upper()) for t in ocr_results
            )
            avg_conf = sum(t[2] for t in ocr_results) / len(ocr_results)

            if text and avg_conf > best_conf:
                best_text, best_conf = text, avg_conf

        return best_text, best_conf


# ============================================================
# PLATE-FORMAT VALIDATION + CORRECTION
# ============================================================

def _template_for_length(n):
    """Return an expected type-pattern ('L'/'D' per position) for a
    given text length, for the standard Indian format. Middle series
    letters are assumed to fill however many chars are left over
    between the fixed 2-letter/2-digit prefix and 4-digit suffix."""
    if n < 8 or n > 11:
        return None
    # LL + DD + (series letters) + DDDD
    series_len = n - 2 - 2 - 4
    if series_len < 0:
        return None
    return "LL" + "DD" + ("L" * series_len) + "DDDD"


def correct_and_validate(text):
    """Try to coerce OCR text into a valid plate number by correcting
    common character confusions at each position, based on whether a
    letter or digit is expected there. Returns (corrected_text, is_valid)."""
    if not text:
        return text, False

    if PLATE_REGEX_STANDARD.match(text) or PLATE_REGEX_BH.match(text):
        return text, True

    template = _template_for_length(len(text))
    if template is None:
        return text, False

    corrected = []
    for ch, expected in zip(text, template):
        if expected == "L" and ch in DIGIT_TO_LETTER:
            corrected.append(DIGIT_TO_LETTER[ch])
        elif expected == "D" and ch in LETTER_TO_DIGIT:
            corrected.append(LETTER_TO_DIGIT[ch])
        else:
            corrected.append(ch)
    corrected_text = "".join(corrected)

    is_valid = bool(
        PLATE_REGEX_STANDARD.match(corrected_text)
        or PLATE_REGEX_BH.match(corrected_text)
    )
    return corrected_text, is_valid


# ============================================================
# MULTI-FRAME VOTING
# ============================================================

def vote_best_plate(candidates):
    """
    candidates: list of (text, confidence) from multiple frames of the
    SAME tracked vehicle during one incident.

    Strategy:
      1. Correct + validate each candidate against the plate format.
      2. If any valid candidates exist, pick the most frequent exact
         string among them (mode); ties broken by highest confidence.
      3. If none are valid, fall back to per-character majority vote
         across candidates sharing the most common length, then
         re-validate the voted result.

    Returns dict: {plate, confidence, valid, votes}
    """
    corrected = []
    for text, conf in candidates:
        c_text, valid = correct_and_validate(text)
        corrected.append((c_text, conf, valid))

    valid_candidates = [(t, c) for t, c, v in corrected if v]

    if valid_candidates:
        counts = Counter(t for t, _ in valid_candidates)
        most_common_text, freq = counts.most_common(1)[0]
        confs = [c for t, c in valid_candidates if t == most_common_text]
        agreement_ratio = freq / len(candidates)
        final_conf = min(0.99, agreement_ratio * (sum(confs) / len(confs)) + 0.15)
        return {
            "plate": most_common_text,
            "confidence": round(final_conf, 2),
            "valid": True,
            "votes": dict(counts),
        }

    # Fallback: character-level majority vote among same-length reads
    lengths = Counter(len(t) for t, _, _ in corrected if t)
    if not lengths:
        return {"plate": "", "confidence": 0.0, "valid": False, "votes": {}}

    common_len, _ = lengths.most_common(1)[0]
    same_len = [t for t, _, _ in corrected if len(t) == common_len]

    voted_chars = []
    for i in range(common_len):
        col = Counter(t[i] for t in same_len)
        voted_chars.append(col.most_common(1)[0][0])
    voted_text = "".join(voted_chars)

    voted_text, valid = correct_and_validate(voted_text)
    avg_conf = sum(c for _, c, _ in corrected) / len(corrected)
    final_conf = round(avg_conf * (0.7 if valid else 0.4), 2)

    return {
        "plate": voted_text,
        "confidence": final_conf,
        "valid": valid,
        "votes": dict(Counter(same_len)),
    }


# ============================================================
# TOP-LEVEL PIPELINE — the only thing external code should call
# ============================================================

class ANPRPipeline:
    def __init__(self, plate_model_path, gpu=True):
        self.detector = PlateDetector(plate_model_path)
        self.ocr = PlateOCR(gpu=gpu)

    def process_incident(self, track_id, vehicle_frames):
        """
        Call this ONLY when an incident (e.g. hit-and-run) has been
        flagged for `track_id` by the incident-detection module.
        Never call this per-frame during normal driving.

        vehicle_frames: list of already-cropped vehicle images
                        (BGR numpy arrays) buffered around the
                        incident timestamp for this track_id.

        Returns dict with plate, confidence, and diagnostics.
        """
        # 1. Detect + crop plate in each buffered vehicle frame
        plate_crops = []
        for frame in vehicle_frames:
            detected = self.detector.detect_plate_crop(frame)
            if detected is not None:
                crop, _det_conf = detected
                plate_crops.append(crop)

        if not plate_crops:
            return {
                "track_id": track_id,
                "plate": "",
                "confidence": 0.0,
                "valid": False,
                "frames_used": 0,
                "note": "No plate detected in any buffered frame."
            }

        # 2. Nano optimization: only OCR the sharpest few crops
        best_crops = select_best_frames(plate_crops)
        if not best_crops:
            return {
                "track_id": track_id,
                "plate": "",
                "confidence": 0.0,
                "valid": False,
                "frames_used": 0,
                "note": "All plate crops were below the sharpness threshold."
            }

        # 3. OCR each selected crop (enhancement happens inside PlateOCR)
        candidates = [self.ocr.read(crop) for crop in best_crops]
        candidates = [(t, c) for t, c in candidates if t]

        if not candidates:
            return {
                "track_id": track_id,
                "plate": "",
                "confidence": 0.0,
                "valid": False,
                "frames_used": len(best_crops),
                "note": "OCR returned no readable text on any selected frame."
            }

        # 4. Multi-frame voting + format validation -> final result
        result = vote_best_plate(candidates)
        result["track_id"] = track_id
        result["frames_used"] = len(best_crops)
        result["raw_candidates"] = candidates
        return result


# ============================================================
# EXAMPLE (simulated — replace with real integration)
# ============================================================

if __name__ == "__main__":
    # In the real system, Person 4/6's incident detector calls this
    # exactly once per flagged track_id, passing whatever vehicle
    # crops it buffered around the incident.
    pipeline = ANPRPipeline(
        plate_model_path="/content/drive/MyDrive/license_plate_project/training/plate_v1/weights/best.pt"
    )

    # simulated_frames = [cv2.imread(p) for p in glob.glob("incident_42/*.jpg")]
    # result = pipeline.process_incident(track_id=42, vehicle_frames=simulated_frames)
    # print(result)
    # e.g. {'plate': 'TN12AB1234', 'confidence': 0.94, 'valid': True,
    #       'track_id': 42, 'frames_used': 4, 'votes': {...}}
    pass
