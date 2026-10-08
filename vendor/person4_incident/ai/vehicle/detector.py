"""
detector.py — YOLO model loading + plain (non-tracking) detection.

Responsibilities (Person 1 scope only):
    * Load a lightweight pretrained YOLO model (Ultralytics).
    * Resolve CPU/GPU device automatically.
    * Run detection, filtered to the 5 supported classes.
    * Return clean Detection objects.

Explicitly NOT handled here: GPS, FastAPI, SQLite, React, or event
generation — those belong to Person 3, Person 4, and Person 5.

tracker.py reuses the YOLO model instance loaded by this class to run
BoT-SORT tracking, so the model is only ever loaded once per process.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger("ai.vehicle.detector")


class ModelLoadError(RuntimeError):
    """Raised when the YOLO model cannot be loaded (bad name, missing deps, etc.)."""


@dataclass
class Detection:
    """A single, class-filtered YOLO detection for one frame (no tracking id)."""
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coordinates
    class_id: int


def resolve_device() -> str:
    """Pick 'cuda' when an NVIDIA GPU is available, otherwise 'cpu'.

    GPU use is opportunistic, never mandatory — this module must always
    work on a plain CPU laptop.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:  # pragma: no cover - defensive, torch import issues etc.
        logger.warning("CUDA availability check failed, falling back to CPU: %s", exc)
    return "cpu"


class VehicleDetector:
    """Loads and owns the YOLO model used for both detection and tracking."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.4,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        half: Optional[bool] = None,
    ):
        """
        imgsz / half are OPTIONAL performance knobs, additive on top of the
        original detector -- they never change what the model was trained
        on or how it scores a frame's classes:
            imgsz: inference resolution passed to Ultralytics. None (the
                   default) means "don't override" -- Ultralytics' own
                   default (640) is used, i.e. IDENTICAL behavior to before
                   this parameter existed.
            half:  FP16 inference. None (the default) means "auto": enabled
                   only when running on CUDA (free speedup, zero effect on
                   a CPU box), never forced on CPU where torch doesn't
                   support it.
        """
        # Import here (not at module top) so a missing `ultralytics`
        # install produces one clear ModelLoadError instead of an import
        # crash the moment this file is imported anywhere (e.g. by tests).
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError(
                "The 'ultralytics' package is not installed. "
                "Run: pip install -r ai/requirements.txt"
            ) from exc

        from ai.vehicle.classes import SUPPORTED_CLASS_IDS

        self.model_name = model_name
        self.confidence = confidence
        self.device = device or resolve_device()
        self.imgsz = imgsz
        self.half = (self.device == "cuda") if half is None else bool(half)

        try:
            logger.info("Loading YOLO model '%s' on device '%s' ...", model_name, self.device)
            self.model = YOLO(model_name)
            self.model.to(self.device)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load YOLO model '{model_name}'.\n"
                f"Check that:\n"
                f"  1. You have an internet connection (first run auto-downloads weights)\n"
                f"  2. 'ultralytics' is installed correctly (pip install -r ai/requirements.txt)\n"
                f"  3. The model name is a valid Ultralytics checkpoint (e.g. 'yolov8n.pt')\n"
                f"Original error: {exc}"
            ) from exc

        logger.info(
            "YOLO model loaded. Filtering to supported COCO class ids: %s",
            SUPPORTED_CLASS_IDS,
        )

    def detect(self, frame) -> List[Detection]:
        """Run plain (non-tracking) detection on a single frame.

        This exists mainly for isolated testing/debugging of the detector.
        The live demo (inference.py) uses tracker.py instead, which calls
        BoT-SORT tracking directly on this same loaded model.
        """
        from ai.vehicle.classes import SUPPORTED_CLASS_IDS, class_name_from_id

        predict_kwargs = dict(
            source=frame,
            conf=self.confidence,
            classes=SUPPORTED_CLASS_IDS,
            device=self.device,
            verbose=False,
        )
        if self.imgsz:
            predict_kwargs["imgsz"] = self.imgsz
        if self.half:
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    object_type=class_name_from_id(class_id),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                )
            )
        return detections
