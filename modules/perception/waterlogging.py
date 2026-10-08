"""
modules.perception.waterlogging
================================

Adapter around the dedicated **WATERLOGGING_MODEL**
(``models/waterlogging_best.pt``).

This is a single-class YOLO detector trained only on ``waterlogging`` --
unlike Person 1's ROAD_DEFECT_MODEL, it is not a multi-class checkpoint, so
it gets its own small adapter rather than a row in ``road_defect_classes``.
It is still wired in exactly the same way as the other three models
(vehicle, road defect, plate): loaded once in the orchestrator, read once
per frame, and skipped cleanly if the weights are not present.

Detections are written into the SAME ``ctx.shared["road_defects"]`` list
that ``RoadDefectPerception`` fills, tagged with ``class_name:
"waterlogging"``.  That list is already the single input the road module,
the annotator, and the intelligence engine (severity, department routing,
priority) all read -- so a waterlogging hit is treated exactly like any
other road defect without touching that downstream code.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.frame_context import FrameContext
from core.logging_setup import get_logger
from core.module_base import ModuleUnavailable

#: Canonical class name this adapter always emits.
CANONICAL_CLASS = "waterlogging"


class WaterloggingPerception:
    owner = "Waterlogging model"

    def __init__(self, config):
        self.config = config
        self.log = get_logger("perception.waterlogging")
        self.section = config.section("perception.waterlogging")
        self.enabled = bool(self.section.get("enabled", True))
        self.frame_stride = max(1, int(self.section.get("frame_stride", 5) or 1))
        self.model = None
        self.device = config.get("platform.device", "cpu")
        self._last: List[Dict[str, Any]] = []

    def setup(self) -> None:
        if not self.enabled:
            raise ModuleUnavailable("perception.waterlogging disabled in config")

        try:
            from ultralytics import YOLO                        # noqa: PLC0415
        except Exception as exc:                                 # noqa: BLE001
            raise ModuleUnavailable(f"ultralytics not installed: {exc}") from exc

        path = self.config.path("perception.waterlogging.model")
        if path is None or not path.exists():
            raise ModuleUnavailable(
                f"WATERLOGGING_MODEL weights not found: {path}"
            )

        self.model = YOLO(str(path))
        raw = getattr(self.model, "names", {}) or {}
        self.log.info(
            "WATERLOGGING_MODEL ready (%s, %s class%s, every %s frames)",
            path.name, len(raw), "" if len(raw) == 1 else "es", self.frame_stride,
        )

    def update(self, ctx: FrameContext) -> List[Dict[str, Any]]:
        """Standing water is a static condition like a pothole -- no need to
        run inference on every frame, so this also honours ``frame_stride``.
        Detections are appended to ``ctx.shared["road_defects"]`` (created by
        ``RoadDefectPerception`` if it ran first, or created here otherwise)
        rather than overwriting it, so both models can contribute in the
        same frame."""
        if self.model is None:
            return []
        if ctx.frame_number % self.frame_stride != 0:
            return []

        results = self.model.predict(
            ctx.frame,
            conf=float(self.section.get("confidence_threshold", 0.4)),
            imgsz=int(self.section.get("image_size", 640)),
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if results:
            result = results[0]
            names = result.names
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    raw_name = str(names.get(class_id, "Waterlogging"))
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class_id": class_id,
                        "raw_class_name": raw_name,
                        "class_name": CANONICAL_CLASS,
                        "confidence": float(box.conf[0].item()),
                        "area": (x2 - x1) * (y2 - y1),
                        "source_model": "WATERLOGGING_MODEL",
                        "source_weights": "models/waterlogging_best.pt",
                    })

        self._last = detections
        ctx.shared["road_defects"] = list(ctx.shared.get("road_defects") or []) + detections
        return detections

    @property
    def available(self) -> bool:
        return self.model is not None
