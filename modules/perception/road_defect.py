"""
modules.perception.road_defect
==============================

Adapter around **Person 1's** ROAD_DEFECT_MODEL
(``vendor/person1_road_defect/weights/best.pt``, copied to
``models/road_defect_best.pt``).

Person 1 shipped one multi-class detector -- exactly as the spec requires
("do not create separate models for every road defect").  This adapter only
loads those weights and maps their raw class names onto the platform's
canonical vocabulary; it never trains, replaces or wraps a second model.

Raw classes in the shipped checkpoint::

    0 Pothole   1 Damaged road   2 D50   3 Repair
    4 Divider   5 Zebra Crossing 6 Traffic Sign

The raw -> canonical map lives in ``config/platform.yaml`` under
``road_defect_classes`` so Person 1 can retrain (e.g. adding
``Waterlogging``) and only YAML changes.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.frame_context import FrameContext
from core.logging_setup import get_logger
from core.module_base import ModuleUnavailable

#: Fallback map used when the config does not override it.
DEFAULT_CLASS_MAP: Dict[str, str] = {
    "pothole": "pothole",
    "damaged road": "damaged_road",
    "d50": "road_damage_other",      # RDD-style code; confirm with Person 1
    "repair": "road_repair",
    "divider": "divider",
    "zebra crossing": "zebra_crossing",
    "traffic sign": "traffic_signboard",
    "waterlogging": "waterlogging",  # ready for the next training run
    "water logging": "waterlogging",
}


class RoadDefectPerception:
    owner = "Person 1"

    def __init__(self, config):
        self.config = config
        self.log = get_logger("perception.road_defect")
        self.section = config.section("perception.road_defect")
        self.enabled = bool(self.section.get("enabled", True))
        self.frame_stride = max(1, int(self.section.get("frame_stride", 5) or 1))
        self.model = None
        self.device = config.get("platform.device", "cpu")
        self.class_map: Dict[str, str] = dict(DEFAULT_CLASS_MAP)
        self.class_map.update({
            str(k).lower(): str(v)
            for k, v in (config.section("road_defect_classes") or {}).items()
        })
        self._last: List[Dict[str, Any]] = []

    def setup(self) -> None:
        if not self.enabled:
            raise ModuleUnavailable("perception.road_defect disabled in config")

        try:
            from ultralytics import YOLO                        # noqa: PLC0415
        except Exception as exc:                                 # noqa: BLE001
            raise ModuleUnavailable(f"ultralytics not installed: {exc}") from exc

        path = self.config.path("perception.road_defect.model")
        if path is None or not path.exists():
            raise ModuleUnavailable(
                f"ROAD_DEFECT_MODEL weights not found: {path}"
            )

        self.model = YOLO(str(path))
        raw = getattr(self.model, "names", {}) or {}
        unmapped = [
            name for name in raw.values()
            if str(name).lower() not in self.class_map
        ]
        if unmapped:
            self.log.warning(
                "unmapped road-defect classes %s -- they will pass through with "
                "a slugified name. Add them under road_defect_classes in the config.",
                unmapped,
            )
        self.log.info(
            "Person 1 ROAD_DEFECT_MODEL ready (%s, %s classes, every %s frames)",
            path.name, len(raw), self.frame_stride,
        )

    def canonical(self, raw_name: str) -> str:
        key = str(raw_name).lower().strip()
        if key in self.class_map:
            return self.class_map[key]
        return key.replace(" ", "_").replace("-", "_")

    def update(self, ctx: FrameContext) -> List[Dict[str, Any]]:
        """Road defects are static features -- running the model on every
        frame buys nothing, so this honours ``frame_stride``.  On skipped
        frames it returns an empty list rather than the stale one, so the
        road module does not re-report the same pothole 30 times a second."""
        if self.model is None:
            return []
        if ctx.frame_number % self.frame_stride != 0:
            return []

        results = self.model.predict(
            ctx.frame,
            conf=float(self.section.get("confidence_threshold", 0.35)),
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
                    raw_name = str(names[class_id])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class_id": class_id,
                        "raw_class_name": raw_name,
                        "class_name": self.canonical(raw_name),
                        "confidence": float(box.conf[0].item()),
                        "area": (x2 - x1) * (y2 - y1),
                    })
        self._last = detections
        ctx.shared["road_defects"] = detections
        return detections

    @property
    def available(self) -> bool:
        return self.model is not None
