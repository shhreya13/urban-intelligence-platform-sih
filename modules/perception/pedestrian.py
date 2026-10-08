"""
modules.perception.pedestrian
==============================

Adapter around **Person 3's** pedestrian/child detector.

Person 3's original ``run_safety.py`` loaded *two* YOLO models itself (a
pedestrian/child model and a second, separate COCO vehicle model) and ran
its own standalone tracking + risk loop on top of both. That duplicated the
vehicle model the platform already runs once per frame for every other
module (Person 2 / Module 2), which is exactly the "run each model once"
rule the rest of the platform follows.

So this adapter keeps only the half of Person 3's stack that is actually
new -- the pedestrian/child detector -- and wires it in the same way as
every other perception source (``vehicle.py``, ``road_defect.py``,
``waterlogging.py``): loaded once here, run once per frame, published onto
the shared ``FrameContext``. ``modules/pedestrian/module.py`` (Module 3)
then pairs these detections against ``ctx.vehicles`` -- the SAME vehicle
detections Module 2 and Module 4 already use -- instead of loading a
second vehicle model.

Detections are written to ``ctx.shared["persons"]`` in the same dict
shape Person 2's vehicle detector uses (``bbox``, ``class_id``,
``class_name``, ``confidence``, ``track_id``, ``area``, ``center``).

PERSON != PEDESTRIAN: this adapter only knows that a human was detected, so
it publishes *persons*. Whether a person is a pedestrian in a vulnerable
crossing situation is decided by Module 3 (``modules/pedestrian/module.py``),
which is the only writer of ``ctx.shared["pedestrians"]``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.frame_context import FrameContext
from core.logging_setup import get_logger
from core.module_base import ModuleUnavailable

#: Class names Person 3's checkpoint is trained on. A plain "person" class
#: id is kept as a fallback so the adapter still does something useful
#: against a generic person detector while a dedicated child-aware model
#: is being trained.
PEDESTRIAN_CLASSES = {"person", "child", "pedestrian"}


class PedestrianPerception:
    """Loads Person 3's pedestrian/child YOLO checkpoint and tracks it."""

    owner = "Person 3"

    def __init__(self, config):
        self.config = config
        self.log = get_logger("perception.pedestrian")
        self.section = config.section("perception.pedestrian")
        self.enabled = bool(self.section.get("enabled", True))
        self.frame_stride = max(1, int(self.section.get("frame_stride", 1) or 1))
        self.device = config.get("platform.device", "cpu")
        self.model = None
        self._last: List[Dict[str, Any]] = []

    # -- lifecycle ------------------------------------------------------

    def setup(self) -> None:
        if not self.enabled:
            raise ModuleUnavailable("perception.pedestrian disabled in config")

        try:
            from ultralytics import YOLO                        # noqa: PLC0415
        except Exception as exc:                                 # noqa: BLE001
            raise ModuleUnavailable(f"ultralytics not installed: {exc}") from exc

        path = self.config.path("perception.pedestrian.model")
        if path is None or not path.exists():
            raise ModuleUnavailable(
                f"pedestrian/child model weights not found: {path}. "
                "Train one with vendor/person3_pedestrian_safety/train.py "
                "and set perception.pedestrian.model in config/platform.yaml "
                "-- until then this module is skipped, not crashed."
            )

        self.model = YOLO(str(path))
        raw_names = getattr(self.model, "names", {}) or {}
        matched_ids = {
            class_id for class_id, name in raw_names.items()
            if str(name).lower() in PEDESTRIAN_CLASSES
        }
        # unknown checkpoint -> keep everything it emits, and don't restrict
        # the tracker's `classes=` filter below (None = no filter)
        self.allowed_ids = matched_ids or set(raw_names.keys())
        self._track_classes = sorted(matched_ids) if matched_ids else None
        self.log.info(
            "pedestrian/child model ready (%s, classes=%s, every %s frame%s)",
            path.name,
            [raw_names[i] for i in sorted(self.allowed_ids)],
            self.frame_stride, "" if self.frame_stride == 1 else "s",
        )

    # -- per frame --------------------------------------------------------

    def update(self, ctx: FrameContext) -> List[Dict[str, Any]]:
        if self.model is None:
            ctx.shared["persons"] = []
            return []

        if ctx.frame_number % self.frame_stride != 0 and self._last:
            ctx.shared["persons"] = [dict(d) for d in self._last]
            return self._last

        results = self.model.track(
            ctx.frame,
            persist=True,
            tracker=str(self.section.get("tracker", "botsort.yaml")),
            conf=float(self.section.get("confidence_threshold", 0.35)),
            iou=float(self.section.get("iou", 0.50)),
            imgsz=int(self.section.get("image_size", 640)),
            device=self.device,
            # Restrict BoT-SORT to the person/child classes, the same way
            # Person 2's VehicleDetector restricts to [2, 3, 5, 7]. Without
            # this, the tracker runs its re-identification pass across every
            # COCO class in the frame (cars, trucks, ...) even though
            # everything but person/child is discarded a few lines below --
            # a second, redundant full-frame tracking pass on top of the
            # filtering, and a possible source of stray one-frame "person"
            # tracks from a vehicle the tracker briefly mis-groups.
            classes=self._track_classes,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if results:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                names = result.names
                ids = boxes.id
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    if class_id not in self.allowed_ids:
                        continue
                    confidence = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    track_id = int(ids[i].item()) if ids is not None else None
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class_id": class_id,
                        "class_name": str(names.get(class_id, "person")).lower(),
                        "confidence": confidence,
                        "track_id": track_id,
                        "area": (x2 - x1) * (y2 - y1),
                        "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        "source_model": "PEDESTRIAN_CHILD_MODEL",
                        "source_weights": "models/pedestrian_child_best.pt",
                    })

        self._last = detections
        ctx.shared["persons"] = [dict(d) for d in detections]
        return detections

    @property
    def available(self) -> bool:
        return self.model is not None
