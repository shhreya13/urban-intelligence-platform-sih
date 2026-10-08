"""
modules.perception.vehicle
==========================

Adapter around **Person 2's** existing vehicle detection + tracking.

It reuses, unmodified from ``vendor/person2_traffic/``:
    * ``src/detection/vehicle_detector.py``  -> VehicleDetector (YOLO + BoT-SORT)
    * ``src/analytics/speed_estimator.py``   -> SpeedEstimator

The vehicle detector runs **once per frame** here, and the result is written
into ``FrameContext.vehicles``.  Traffic (Person 2) and incident detection
(Person 4) both read that same list, which is what stops two separate vehicle
models being loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from core.frame_context import FrameContext
from core.logging_setup import get_logger
from core.module_base import ModuleUnavailable
from core.vendor_loader import load_vendor_attr, load_vendor_module, VendorImportError

from .clock import VideoClock, patch_module_clock

VEHICLE_DETECTOR_SRC = "person2_traffic/src/detection/vehicle_detector.py"
SPEED_ESTIMATOR_SRC = "person2_traffic/src/analytics/speed_estimator.py"


class VehiclePerception:
    """Shared vehicle detection + tracking + speed, run once per frame."""

    owner = "Person 2"

    def __init__(self, config, clock: VideoClock | None = None):
        self.config = config
        self.log = get_logger("perception.vehicle")
        self.section = config.section("perception.vehicle")
        self.clock = clock or VideoClock()
        self.detector = None
        self.speed_estimator = None
        self.enabled = bool(self.section.get("enabled", True))
        self.frame_stride = max(1, int(self.section.get("frame_stride", 1) or 1))
        self._last: List[Dict[str, Any]] = []
        self.model_path: Path | None = None

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        if not self.enabled:
            raise ModuleUnavailable("perception.vehicle disabled in config")

        model_path = self.config.path("perception.vehicle.model")
        if model_path is None or not model_path.exists():
            raise ModuleUnavailable(
                f"vehicle weights not found: {model_path}. "
                "Set perception.vehicle.model in config/platform.yaml."
            )
        self.model_path = model_path

        try:
            VehicleDetector = load_vendor_attr(VEHICLE_DETECTOR_SRC, "VehicleDetector")
            speed_module = load_vendor_module(SPEED_ESTIMATOR_SRC)
        except VendorImportError as exc:
            raise ModuleUnavailable(f"Person 2 code could not be imported: {exc}") from exc

        if self.section.get("use_video_clock", True):
            patch_module_clock([speed_module], self.clock)

        self.detector = VehicleDetector(
            model_path=str(model_path),
            confidence_threshold=float(self.section.get("confidence_threshold", 0.25)),
            image_size=int(self.section.get("image_size", 640)),
        )
        self.speed_estimator = speed_module.SpeedEstimator(
            smoothing_factor=float(self.section.get("speed_smoothing", 0.7)),
        )
        self.log.info(
            "Person 2 VehicleDetector ready (%s, conf=%.2f, imgsz=%s)",
            model_path.name,
            float(self.section.get("confidence_threshold", 0.25)),
            self.section.get("image_size", 640),
        )

    # -- per frame ----------------------------------------------------------

    def update(self, ctx: FrameContext) -> List[Dict[str, Any]]:
        """Detect + track + estimate speed, then publish into the context."""
        if self.detector is None:
            ctx.vehicles = []
            return []

        if ctx.frame_number % self.frame_stride == 0 or not self._last:
            detections = self.detector.track(ctx.frame)
            detections = self.speed_estimator.update(detections)
            self._last = detections
        ctx.vehicles = self._last
        return self._last

    # -- helpers used by the traffic module ---------------------------------

    def average_speed(self, detections) -> float:
        if self.speed_estimator is None:
            return 0.0
        return self.speed_estimator.average_speed(detections)

    def moving_count(self, detections, minimum_speed: float = 5.0) -> int:
        if self.speed_estimator is None:
            return 0
        return self.speed_estimator.moving_vehicle_count(detections, minimum_speed)

    def slow_count(self, detections, maximum_speed: float = 5.0) -> int:
        if self.speed_estimator is None:
            return 0
        return self.speed_estimator.slow_vehicle_count(detections, maximum_speed)

    @property
    def available(self) -> bool:
        return self.detector is not None
