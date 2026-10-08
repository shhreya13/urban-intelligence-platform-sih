"""
MODULE 1 -- ROAD / INFRASTRUCTURE DETECTION      (owner: Person 1)
==================================================================

Turns Person 1's ROAD_DEFECT_MODEL detections into common Events.

What it emits per detection:
    class, confidence, bounding box, timestamp, frame number, GPS
-- exactly the contract in the spec.

Two things worth knowing:

1. **De-duplication inside one pass.**  A pothole sits in the camera for
   dozens of frames.  Reporting it every frame would drown the backend, so
   the module suppresses a repeat of the same class in roughly the same
   place until ``emit_cooldown_s`` has passed.  Cross-*bus* deduplication is
   a different problem and belongs to Module 6.

2. **Missing infrastructure.**  The spec asks for "missing divider" and
   "missing/damaged zebra crossing".  Those are *absences*, and no detector
   can see an absence -- so instead of inventing a model, this module keeps
   a per-road-segment baseline: once a segment has been seen to contain a
   zebra crossing / divider on several passes, a later pass through the same
   segment that contains none raises a ``missing_*`` event.  Set
   ``infrastructure_absence.enabled: false`` to switch it off.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from core.event import Event, EventType, Severity, SourceModule, make_event
from core.frame_context import FrameContext
from core.module_base import BaseModule, ModuleUnavailable

#: Canonical class -> default severity.
SEVERITY_BY_CLASS: Dict[str, str] = {
    "pothole": Severity.HIGH,
    "damaged_road": Severity.HIGH,
    "road_damage_other": Severity.MEDIUM,
    "waterlogging": Severity.HIGH,
    "missing_divider": Severity.MEDIUM,
    "missing_zebra_crossing": Severity.MEDIUM,
    "damaged_zebra_crossing": Severity.MEDIUM,
    "zebra_crossing": Severity.INFO,
    "divider": Severity.INFO,
    "traffic_signboard": Severity.INFO,
    "road_repair": Severity.INFO,
}

#: Classes that describe healthy infrastructure rather than a problem.
INFRASTRUCTURE_CLASSES = {"zebra_crossing", "divider", "traffic_signboard", "road_repair"}

#: Which infrastructure absences we are willing to infer.
ABSENCE_MAP = {
    "zebra_crossing": "missing_zebra_crossing",
    "divider": "missing_divider",
}


class RoadDetectionModule(BaseModule):
    name = "road_detection"
    owner = "Person 1"

    def __init__(self, config, perception=None, evidence=None, waterlogging_perception=None):
        super().__init__(config.section("modules.road_detection"))
        self.platform_config = config
        self.perception = perception
        self.waterlogging_perception = waterlogging_perception
        self.evidence = evidence

        self.min_confidence = float(self.config.get("min_confidence", 0.35))
        self.cooldown_s = float(self.config.get("emit_cooldown_s", 3.0))
        self.save_evidence = bool(self.config.get("save_evidence", True))
        self.emit_infrastructure = bool(self.config.get("emit_infrastructure", True))

        absence = self.config.get("infrastructure_absence", {}) or {}
        self.absence_enabled = bool(absence.get("enabled", True))
        self.absence_min_baseline = int(absence.get("min_baseline_observations", 3))

        # cooldown key -> last emit time (video seconds)
        self._last_emit: Dict[Tuple[str, int, int], float] = {}
        # road segment -> {class -> times seen}
        self._segment_baseline: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # road segment -> classes seen during the current traversal
        self._segment_current: Dict[str, set] = defaultdict(set)
        self._segment_reported_absence: set = set()
        self._active_segment: Optional[str] = None
        self.detection_count = 0

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        road_ready = self.perception is not None and self.perception.available
        water_ready = (
            self.waterlogging_perception is not None
            and self.waterlogging_perception.available
        )
        if not road_ready and not water_ready:
            raise ModuleUnavailable(
                "neither ROAD_DEFECT_MODEL nor WATERLOGGING_MODEL perception "
                "is available (see the perception.* log lines above)"
            )
        if not road_ready:
            self.log.warning(
                "ROAD_DEFECT_MODEL unavailable -- running on WATERLOGGING_MODEL only"
            )
        if not water_ready:
            self.log.warning(
                "WATERLOGGING_MODEL unavailable -- running on ROAD_DEFECT_MODEL only"
            )

    # -- per frame ----------------------------------------------------------

    def process(self, ctx: FrameContext) -> List[Event]:
        detections: List[Dict[str, Any]] = ctx.shared.get("road_defects") or []
        events: List[Event] = []

        if ctx.road_segment and ctx.road_segment != self._active_segment:
            events.extend(self._close_segment(ctx))
            self._active_segment = ctx.road_segment

        for detection in detections:
            self.detection_count += 1
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            if ctx.road_segment:
                self._segment_current[ctx.road_segment].add(class_name)

            if confidence < self.min_confidence:
                continue
            if class_name in INFRASTRUCTURE_CLASSES and not self.emit_infrastructure:
                continue
            if not self._should_emit(class_name, detection["bbox"], ctx):
                continue

            evidence_image = None
            if self.save_evidence and self.evidence is not None:
                crop = self.evidence.crop(ctx.frame, detection["bbox"])
                evidence_image = self.evidence.save_image(
                    crop, "road_defect", ctx.bus_id, ctx.frame_number
                )

            events.append(make_event(
                EventType.ROAD_DEFECT,
                SourceModule.ROAD,
                confidence=round(confidence, 3),
                severity=self._severity(class_name, detection, ctx),
                class_name=class_name,
                bounding_box=detection["bbox"],
                evidence_image=evidence_image,
                metadata={
                    "raw_class_name": detection.get("raw_class_name"),
                    "area_px": round(float(detection.get("area", 0.0)), 1),
                    "relative_area": round(
                        float(detection.get("area", 0.0))
                        / max(1.0, float(ctx.width * ctx.height)), 5
                    ),
                    "model": detection.get("source_model", "ROAD_DEFECT_MODEL"),
                    "detector": detection.get(
                        "source_weights", "person1_road_defect/weights/best.pt"
                    ),
                },
            ))

        return events

    def finalize(self) -> List[Event]:
        events: List[Event] = []
        if self._active_segment:
            events.extend(self._close_segment(None))
        return events

    # -- internals ----------------------------------------------------------

    def _should_emit(self, class_name: str, bbox, ctx: FrameContext) -> bool:
        """One event per class per screen-region per cooldown window."""
        cell_x = int(((bbox[0] + bbox[2]) / 2) // max(1, ctx.width // 6 or 1))
        cell_y = int(((bbox[1] + bbox[3]) / 2) // max(1, ctx.height // 6 or 1))
        key = (class_name, cell_x, cell_y)
        now = ctx.video_time_s if ctx.video_time_s else time.time()
        last = self._last_emit.get(key)
        if last is not None and (now - last) < self.cooldown_s:
            return False
        self._last_emit[key] = now
        return True

    def _severity(self, class_name: str, detection, ctx: FrameContext) -> str:
        base = SEVERITY_BY_CLASS.get(class_name, Severity.LOW)
        # A defect filling a large part of the frame is close and big.
        area_ratio = float(detection.get("area", 0.0)) / max(
            1.0, float(ctx.width * ctx.height)
        )
        if class_name in ("pothole", "damaged_road", "waterlogging") and area_ratio > 0.045:
            return Severity.CRITICAL
        return base

    def _close_segment(self, ctx: Optional[FrameContext]) -> List[Event]:
        """Called when the bus leaves a road segment: fold what we saw into
        the baseline, and raise a missing_* event where an established
        feature was not observed on this pass."""
        events: List[Event] = []
        segment = self._active_segment
        if not segment:
            return events

        seen = self._segment_current.pop(segment, set())
        baseline = self._segment_baseline[segment]

        if self.absence_enabled and ctx is not None:
            for feature, absence_type in ABSENCE_MAP.items():
                key = (segment, absence_type)
                if key in self._segment_reported_absence:
                    continue
                if baseline.get(feature, 0) >= self.absence_min_baseline and feature not in seen:
                    self._segment_reported_absence.add(key)
                    events.append(make_event(
                        EventType.ROAD_DEFECT,
                        SourceModule.ROAD,
                        confidence=0.55,
                        severity=SEVERITY_BY_CLASS.get(absence_type, Severity.MEDIUM),
                        class_name=absence_type,
                        road_segment=segment,
                        metadata={
                            "inferred": True,
                            "reason": (
                                f"'{feature}' was observed on "
                                f"{baseline[feature]} previous passes of this "
                                "segment but not on this one"
                            ),
                            "baseline_observations": baseline[feature],
                        },
                    ))

        for feature in seen:
            baseline[feature] += 1
        return events

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        data["raw_detections"] = self.detection_count
        return data
