"""
tracker.py — BoT-SORT multi-object tracking on top of the YOLO detector.

STRICT REQUIREMENT: this module uses BoT-SORT ONLY. It never falls back to
ByteTrack, silently or otherwise — `BOTSORT_CONFIG_PATH` is passed
explicitly to Ultralytics' `model.track(...)` call below.

Responsibilities:
    * Feed each frame's YOLO detections into BoT-SORT.
    * Read back the stable tracking IDs BoT-SORT assigns.
    * Convert raw tracking results into clean `TrackedObject` records that
      match the structured-output schema Person 3's Event Engine expects.

This file does NOT open a camera, draw anything, or touch GPS/FastAPI/
SQLite — see inference.py for the live demo loop.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
import logging

from ai.vehicle.classes import SUPPORTED_CLASS_IDS, class_name_from_id
from ai.vehicle.detector import VehicleDetector

logger = logging.getLogger("ai.vehicle.tracker")

# Absolute path to our own tuned BoT-SORT config (see botsort_custom.yaml).
# Passing a real path — rather than relying on any default — is what
# guarantees BoT-SORT is what actually runs.
BOTSORT_CONFIG_PATH = str(Path(__file__).parent / "botsort_custom.yaml")


@dataclass
class TrackedObject:
    """One tracked object in one frame — the schema shared with Person 3.

    NOTE: `track_id` is the identifier BoT-SORT assigns. It is NOT a
    guaranteed-permanent real-world identity — see tracker limitations in
    ai/README.md (occlusion, re-entry, crowding, motion blur, etc. can all
    cause an id change).
    """
    object_type: str
    track_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    frame_id: int
    timestamp: str  # ISO-8601, local timezone

    def to_dict(self) -> dict:
        return asdict(self)


class VehicleTracker:
    """Runs BoT-SORT tracking using a shared VehicleDetector's YOLO model."""

    def __init__(self, detector: VehicleDetector):
        self.detector = detector
        self._frame_id = 0

    def process_frame(self, frame) -> List[TrackedObject]:
        """Run detection + BoT-SORT tracking on a single frame.

        Returns one TrackedObject per box BoT-SORT has confirmed with a
        stable track id this frame. Boxes BoT-SORT hasn't confirmed yet
        (very first frame or two of a brand-new detection) are skipped so
        no placeholder/None track_ids leak into the structured output.
        """
        self._frame_id += 1
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")

        # `persist=True` keeps BoT-SORT's internal track state alive across
        # calls instead of resetting it every single frame.
        track_kwargs = dict(
            source=frame,
            conf=self.detector.confidence,
            classes=SUPPORTED_CLASS_IDS,
            device=self.detector.device,
            tracker=BOTSORT_CONFIG_PATH,
            persist=True,
            verbose=False,
        )
        # Same optional, additive perf knobs as detector.py's detect() --
        # None/False leaves Ultralytics' own defaults untouched.
        if getattr(self.detector, "imgsz", None):
            track_kwargs["imgsz"] = self.detector.imgsz
        if getattr(self.detector, "half", False):
            track_kwargs["half"] = True

        results = self.detector.model.track(**track_kwargs)

        tracked: List[TrackedObject] = []
        if not results:
            return tracked

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return tracked

        ids = boxes.id.int().tolist()
        for box, track_id in zip(boxes, ids):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            tracked.append(
                TrackedObject(
                    object_type=class_name_from_id(class_id),
                    track_id=int(track_id),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    frame_id=self._frame_id,
                    timestamp=timestamp,
                )
            )
        return tracked

    def reset(self) -> None:
        """Clear BoT-SORT's internal track state (e.g. for a fresh camera session)."""
        predictor = getattr(self.detector.model, "predictor", None)
        if predictor is not None:
            predictor.trackers = None
        self._frame_id = 0
