"""
core.frame_context
==================

The per-frame object that travels through the pipeline.

Two jobs:

1. Carry the frame plus its situational metadata (bus, camera, GPS, time).
2. Carry the SHARED perception results so that vehicle detection runs
   **once per frame** and every module that needs it reads the same list.
   This is what stops Person 2 and Person 4 from each loading their own
   vehicle model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .event import GPS


@dataclass
class FrameContext:
    frame: Any                                   # np.ndarray BGR, the raw frame
    frame_number: int = 0
    video_time_s: float = 0.0
    wall_time: float = 0.0

    bus_id: Optional[str] = None
    camera_id: Optional[str] = None
    gps: Optional[GPS] = None
    road_segment: Optional[str] = None

    fps: float = 30.0
    width: int = 0
    height: int = 0

    # ---- shared perception results (filled by the orchestrator) ----
    # Each detection dict is the Person-2 format:
    #   bbox, class_id, class_name, confidence, track_id, width, height,
    #   area, center, speed
    vehicles: List[Dict[str, Any]] = field(default_factory=list)

    # Free-form scratch space shared between modules within one frame.
    shared: Dict[str, Any] = field(default_factory=dict)

    def annotate_source(self, source) -> "FrameContext":
        self.fps = source.fps
        self.width = source.width
        self.height = source.height
        return self
