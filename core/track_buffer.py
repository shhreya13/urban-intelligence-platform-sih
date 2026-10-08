"""
core.track_buffer
=================

A rolling buffer of recent vehicle crops, keyed by track id.

This is the piece that makes **event-triggered ANPR** possible without the
incident module importing the ANPR module:

    incident module :  "identify track 24"      (an Event)
    orchestrator    :  crops = buffer.get(24)
    ANPR module     :  process_incident(24, crops)

The buffer is filled every frame (cheaply -- it is a numpy slice, no model
runs), so when an incident fires we already hold the frames from *before*
it happened, which is exactly what a hit-and-run needs.

It is bounded in both directions: at most ``max_tracks`` vehicles and at
most ``per_track`` crops each, oldest evicted first, so a long video cannot
grow it without limit.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Any, Deque, Dict, List, Optional


class VehicleCropBuffer:
    def __init__(
        self,
        per_track: int = 12,
        stride: int = 2,
        max_tracks: int = 80,
        min_height: int = 40,
        pad: float = 0.06,
    ):
        self.per_track = per_track
        self.stride = max(1, stride)
        self.max_tracks = max_tracks
        self.min_height = min_height
        self.pad = pad
        self._tracks: "OrderedDict[int, Deque[Dict[str, Any]]]" = OrderedDict()

    # -- writing ------------------------------------------------------------

    def update(self, ctx) -> None:
        if ctx.frame_number % self.stride != 0:
            return
        frame = ctx.frame
        if frame is None:
            return
        height, width = frame.shape[:2]

        for vehicle in (ctx.vehicles or []):
            track_id = vehicle.get("track_id")
            if track_id is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in vehicle["bbox"]]
            if (y2 - y1) < self.min_height:
                continue
            pad_x = int((x2 - x1) * self.pad)
            pad_y = int((y2 - y1) * self.pad)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            bucket = self._tracks.get(track_id)
            if bucket is None:
                bucket = deque(maxlen=self.per_track)
                self._tracks[track_id] = bucket
            bucket.append({
                "crop": crop,
                "frame_number": ctx.frame_number,
                "video_time_s": ctx.video_time_s,
                "bbox": [x1, y1, x2, y2],
                "area": (x2 - x1) * (y2 - y1),
                "class_name": vehicle.get("class_name"),
            })
            self._tracks.move_to_end(track_id)

        while len(self._tracks) > self.max_tracks:
            self._tracks.popitem(last=False)

    # -- reading ------------------------------------------------------------

    def get(self, track_id: int) -> List[Any]:
        """Crops for a track, biggest (usually closest/clearest) first."""
        bucket = self._tracks.get(track_id)
        if not bucket:
            return []
        ordered = sorted(bucket, key=lambda item: item["area"], reverse=True)
        return [item["crop"] for item in ordered]

    def get_records(self, track_id: int) -> List[Dict[str, Any]]:
        bucket = self._tracks.get(track_id)
        return list(bucket) if bucket else []

    def best_crop(self, track_id: int) -> Optional[Any]:
        crops = self.get(track_id)
        return crops[0] if crops else None

    def has(self, track_id: int) -> bool:
        return bool(self._tracks.get(track_id))

    def __len__(self) -> int:
        return len(self._tracks)


class FrameRingBuffer:
    """Recent full frames, so an incident can be exported as a short clip."""

    def __init__(self, size: int = 60, stride: int = 2):
        self.stride = max(1, stride)
        self._frames: Deque[Any] = deque(maxlen=size)

    def update(self, ctx) -> None:
        if ctx.frame_number % self.stride == 0 and ctx.frame is not None:
            self._frames.append(ctx.frame.copy())

    def snapshot(self) -> List[Any]:
        return list(self._frames)
