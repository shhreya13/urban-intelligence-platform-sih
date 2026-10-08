"""
core.evidence
=============

One place that writes evidence images / clips to disk and hands back a
**relative** path.  Relative matters: the path is stored in the event, saved
in the database and served by the backend at ``/evidence/<path>``, so it has
to be portable between the Windows dev machine and wherever the backend runs.

    store = EvidenceStore(config)
    rel = store.save_image(frame, "road_defect", bus_id="BUS-01")
    # -> "road_defect/BUS-01_000412_ab12cd.jpg"
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from .logging_setup import get_logger


class EvidenceStore:
    def __init__(self, config):
        self.root = config.path("backend.evidence_dir") or Path("outputs/evidence")
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("evidence")
        self.enabled = True
        self.saved = 0

    def _target(self, kind: str, bus_id: Optional[str], frame_number: int, ext: str) -> Path:
        folder = self.root / kind
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{bus_id or 'BUS'}_{frame_number:06d}_{uuid.uuid4().hex[:6]}"
        return folder / f"{stem}.{ext}"

    def save_image(
        self,
        image,
        kind: str,
        bus_id: Optional[str] = None,
        frame_number: int = 0,
    ) -> Optional[str]:
        if not self.enabled or image is None or getattr(image, "size", 0) == 0:
            return None
        try:
            import cv2                                          # noqa: PLC0415
            path = self._target(kind, bus_id, frame_number, "jpg")
            ok = cv2.imwrite(str(path), image)
            if not ok:
                return None
            self.saved += 1
            return str(path.relative_to(self.root)).replace("\\", "/")
        except Exception as exc:                                 # noqa: BLE001
            self.log.warning("could not save %s evidence: %s", kind, exc)
            return None

    def save_clip(
        self,
        frames,
        kind: str,
        fps: float = 25.0,
        bus_id: Optional[str] = None,
        frame_number: int = 0,
    ) -> Optional[str]:
        frames = [f for f in (frames or []) if f is not None]
        if not self.enabled or not frames:
            return None
        try:
            import cv2                                          # noqa: PLC0415
            height, width = frames[0].shape[:2]
            path = self._target(kind, bus_id, frame_number, "mp4")
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height)
            )
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
            writer.release()
            self.saved += 1
            return str(path.relative_to(self.root)).replace("\\", "/")
        except Exception as exc:                                 # noqa: BLE001
            self.log.warning("could not save %s clip: %s", kind, exc)
            return None

    @staticmethod
    def crop(frame, bbox, pad: float = 0.12):
        """Crop a bbox out of a frame with a little padding, safely."""
        if frame is None or bbox is None:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad_x = int((x2 - x1) * pad)
        pad_y = int((y2 - y1) * pad)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()
