"""
pipeline.video_source
=====================

Video input for this phase: a local file, a webcam index, or an RTSP/HTTP
stream URL.  No Jetson, no bus hardware.

The rest of the pipeline only ever sees ``FrameContext`` objects, so
replacing this class with a CSI-camera reader later changes nothing
downstream -- which is the "hardware-independent" requirement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from core.frame_context import FrameContext
from core.logging_setup import get_logger


class VideoSourceError(RuntimeError):
    pass


class VideoSource:
    def __init__(
        self,
        source,
        resize_width: int = 0,
        start_frame: int = 0,
        max_frames: int = 0,
        loop: bool = False,
    ):
        import cv2                                              # noqa: PLC0415
        self._cv2 = cv2
        self.log = get_logger("video")

        self.raw_source = source
        self.is_camera = False
        opencv_source = source

        if isinstance(source, str) and source.isdigit():
            opencv_source = int(source)
        if isinstance(opencv_source, int):
            self.is_camera = True
        elif isinstance(opencv_source, str) and "://" in opencv_source:
            self.is_camera = True          # network stream: treat as live
        else:
            path = Path(str(opencv_source))
            if not path.exists():
                raise VideoSourceError(
                    f"video not found: {path}\n"
                    "Put a clip in data/videos/ and point video.source at it, "
                    "or set video.source: 0 to use a webcam."
                )
            opencv_source = str(path)

        self.capture = cv2.VideoCapture(opencv_source)
        if not self.capture.isOpened():
            raise VideoSourceError(f"could not open video source: {source}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        self.native_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.native_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.is_camera:
            self.total_frames = 0

        self.resize_width = int(resize_width or 0)
        self.width = self.native_width
        self.height = self.native_height
        if self.resize_width and self.native_width:
            scale = self.resize_width / float(self.native_width)
            self.width = self.resize_width
            self.height = int(round(self.native_height * scale))

        self.start_frame = max(0, int(start_frame or 0))
        self.max_frames = max(0, int(max_frames or 0))
        self.loop = bool(loop)

        if self.start_frame and not self.is_camera:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        self.log.info(
            "source=%s  %sx%s @ %.1ffps  frames=%s%s",
            source, self.width, self.height, self.fps,
            self.total_frames or "unknown",
            f"  (resized from {self.native_width}x{self.native_height})"
            if self.resize_width else "",
        )

    # -- iteration ----------------------------------------------------------

    @property
    def expected_frames(self) -> int:
        if self.max_frames:
            return self.max_frames
        if self.total_frames:
            return max(0, self.total_frames - self.start_frame)
        return 0

    def frames(self) -> Iterator[FrameContext]:
        index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                if self.loop and not self.is_camera:
                    self.capture.set(self._cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                    continue
                break

            if self.resize_width and frame.shape[1] != self.width:
                frame = self._cv2.resize(frame, (self.width, self.height))

            ctx = FrameContext(
                frame=frame,
                frame_number=index,
                video_time_s=round((self.start_frame + index) / self.fps, 3),
                width=frame.shape[1],
                height=frame.shape[0],
                fps=self.fps,
            )
            yield ctx

            index += 1
            if self.max_frames and index >= self.max_frames:
                break

    def release(self) -> None:
        try:
            self.capture.release()
        except Exception:                                        # noqa: BLE001
            pass


def build_video_source(config) -> VideoSource:
    section = config.section("video")
    source = section.get("source", 0)
    if isinstance(source, str) and not source.isdigit() and "://" not in source:
        source = str(config.resolve(source))
    return VideoSource(
        source=source,
        resize_width=section.get("resize_width", 0),
        start_frame=section.get("start_frame", 0),
        max_frames=section.get("max_frames", 0),
        loop=section.get("loop", False),
    )
