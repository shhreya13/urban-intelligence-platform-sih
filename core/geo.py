"""
core.geo
========

GPS helpers shared by every module and by the intelligence engine.

* ``haversine_m``      -- distance between two lat/lon points, in metres.
* ``road_segment_id``  -- quantises a lat/lon into a stable road-segment key.
  Two buses passing the same pothole from different lanes land in the same
  segment, which is what makes cross-bus deduplication work.
* ``GPSProvider``      -- pluggable source of position for the current frame.
  In this phase there is no bus hardware, so the default provider walks a
  straight line between two configured points.  Swap the class, not the
  pipeline, when a real GPS/NMEA feed arrives.
"""

from __future__ import annotations

import math
from typing import Optional

from .event import GPS

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def road_segment_id(gps: Optional[GPS], segment_size_m: float = 60.0) -> Optional[str]:
    """Quantise a position onto a grid of ``segment_size_m`` squares.

    Returns something like ``SEG-13.08270-80.27070``.  It is deterministic,
    so the same physical stretch of road always produces the same key for
    every bus in the fleet.
    """
    if gps is None:
        return None
    lat_step = segment_size_m / 111_320.0
    lon_scale = max(math.cos(math.radians(gps.latitude)), 1e-6)
    lon_step = segment_size_m / (111_320.0 * lon_scale)
    lat_q = round(gps.latitude / lat_step) * lat_step
    lon_q = round(gps.longitude / lon_step) * lon_step
    return f"SEG-{lat_q:.5f}-{lon_q:.5f}"


class GPSProvider:
    """Base interface.  ``position(frame_number, total_frames)`` -> GPS|None."""

    def position(self, frame_number: int, total_frames: int = 0) -> Optional[GPS]:
        raise NotImplementedError


class NoGPS(GPSProvider):
    def position(self, frame_number: int, total_frames: int = 0) -> Optional[GPS]:
        return None


class FixedGPS(GPSProvider):
    def __init__(self, latitude: float, longitude: float):
        self._gps = GPS(latitude=latitude, longitude=longitude, accuracy_m=5.0)

    def position(self, frame_number: int, total_frames: int = 0) -> Optional[GPS]:
        return self._gps


class SimulatedRouteGPS(GPSProvider):
    """Interpolates along a straight line for the length of the video.

    This is explicitly a *development stand-in* for bus hardware.  It is not
    a fake detection -- it only supplies the position metadata that a real
    bus would provide, so the GPS-dependent logic (segments, dedup,
    multi-bus correlation) can be exercised on a normal laptop.
    """

    def __init__(self, start, end, fps: float = 30.0, speed_kmh: float = 28.0):
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self.fps = fps or 30.0
        self.speed_kmh = speed_kmh
        self._total_m = haversine_m(*self.start, *self.end) or 1.0
        self._heading = self._bearing(self.start, self.end)

    @staticmethod
    def _bearing(a, b) -> float:
        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dl = lon2 - lon1
        y = math.sin(dl) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dl)
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    def position(self, frame_number: int, total_frames: int = 0) -> Optional[GPS]:
        if total_frames and total_frames > 1:
            t = min(1.0, max(0.0, frame_number / float(total_frames - 1)))
        else:
            # No known length (webcam/stream): advance at the configured speed.
            travelled = (self.speed_kmh / 3.6) * (frame_number / self.fps)
            t = min(1.0, travelled / self._total_m)
        lat = self.start[0] + (self.end[0] - self.start[0]) * t
        lon = self.start[1] + (self.end[1] - self.start[1]) * t
        return GPS(
            latitude=round(lat, 7),
            longitude=round(lon, 7),
            accuracy_m=5.0,
            speed_kmh=self.speed_kmh,
            heading_deg=round(self._heading, 1),
        )


def build_gps_provider(config, fps: float = 30.0) -> GPSProvider:
    """Factory driven by the ``gps`` section of the platform config."""
    section = config.section("gps")
    mode = str(section.get("mode", "simulated")).lower()
    if mode == "none":
        return NoGPS()
    start = section.get("start") or [13.0827, 80.2707]
    if mode == "fixed":
        return FixedGPS(float(start[0]), float(start[1]))
    end = section.get("end") or [start[0] + 0.008, start[1] + 0.009]
    return SimulatedRouteGPS(start, end, fps=fps)
