"""
core.event
==========

THE COMMON EVENT FORMAT.

Every module in the platform (road defect, traffic, incident,
ANPR, intelligence) produces objects of exactly this shape.  Nothing else
crosses a module boundary.  Modules never import each other's internals --
they only ever exchange ``Event`` objects through the orchestrator.

Design rules (from the integration spec):
  * ONE schema for every module.
  * A module is never forced to fill a field it does not have --
    everything except ``event_type`` / ``source_module`` has a sane default.
  * Anything module-specific lives inside ``metadata`` so the schema itself
    never has to grow a per-module column.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

class EventType:
    """Canonical event_type values.  Kept as plain strings so a module can
    still emit something new without a schema migration."""

    # Module 1 - road infrastructure
    ROAD_DEFECT = "road_defect"

    # Module 2 - traffic intelligence
    TRAFFIC_CONGESTION = "traffic_congestion"
    TRAFFIC_OBSERVATION = "traffic_observation"
    TRAFFIC_BOTTLENECK = "traffic_bottleneck"

    # Module 3 - pedestrian safety
    PEDESTRIAN_RISK = "pedestrian_vehicle_risk"

    # Module 4 - incidents
    RASH_DRIVING = "rash_driving"
    POSSIBLE_COLLISION = "possible_collision"
    COLLISION = "collision"
    HIT_AND_RUN = "hit_and_run"
    ABNORMAL_MOVEMENT = "abnormal_movement"

    # Module 5 - ANPR
    VEHICLE_IDENTIFICATION = "vehicle_identification"

    # Module 6 - intelligence engine
    PERSISTENT_ISSUE = "persistent_issue"

    #: event types that represent an incident (used by the API and dashboard)
    INCIDENT_TYPES = (
        "collision", "possible_collision", "hit_and_run",
        "rash_driving", "abnormal_movement", "pedestrian_vehicle_risk",
    )


class IncidentStatus:
    """Status of an incident record (spec 6: never claim certainty).

    ``SUSPECTED`` is the honest default for anything the geometry inferred;
    ``CONFIRMED`` is reserved for a collision corroborated by kinematics.
    """

    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    CLEARED = "cleared"


class HitAndRunState:
    """The explicit hit-and-run state machine (spec 9).

    NORMAL -> SUSPICIOUS_INTERACTION -> POSSIBLE_COLLISION
           -> POST_COLLISION_MONITORING -> VEHICLE_LEAVING
           -> HIT_AND_RUN_CONFIRMED

    A vehicle that stays put instead resolves to COLLISION_ONLY, and one
    whose track is lost without a departure run resolves to TRACK_LOST.
    Neither of those emits a hit-and-run.
    """

    NORMAL = "NORMAL"
    SUSPICIOUS_INTERACTION = "SUSPICIOUS_INTERACTION"
    POSSIBLE_COLLISION = "POSSIBLE_COLLISION"
    POST_COLLISION_MONITORING = "POST_COLLISION_MONITORING"
    VEHICLE_LEAVING = "VEHICLE_LEAVING"
    HIT_AND_RUN_CONFIRMED = "HIT_AND_RUN_CONFIRMED"
    COLLISION_ONLY = "COLLISION_ONLY"
    TRACK_LOST = "TRACK_LOST"


class SourceModule:
    ROAD = "road_detection"
    TRAFFIC = "traffic"
    PEDESTRIAN = "pedestrian_safety"
    INCIDENT = "incident"
    ANPR = "anpr"
    INTELLIGENCE = "intelligence"


class Severity:
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    ORDER = {INFO: 0, LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4}

    @classmethod
    def rank(cls, value: Optional[str]) -> int:
        return cls.ORDER.get((value or cls.INFO).upper(), 0)

    @classmethod
    def max(cls, a: Optional[str], b: Optional[str]) -> str:
        return a if cls.rank(a) >= cls.rank(b) else b


# ---------------------------------------------------------------------------
# GPS
# ---------------------------------------------------------------------------

@dataclass
class GPS:
    latitude: float = 0.0
    longitude: float = 0.0
    accuracy_m: Optional[float] = None
    speed_kmh: Optional[float] = None
    heading_deg: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "accuracy_m": self.accuracy_m,
            "speed_kmh": self.speed_kmh,
            "heading_deg": self.heading_deg,
        }

    @staticmethod
    def from_any(value) -> Optional["GPS"]:
        if value is None:
            return None
        if isinstance(value, GPS):
            return value
        if isinstance(value, dict):
            return GPS(
                latitude=float(value.get("latitude", 0.0) or 0.0),
                longitude=float(value.get("longitude", 0.0) or 0.0),
                accuracy_m=value.get("accuracy_m"),
                speed_kmh=value.get("speed_kmh"),
                heading_deg=value.get("heading_deg"),
            )
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            return GPS(latitude=float(value[0]), longitude=float(value[1]))
        return None


# ---------------------------------------------------------------------------
# The event
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Event:
    """The single interchange format of the whole platform."""

    event_type: str
    source_module: str

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    bus_id: Optional[str] = None
    camera_id: Optional[str] = None
    timestamp: str = field(default_factory=_now_iso)

    gps: Optional[GPS] = None

    confidence: float = 0.0
    severity: str = Severity.INFO

    track_id: Optional[int] = None
    class_name: Optional[str] = None
    bounding_box: Optional[List[int]] = None      # [x1, y1, x2, y2] pixels

    evidence_image: Optional[str] = None          # path, relative to outputs/
    evidence_video: Optional[str] = None

    frame_number: Optional[int] = None
    video_time_s: Optional[float] = None
    road_segment: Optional[str] = None

    # Filled in by the intelligence engine (module 6), null before that.
    issue_id: Optional[str] = None
    priority_score: Optional[float] = None
    department: Optional[str] = None
    duplicate_of: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["gps"] = self.gps.as_dict() if self.gps else None
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        data = dict(data)
        data["gps"] = GPS.from_any(data.get("gps"))
        known = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        extra = {k: v for k, v in data.items() if k not in known}
        clean = {k: v for k, v in data.items() if k in known}
        if extra:
            clean.setdefault("metadata", {})
            clean["metadata"] = {**(clean.get("metadata") or {}), **extra}
        return cls(**clean)

    # -- convenience --------------------------------------------------------

    @property
    def epoch(self) -> float:
        try:
            return datetime.fromisoformat(self.timestamp).timestamp()
        except Exception:
            return time.time()

    def with_context(self, ctx) -> "Event":
        """Stamp bus/camera/gps/frame info from a FrameContext, without
        overwriting anything the module set deliberately."""
        if self.bus_id is None:
            self.bus_id = ctx.bus_id
        if self.camera_id is None:
            self.camera_id = ctx.camera_id
        if self.gps is None:
            self.gps = ctx.gps
        if self.frame_number is None:
            self.frame_number = ctx.frame_number
        if self.video_time_s is None:
            self.video_time_s = ctx.video_time_s
        if self.road_segment is None:
            self.road_segment = ctx.road_segment
        return self

    def __repr__(self) -> str:                                  # pragma: no cover
        return (
            f"<Event {self.event_type} src={self.source_module} "
            f"sev={self.severity} conf={self.confidence:.2f} "
            f"cls={self.class_name} track={self.track_id}>"
        )


def make_event(event_type: str, source_module: str, **kwargs) -> Event:
    """Small helper so modules don't need to import the dataclass directly."""
    kwargs["gps"] = GPS.from_any(kwargs.get("gps"))
    return Event(event_type=event_type, source_module=source_module, **kwargs)
