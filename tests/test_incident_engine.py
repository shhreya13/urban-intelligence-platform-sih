"""
Incident detection + hit-and-run state machine.

These are the six cases the spec asks for, run against the REAL adapter and
Person 4's real detectors -- no mocks of the logic under test.  Vehicle
tracks are synthesised so each scenario is deterministic:

    CASE 1  two vehicles pass normally          -> no collision
    CASE 2  a vehicle temporarily disappears    -> no hit-and-run
    CASE 3  collision, both vehicles stay       -> collision, no hit-and-run
    CASE 4  collision then departure            -> hit-and-run
    CASE 5  hit-and-run                         -> ANPR requested
    CASE 6  low OCR confidence                  -> UNKNOWN, never invented

    pytest tests/test_incident_engine.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import load_config
from core.event import EventType, GPS, HitAndRunState, Severity
from core.frame_context import FrameContext
from modules.incident import IncidentModule

FPS = 30.0
WIDTH, HEIGHT = 1280, 720


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def vehicle(track_id: int, x: float, y: float, w: float = 90.0, h: float = 70.0,
            class_name: str = "car", speed: float = 0.0):
    return {
        "track_id": track_id,
        "bbox": [int(x), int(y), int(x + w), int(y + h)],
        "class_name": class_name,
        "confidence": 0.9,
        "speed": speed,
        "center": (x + w / 2, y + h / 2),
    }


def context(frame_number: int, vehicles):
    return FrameContext(
        frame=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
        frame_number=frame_number,
        video_time_s=frame_number / FPS,
        width=WIDTH, height=HEIGHT, fps=FPS,
        bus_id="BUS-TEST", camera_id="CAM-1",
        gps=GPS(13.0827, 80.2707), road_segment="SEG-TEST",
        vehicles=list(vehicles),
    )


@pytest.fixture
def module():
    config = load_config(overrides={
        "modules": {"incident": {
            "minimum_track_age": 2,
            "evidence": {"enabled": False},
        }}
    })
    mod = IncidentModule(config, evidence=None)
    assert mod.safe_setup(), "Person 4's detectors should load"
    return mod


def run(module, frames):
    """Feed a list of vehicle-lists through the module, collect all events."""
    events = []
    for number, vehicles in enumerate(frames):
        events.extend(module.safe_process(context(number, vehicles)))
    events.extend(module.safe_finalize())
    return events


def types(events):
    return [e.event_type for e in events]


# ---------------------------------------------------------------------------
# CASE 1 -- two vehicles pass normally -> NO collision
# ---------------------------------------------------------------------------

def test_case1_normal_pass_is_not_a_collision(module):
    """Two vehicles in adjacent lanes, never overlapping, constant speed."""
    frames = []
    for i in range(70):
        frames.append([
            vehicle(1, 200 + i * 6, 400),        # left lane, moving right
            vehicle(2, 900 - i * 6, 560),        # right lane, moving left
        ])
    events = run(module, frames)

    assert EventType.COLLISION not in types(events)
    assert EventType.POSSIBLE_COLLISION not in types(events)
    assert EventType.HIT_AND_RUN not in types(events)


def test_case1b_convoy_overlap_is_not_a_collision(module):
    """Two vehicles travelling together with a constantly overlapping box --
    the classic co-moving false positive.  A real collision has to arrive
    from a separated state."""
    frames = []
    for i in range(70):
        frames.append([
            vehicle(1, 300 + i * 5, 400),
            vehicle(2, 340 + i * 5, 404),        # permanently overlapping
        ])
    events = run(module, frames)
    assert EventType.COLLISION not in types(events)


# ---------------------------------------------------------------------------
# CASE 2 -- temporary tracking loss -> NO hit-and-run
# ---------------------------------------------------------------------------

def test_case2_temporary_track_loss_is_not_a_hit_and_run(module):
    """A vehicle collides, then its track drops out entirely.  Disappearance
    is NOT departure: nothing may be emitted."""
    frames = _collision_frames()
    # after the impact the offender simply vanishes (occlusion / id switch)
    for _ in range(80):
        frames.append([vehicle(2, 640, 404)])

    events = run(module, frames)
    assert EventType.HIT_AND_RUN not in types(events), (
        "a vanished track must never be reported as a hit-and-run"
    )
    case = module._cases.get(1)
    assert case is not None
    assert case.state == HitAndRunState.TRACK_LOST


def test_case2b_track_reappears_within_tolerance(module):
    """A short dropout inside the tolerance window is recovered, and the
    vehicle staying put still resolves as collision-only."""
    frames = _collision_frames()
    for _ in range(10):                       # brief dropout, < track_loss_timeout
        frames.append([vehicle(2, 640, 404)])
    for _ in range(60):                       # back, and staying at the scene
        frames.append([vehicle(1, 600, 400), vehicle(2, 640, 404)])

    events = run(module, frames)
    assert EventType.HIT_AND_RUN not in types(events)
    assert module._cases[1].state in (
        HitAndRunState.COLLISION_ONLY,
        HitAndRunState.POST_COLLISION_MONITORING,
    )


# ---------------------------------------------------------------------------
# CASE 3 -- collision, both vehicles remain -> collision only
# ---------------------------------------------------------------------------

def test_case3_collision_with_both_vehicles_staying(module):
    frames = _collision_frames()
    for _ in range(140):                      # both stay put at the scene
        frames.append([vehicle(1, 600, 400), vehicle(2, 640, 404)])

    events = run(module, frames)
    assert EventType.COLLISION in types(events), "the impact itself must be reported"
    assert EventType.HIT_AND_RUN not in types(events)
    assert module._cases[1].state == HitAndRunState.COLLISION_ONLY


def test_case3b_collision_is_critical_and_requests_anpr(module):
    frames = _collision_frames()
    frames += [[vehicle(1, 600, 400), vehicle(2, 640, 404)] for _ in range(20)]
    events = run(module, frames)

    collisions = [e for e in events if e.event_type == EventType.COLLISION]
    assert collisions, "expected a confirmed collision"
    collision = collisions[0]
    assert collision.severity == Severity.CRITICAL
    # A confirmed collision now requests a plate read directly -- it no
    # longer waits on a hit-and-run that may never happen.
    assert collision.metadata["requires_identification"] is True
    assert collision.metadata["anpr_track_id"] == collision.track_id
    assert collision.metadata["status"] == "confirmed"
    assert collision.metadata["detector"] == "person4.CollisionDetector"


# ---------------------------------------------------------------------------
# CASE 4 -- collision then departure -> HIT-AND-RUN
# ---------------------------------------------------------------------------

def test_case4_collision_then_departure_is_a_hit_and_run(module):
    frames = _collision_frames()
    # the offender accelerates away from the incident point, frame by frame
    for i in range(1, 60):
        frames.append([
            vehicle(1, 600 + i * 14, 400),
            vehicle(2, 640, 404),             # the victim stays
        ])

    events = run(module, frames)
    hits = [e for e in events if e.event_type == EventType.HIT_AND_RUN]
    assert hits, "a confirmed collision followed by a sustained departure is a hit-and-run"

    hit = hits[0]
    assert hit.track_id == 1
    assert hit.severity == Severity.CRITICAL
    assert hit.metadata["status"] == "suspected", "never claimed as certain"
    assert hit.metadata["escape_distance_px"] > 0
    assert module._cases[1].state == HitAndRunState.HIT_AND_RUN_CONFIRMED


def test_case4b_state_machine_records_the_full_timeline(module):
    frames = _collision_frames()
    for i in range(1, 60):
        frames.append([vehicle(1, 600 + i * 14, 400), vehicle(2, 640, 404)])
    events = run(module, frames)

    hit = next(e for e in events if e.event_type == EventType.HIT_AND_RUN)
    states = [step["state"] for step in hit.metadata["timeline"]]
    assert HitAndRunState.POSSIBLE_COLLISION in states
    assert HitAndRunState.POST_COLLISION_MONITORING in states
    assert HitAndRunState.VEHICLE_LEAVING in states
    assert states[-1] == HitAndRunState.HIT_AND_RUN_CONFIRMED
    # every step is timestamped against the video
    assert all("video_time_s" in step for step in hit.metadata["timeline"])


# ---------------------------------------------------------------------------
# CASE 5 -- hit-and-run triggers ANPR
# ---------------------------------------------------------------------------

def test_case5_hit_and_run_requests_identification(module):
    frames = _collision_frames()
    for i in range(1, 60):
        frames.append([vehicle(1, 600 + i * 14, 400), vehicle(2, 640, 404)])
    events = run(module, frames)

    hit = next(e for e in events if e.event_type == EventType.HIT_AND_RUN)
    assert hit.metadata["requires_identification"] is True
    assert hit.metadata["anpr_track_id"] == 1

    # The confirmed collision now also asks for ANPR directly, on top of
    # the hit-and-run confirmation -- OCR no longer waits on escape alone.
    asked = [
        e.event_type for e in events
        if e.metadata.get("requires_identification")
    ]
    assert asked == [EventType.COLLISION, EventType.HIT_AND_RUN]


def test_case5b_anpr_never_runs_on_a_clean_pass(module):
    frames = [[vehicle(1, 200 + i * 6, 400), vehicle(2, 900 - i * 6, 560)]
              for i in range(70)]
    events = run(module, frames)
    assert not [e for e in events if e.metadata.get("requires_identification")]


# ---------------------------------------------------------------------------
# CASE 6 -- a poor OCR read is UNKNOWN, never invented
# ---------------------------------------------------------------------------

def test_case6_low_confidence_ocr_returns_unknown():
    """The ANPR adapter must publish UNKNOWN rather than a plate string the
    OCR is not confident about."""
    from core.event import Event, SourceModule, make_event
    from modules.anpr import ANPRModule

    config = load_config(overrides={
        "modules": {"anpr": {"min_plate_confidence": 0.60}}
    })
    anpr = ANPRModule(config, evidence=None)

    source = make_event(
        EventType.HIT_AND_RUN, SourceModule.INCIDENT,
        track_id=7, severity=Severity.CRITICAL,
        metadata={"requires_identification": True, "anpr_track_id": 7},
    )

    low = anpr._build_event(source, None, 7, {
        "plate": "TN12AB1234", "confidence": 0.31, "valid": True, "frames_used": 4,
    })
    assert low.metadata["registration_number"] == "UNKNOWN"
    assert low.metadata["low_confidence_text"] == "TN12AB1234"
    assert "below" in (low.metadata["note"] or "")

    good = anpr._build_event(source, None, 7, {
        "plate": "TN12AB1234", "confidence": 0.88, "valid": True, "frames_used": 5,
    })
    assert good.metadata["registration_number"] == "TN12AB1234"


def test_case6b_no_crops_is_unknown_not_a_guess():
    from core.event import SourceModule, make_event
    from modules.anpr import ANPRModule

    anpr = ANPRModule(load_config(), evidence=None)
    source = make_event(
        EventType.HIT_AND_RUN, SourceModule.INCIDENT, track_id=9,
        metadata={"requires_identification": True, "anpr_track_id": 9},
    )
    event = anpr._build_event(source, None, 9, {
        "plate": "", "confidence": 0.0, "valid": False, "frames_used": 0,
        "note": "no buffered vehicle crops were available for this track",
    })
    assert event.metadata["registration_number"] == "UNKNOWN"
    assert event.metadata["plate_confidence"] == 0.0


# ---------------------------------------------------------------------------
# resilience
# ---------------------------------------------------------------------------

def test_a_broken_frame_does_not_kill_the_module(module):
    ctx = context(0, [{"track_id": 1, "bbox": None, "class_name": "car"}])
    assert module.safe_process(ctx) == []
    assert module.status == "READY", "one bad frame must not disable the module"


def test_rash_driving_needs_more_than_one_signal(module):
    """A vehicle that is merely fast, with no acceleration or lateral
    movement, is not rash driving."""
    frames = [[vehicle(1, 100 + i * 8, 400)] for i in range(60)]
    events = run(module, frames)
    assert EventType.RASH_DRIVING not in types(events)


# ---------------------------------------------------------------------------
# the shared collision prologue
# ---------------------------------------------------------------------------

def _collision_frames():
    """Vehicle 1 approaches vehicle 2 from a clear gap, contacts it, and
    stops hard -- the signature Person 4's detector confirms."""
    frames = []
    # 1. separated approach: the gap must exist before contact
    for i in range(24):
        frames.append([
            vehicle(1, 300 + i * 12, 400),
            vehicle(2, 640, 404),
        ])
    # 2. contact: boxes overlap for a few frames while the approach stops
    for i in range(6):
        frames.append([
            vehicle(1, 588 + i, 400),
            vehicle(2, 640, 404),
        ])
    return frames
