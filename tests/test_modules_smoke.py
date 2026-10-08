"""
Per-module smoke tests against the REAL weights, on real frames.

These are the "test each module independently" checklist from the
integration spec, expressed as code.  Anything whose model or dependency is
missing is skipped, not failed -- the platform is allowed to run without it.

    pytest tests/test_modules_smoke.py -v
"""

from pathlib import Path

import pytest

from core.config import load_config
from core.event import GPS
from core.evidence import EvidenceStore
from core.frame_context import FrameContext
from core.module_base import ModuleStatus
from core.track_buffer import VehicleCropBuffer

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "data" / "videos" / "cars.mp4"

cv2 = pytest.importorskip("cv2")
pytest.importorskip("ultralytics")


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def frames():
    if not DEMO.exists():
        pytest.skip(f"demo video missing -- run: python scripts/fetch_demo_video.py")
    capture = cv2.VideoCapture(str(DEMO))
    out = []
    for index in range(24):
        ok, frame = capture.read()
        if not ok:
            break
        out.append(cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1]))))
    capture.release()
    if not out:
        pytest.skip("could not decode the demo video")
    return out


def contexts(frames):
    for index, frame in enumerate(frames):
        yield FrameContext(
            frame=frame, frame_number=index, video_time_s=index / 30.0,
            width=frame.shape[1], height=frame.shape[0], fps=30.0,
            bus_id="BUS-TEST", camera_id="CAM-1",
            gps=GPS(13.0827, 80.2707), road_segment="SEG-TEST",
        )


# ------------------------------------------------- 1. vehicle detection

def test_vehicle_detection_and_tracking(config, frames):
    from modules.perception import VehiclePerception
    perception = VehiclePerception(config)
    try:
        perception.setup()
    except Exception as exc:
        pytest.skip(f"vehicle model unavailable: {exc}")

    seen, track_ids = 0, set()
    for ctx in contexts(frames):
        for vehicle in perception.update(ctx):
            seen += 1
            for key in ("bbox", "class_name", "confidence", "track_id", "speed"):
                assert key in vehicle
            if vehicle["track_id"] is not None:
                track_ids.add(vehicle["track_id"])

    assert seen > 0, "Person 2's detector found no vehicles in a traffic clip"
    assert track_ids, "BoT-SORT produced no track ids"


# ------------------------------------------------- 2. road defect model

def test_road_defect_model_loads_and_runs(config, frames):
    from modules.perception import RoadDefectPerception
    perception = RoadDefectPerception(config)
    try:
        perception.setup()
    except Exception as exc:
        pytest.skip(f"ROAD_DEFECT_MODEL unavailable: {exc}")

    assert perception.model is not None
    names = set(str(v).lower() for v in perception.model.names.values())
    assert "pothole" in names, f"unexpected classes: {perception.model.names}"
    for ctx in contexts(frames):
        for detection in perception.update(ctx):
            assert detection["class_name"] == perception.canonical(detection["raw_class_name"])
            assert 0.0 <= detection["confidence"] <= 1.0


# ------------------------------------------------- 3. incident detection

def test_incident_module_loads_person4_detectors(config, frames):
    """Person 4's real classes, on real frames, with no model of their own."""
    from modules.perception import VehiclePerception
    from modules.perception.clock import VideoClock
    from modules.incident import IncidentModule

    clock = VideoClock()
    perception = VehiclePerception(config, clock=clock)
    try:
        perception.setup()
    except Exception as exc:
        pytest.skip(f"vehicle model unavailable: {exc}")

    module = IncidentModule(config, evidence=None)
    assert module.safe_setup(), module.status_detail

    for ctx in contexts(frames):
        perception.update(ctx)
        module.safe_process(ctx)

    assert module.collision_strict is not None
    assert module.hit_and_run is not None
    assert module.status == ModuleStatus.READY
    assert module.error_count == 0
    # ordinary traffic footage must not manufacture incidents
    assert module.events_emitted == 0 or all(
        c.state != "HIT_AND_RUN_CONFIRMED" for c in module._cases.values()
    )


# ------------------------------------------------- 4. traffic analytics

def test_traffic_module_emits_observations(config, frames):
    from modules.perception import VehiclePerception
    from modules.perception.clock import VideoClock
    from modules.traffic import TrafficModule

    clock = VideoClock()
    perception = VehiclePerception(config, clock=clock)
    try:
        perception.setup()
    except Exception as exc:
        pytest.skip(f"vehicle model unavailable: {exc}")

    module = TrafficModule(config, perception=perception, clock=clock)
    assert module.safe_setup(), module.status_detail

    events = []
    for ctx in contexts(frames):
        clock.set_video_time(ctx.video_time_s)
        perception.update(ctx)
        events += module.safe_process(ctx)
        assert "traffic" in ctx.shared

    snapshot = ctx.shared["traffic"]
    for key in ("vehicle_count", "density_percentage", "traffic_level",
                "average_speed_px_s", "bottleneck_score", "counted_total"):
        assert key in snapshot
    assert snapshot["traffic_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert module.status == ModuleStatus.READY


# ------------------------------------------------- 6. ANPR (event-triggered)

def test_anpr_never_runs_per_frame(config, frames):
    from modules.anpr import ANPRModule
    module = ANPRModule(config)
    if not module.safe_setup():
        pytest.skip(f"ANPR unavailable: {module.status_detail}")
    for ctx in contexts(frames):
        assert module.safe_process(ctx) == [], "ANPR must never emit per frame"
    assert module.trigger_count == 0


def test_anpr_reads_a_real_vehicle_crop(config, frames):
    """Runs Person 5's pipeline for real, on vehicle crops taken from the
    demo clip.  Asserts the plumbing, not a specific plate string -- OCR
    accuracy depends on the footage."""
    from modules.anpr import ANPRModule
    from modules.perception import VehiclePerception
    from core.event import EventType, Severity, make_event

    perception = VehiclePerception(config)
    try:
        perception.setup()
    except Exception as exc:
        pytest.skip(f"vehicle model unavailable: {exc}")

    module = ANPRModule(config)
    if not module.safe_setup():
        pytest.skip(f"ANPR unavailable: {module.status_detail}")

    buffer = VehicleCropBuffer(per_track=8, stride=1, min_height=30)
    for ctx in contexts(frames):
        perception.update(ctx)
        buffer.update(ctx)

    track_id = next(
        (t for t in range(1, 200) if len(buffer.get(t)) >= 2), None
    )
    if track_id is None:
        pytest.skip("no vehicle was tracked long enough to buffer crops")

    trigger = make_event(
        EventType.HIT_AND_RUN, "incident", track_id=track_id,
        severity=Severity.CRITICAL, confidence=0.9,
        metadata={"requires_identification": True, "anpr_track_id": track_id},
    )
    assert module.should_trigger(trigger)

    result = module.identify(trigger, buffer.get(track_id))
    assert result is not None
    assert result.event_type == "vehicle_identification"
    assert "registration_number" in result.metadata
    assert "plate_confidence" in result.metadata
    assert result.metadata["triggered_by_event_type"] == "hit_and_run"
    # one identification per track, ever
    assert not module.should_trigger(trigger)


# ------------------------------------------------- 7. graceful degradation

def test_missing_model_marks_module_unavailable_not_crashed():
    config = load_config(overrides={
        "perception": {"road_defect": {"model": "models/does_not_exist.pt"}}
    })
    from modules.perception import RoadDefectPerception
    from modules.road_detection import RoadDetectionModule

    perception = RoadDefectPerception(config)
    with pytest.raises(Exception):
        perception.setup()
    assert perception.available is False

    module = RoadDetectionModule(config, perception=perception)
    assert module.safe_setup() is False
    assert module.status == ModuleStatus.UNAVAILABLE
    # and it is simply skipped instead of raising
    ctx = FrameContext(frame=None, frame_number=1)
    assert module.safe_process(ctx) == []


def test_disabled_module_is_skipped():
    config = load_config(overrides={"modules": {"traffic": {"enabled": False}}})
    from modules.traffic import TrafficModule
    module = TrafficModule(config)
    assert module.safe_setup() is False
    assert module.status == ModuleStatus.DISABLED


# ------------------------------------------------- 8. pedestrian safety (Module 3)

def test_pedestrian_perception_missing_model_marks_unavailable_not_crashed():
    config = load_config(overrides={
        "perception": {"pedestrian": {"model": "models/does_not_exist.pt"}}
    })
    from modules.perception import PedestrianPerception
    from modules.pedestrian import PedestrianSafetyModule

    perception = PedestrianPerception(config)
    with pytest.raises(Exception):
        perception.setup()
    assert perception.available is False

    # the vulnerable-pedestrian module itself only needs Person 3's vendored
    # risk_engine and its config -- neither requires the missing weights --
    # so it stays READY even while perception is UNAVAILABLE, exactly like
    # road_detection stays READY on waterlogging alone (see module docstring).
    module = PedestrianSafetyModule(config)
    assert module.safe_setup() is True


def test_pedestrian_module_person_is_not_a_pedestrian_and_events_are_real():
    """Two regressions in one:

    1. Person 3's original run_safety.py produced
       {"event": "pedestrian_vehicle_risk", "pedestrian_id": "P3", ...} which
       does not parse as a core.event.Event. The module must produce Events.
    2. The integrated module used to call any person near a vehicle inside a
       fixed image rectangle a pedestrian. PERSON != PEDESTRIAN: without a
       DETECTED zebra crossing there must be no event at all, and with one,
       exactly one event per continuous crossing while vehicles move.
    (The full 12-scenario matrix lives in tests/test_vulnerable_pedestrian.py.)
    """
    from modules.pedestrian import PedestrianSafetyModule
    from core.event import Event

    def drive(module, with_zebra):
        events = []
        for f in range(60):
            ctx = FrameContext(
                frame=None, frame_number=f, video_time_s=f / 30.0,
                width=640, height=480, fps=30.0,
                bus_id="BUS-TEST", camera_id="CAM-1", gps=GPS(13.0827, 80.2707),
            )
            ctx.shared["persons"] = [{
                "bbox": [300, 180, 320, 240], "class_id": 0, "class_name": "person",
                "confidence": 0.9, "track_id": 1, "area": 1200,
            }]
            ctx.vehicles = [{
                "bbox": [20 + 4 * f, 120, 100 + 4 * f, 170], "class_id": 2,
                "class_name": "car", "confidence": 0.85, "track_id": 7,
                "area": 4000, "center": (60 + 4 * f, 145), "speed": 120.0,
            }]
            if f % 5 == 0:
                ctx.shared["road_defects"] = [{
                    "bbox": [100, 200, 540, 260], "class_name": "zebra_crossing",
                    "raw_class_name": "Zebra Crossing", "confidence": 0.8,
                }] if with_zebra else []
            events.extend(module.safe_process(ctx))
        return events

    config = load_config(overrides={"modules": {"pedestrian_safety": {"save_evidence": False}}})

    module = PedestrianSafetyModule(config)
    assert module.safe_setup() is True
    assert drive(module, with_zebra=False) == []

    module = PedestrianSafetyModule(config)
    assert module.safe_setup() is True
    events = drive(module, with_zebra=True)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "pedestrian_vehicle_risk"
    assert event.source_module == "pedestrian_safety"
    assert event.severity in ("HIGH", "CRITICAL")
    assert event.track_id == 1
    assert event.metadata["event_label"] == "VULNERABLE_PEDESTRIAN_EVENT"
    assert event.metadata["zebra_crossing_detected"] is True
    assert event.metadata["vehicle_track_ids"] == [7]
    # round-trips through the common Event schema without a TypeError
    Event.from_dict(event.to_dict())
