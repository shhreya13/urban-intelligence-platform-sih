"""The common event format is the contract between every module -- if this
file fails, integration is broken by definition."""

from core.event import Event, EventType, GPS, Severity, make_event
from core.frame_context import FrameContext


def test_minimal_event_has_every_schema_field():
    event = make_event(EventType.ROAD_DEFECT, "road_detection")
    data = event.to_dict()
    for field in (
        "event_id", "event_type", "source_module", "bus_id", "camera_id",
        "timestamp", "gps", "confidence", "severity", "track_id", "class_name",
        "bounding_box", "evidence_image", "evidence_video", "metadata",
    ):
        assert field in data, f"schema is missing {field}"


def test_modules_are_not_forced_to_fill_fields_they_lack():
    event = make_event(EventType.TRAFFIC_CONGESTION, "traffic")
    assert event.track_id is None
    assert event.bounding_box is None
    assert event.evidence_image is None
    assert event.gps is None


def test_round_trip_through_json():
    event = make_event(
        EventType.HIT_AND_RUN, "incident",
        gps={"latitude": 13.08, "longitude": 80.27},
        track_id=24, confidence=0.88, severity=Severity.CRITICAL,
        metadata={"evidence": "did not stop"},
    )
    restored = Event.from_dict(event.to_dict())
    assert restored.event_id == event.event_id
    assert restored.gps.latitude == 13.08
    assert restored.metadata["evidence"] == "did not stop"


def test_unknown_keys_land_in_metadata_instead_of_raising():
    restored = Event.from_dict({
        "event_type": "road_defect", "source_module": "road_detection",
        "something_new_a_module_added": 42,
    })
    assert restored.metadata["something_new_a_module_added"] == 42


def test_context_stamping_does_not_overwrite_module_values():
    ctx = FrameContext(frame=None, frame_number=7, video_time_s=1.4,
                       bus_id="BUS-09", camera_id="CAM-1",
                       gps=GPS(1.0, 2.0), road_segment="SEG-A")
    event = make_event("road_defect", "road_detection", bus_id="BUS-OVERRIDE")
    event.with_context(ctx)
    assert event.bus_id == "BUS-OVERRIDE"      # module value wins
    assert event.camera_id == "CAM-1"
    assert event.frame_number == 7
    assert event.road_segment == "SEG-A"


def test_severity_ordering():
    assert Severity.rank("CRITICAL") > Severity.rank("HIGH") > Severity.rank("LOW")
    assert Severity.max("LOW", "HIGH") == "HIGH"
