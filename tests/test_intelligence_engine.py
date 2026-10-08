"""
Module 6 tests: deduplication, multi-bus correlation, persistence, repair
verification, priority and department routing.

This is the scenario from the spec, run for real:

    Bus 1: POTHOLE at GPS A
    Bus 2: POTHOLE at GPS A      ->  one issue, POT-001, 3 observations
    Bus 3: POTHOLE at GPS A
"""

import tempfile
from pathlib import Path

from backend.db import Database
from core.config import load_config
from core.event import EventType, GPS, Severity, make_event
from modules.intelligence import IntelligenceEngine, route_department

GPS_A = {"latitude": 13.08270, "longitude": 80.27070}
GPS_FAR = {"latitude": 13.09500, "longitude": 80.28500}


def engine_with_db(tmp_path, **overrides):
    config = load_config(overrides=overrides or None)
    database = Database(Path(tmp_path) / "test.db").initialise()
    engine = IntelligenceEngine(config, database=database)
    engine.setup()
    return engine, database


def pothole(bus, gps=GPS_A, timestamp=None, class_name="pothole"):
    event = make_event(
        EventType.ROAD_DEFECT, "road_detection",
        bus_id=bus, camera_id="CAM-1", gps=gps,
        confidence=0.72, severity=Severity.HIGH, class_name=class_name,
        bounding_box=[10, 10, 90, 60],
    )
    if timestamp:
        event.timestamp = timestamp
    return event


def test_three_buses_reporting_the_same_pothole_become_one_issue():
    with tempfile.TemporaryDirectory() as tmp:
        engine, _ = engine_with_db(tmp)
        events = [pothole("BUS-01"), pothole("BUS-02"), pothole("BUS-03")]
        for event in events:
            engine.process_events([event])

        assert len({e.issue_id for e in events}) == 1, "should collapse into one issue"
        issue_id = events[0].issue_id
        assert issue_id.startswith("POT-")

        issue = engine.issues()[0]
        assert issue["observation_count"] == 3
        assert issue["bus_count"] == 3
        assert set(issue["buses"]) == {"BUS-01", "BUS-02", "BUS-03"}
        # the 2nd and 3rd are marked as duplicates of the first sighting
        assert events[1].duplicate_of is not None
        assert events[2].duplicate_of is not None


def test_a_pothole_somewhere_else_is_a_separate_issue():
    with tempfile.TemporaryDirectory() as tmp:
        engine, _ = engine_with_db(tmp)
        here = pothole("BUS-01", GPS_A)
        far = pothole("BUS-01", GPS_FAR)
        engine.process_events([here])
        engine.process_events([far])
        assert here.issue_id != far.issue_id


def test_different_defect_classes_do_not_merge():
    with tempfile.TemporaryDirectory() as tmp:
        engine, _ = engine_with_db(tmp)
        hole = pothole("BUS-01", class_name="pothole")
        water = pothole("BUS-01", class_name="waterlogging")
        engine.process_events([hole])
        engine.process_events([water])
        assert hole.issue_id != water.issue_id
        assert water.issue_id.startswith("WTR-")


def test_department_routing_matches_the_spec():
    assert route_department("road_defect", "pothole") == "Roads/PWD"
    assert route_department("road_defect", "damaged_road") == "Roads/PWD"
    assert route_department("road_defect", "waterlogging") == "Drainage"
    assert route_department("traffic_congestion", None) == "Traffic"
    assert route_department("hit_and_run", "car") == "Traffic Police"
    assert route_department("hit_and_run", None) == "Traffic Police"
    assert route_department("bus_delay", None) == "Transport"


def test_priority_rises_with_persistence_and_severity():
    with tempfile.TemporaryDirectory() as tmp:
        engine, _ = engine_with_db(tmp)
        first = pothole("BUS-01")
        engine.process_events([first])
        low = first.priority_score

        for bus in ("BUS-02", "BUS-03", "BUS-04"):
            for _ in range(4):
                engine.process_events([pothole(bus)])
        last = pothole("BUS-05")
        engine.process_events([last])

        assert last.priority_score > low
        assert last.metadata["priority_band"] in ("MEDIUM", "HIGH", "CRITICAL")
        assert set(last.metadata["priority_components"]) == {
            "severity", "persistence", "traffic_impact", "safety_risk"
        }


def _pass_through(engine, bus, segment_gps, defect=None):
    """Simulate one bus traversing a segment: a traffic observation there,
    optionally a defect sighting, then moving on to the next segment."""
    if defect:
        engine.process_events([pothole(bus, segment_gps, class_name=defect)])
    engine.process_events([make_event(
        EventType.TRAFFIC_OBSERVATION, "traffic", bus_id=bus, gps=segment_gps,
        metadata={"traffic_level": "LOW", "vehicle_count": 3},
    )])
    # leaving the segment
    engine.process_events([make_event(
        EventType.TRAFFIC_OBSERVATION, "traffic", bus_id=bus, gps=GPS_FAR,
        metadata={"traffic_level": "LOW", "vehicle_count": 2},
    )])


def test_repair_verification_resolves_then_reopens():
    overrides = {"modules": {"intelligence": {
        "persistence": {"resolve_after_clean_passes": 2}}}}
    with tempfile.TemporaryDirectory() as tmp:
        engine, _ = engine_with_db(tmp, **overrides)

        _pass_through(engine, "BUS-01", GPS_A, defect="pothole")
        issue = engine.issues()[0]
        assert issue["status"] == "OPEN"

        # authority repairs it -> two clean passes with no detection
        _pass_through(engine, "BUS-02", GPS_A)
        _pass_through(engine, "BUS-03", GPS_A)
        issue = engine.issues()[0]
        assert issue["status"] == "RESOLVED", issue
        assert issue["resolved_at"] is not None

        # ... and a later bus sees it again
        _pass_through(engine, "BUS-04", GPS_A, defect="pothole")
        issue = engine.issues()[0]
        assert issue["status"] == "REOPENED"
        assert issue["reopened_count"] == 1


def test_issue_state_changes_are_published_as_events():
    with tempfile.TemporaryDirectory() as tmp:
        engine, _ = engine_with_db(tmp)
        generated = engine.process_events([pothole("BUS-01")])
        assert [e.event_type for e in generated] == ["persistent_issue"]
        assert generated[0].metadata["issue_state"] == "OPENED"
        assert generated[0].metadata["department"] == "Roads/PWD"


def test_state_survives_a_restart_via_the_database():
    with tempfile.TemporaryDirectory() as tmp:
        engine, database = engine_with_db(tmp)
        event = pothole("BUS-01")
        engine.process_events([event])
        issue_id = event.issue_id
        database.close()

        # a second pipeline run, e.g. bus 2's video processed tomorrow
        config = load_config()
        database2 = Database(Path(tmp) / "test.db").initialise()
        engine2 = IntelligenceEngine(config, database=database2)
        engine2.setup()
        second = pothole("BUS-02")
        engine2.process_events([second])
        assert second.issue_id == issue_id, "correlation must span runs"
        assert engine2.issues()[0]["observation_count"] == 2
