"""
Vulnerable-pedestrian rule (Module 3) -- PERSON != PEDESTRIAN.

Frame-by-frame scenario tests for SIH26124 "identify vulnerable pedestrian
situations such as school children crossing roads". Every scenario feeds
the module exactly what the pipeline publishes on the FrameContext:

    ctx.shared["persons"]       person detections + tracker ids (Person 3 perception)
    ctx.vehicles                vehicle detections + tracker ids (Person 2 perception)
    ctx.shared["road_defects"]  ROAD_DEFECT_MODEL output, ONLY on its strided
                                frames (every 5th frame, as configured)

No model weights are needed, so these always run:

    python -m pytest tests/test_vulnerable_pedestrian.py -v
"""

from __future__ import annotations

import json

import pytest

from core.config import load_config
from core.event import GPS, Event
from core.frame_context import FrameContext
from modules.pedestrian import PedestrianSafetyModule
from modules.pedestrian.crossing_logic import CrossingPhase

FPS = 30.0
W, H = 1280, 720
ROAD_STRIDE = 5

#: A zebra crossing as Person 1's model reports it (canonical class name).
ZEBRA_BOX = [200, 400, 1080, 520]
ZEBRA = {
    "bbox": ZEBRA_BOX, "class_id": 5, "raw_class_name": "Zebra Crossing",
    "class_name": "zebra_crossing", "confidence": 0.82, "area": 880 * 120,
}
POTHOLE = {
    "bbox": [500, 560, 620, 620], "class_id": 0, "raw_class_name": "Pothole",
    "class_name": "pothole", "confidence": 0.77, "area": 7200,
}


# --------------------------------------------------------------------------- helpers

def make_module(**module_overrides) -> PedestrianSafetyModule:
    section = {"save_evidence": False}
    section.update(module_overrides)
    config = load_config(overrides={"modules": {"pedestrian_safety": section}})
    module = PedestrianSafetyModule(config)
    assert module.safe_setup() is True
    return module


def person(track_id, x, feet_y, class_name="person", width=40, height=140, conf=0.88):
    return {
        "bbox": [int(x), int(feet_y - height), int(x + width), int(feet_y)],
        "class_id": 0, "class_name": class_name, "confidence": conf,
        "track_id": track_id, "area": width * height,
        "center": (int(x + width / 2), int(feet_y - height / 2)),
    }


def vehicle(track_id, x, y, w=160, h=90, cls="car", conf=0.9, speed=0.0):
    return {
        "bbox": [int(x), int(y), int(x + w), int(y + h)], "class_id": 2,
        "class_name": cls, "confidence": conf, "track_id": track_id,
        "area": w * h, "center": (int(x + w / 2), int(y + h / 2)), "speed": speed,
    }


def moving_cars(f):
    """Two cars travelling on the road beside the crossing (4-5 px/frame)."""
    return [
        vehicle(11, 60 + 4 * f % 900, 300),
        vehicle(12, 1100 - (5 * f % 900), 540, w=180, h=100),
    ]


def parked_cars(f):
    """Parked cars: only detector box jitter (+/-2 px), no real motion."""
    jitter = [0, 2, -1, 1, -2, 0, 1][f % 7]
    return [
        vehicle(21, 80 + jitter, 300 - jitter),
        vehicle(22, 1120 - jitter, 545 + jitter, w=150, h=95),
    ]


def run(module, frames, persons_fn, vehicles_fn, zebra=True, start=0, road_extra=()):
    """Drive the module the way the orchestrator does. Returns (events, per-frame
    snapshots of what the module published)."""
    events, snaps = [], []
    for f in range(start, start + frames):
        ctx = FrameContext(
            frame=None, frame_number=f, video_time_s=f / FPS,
            width=W, height=H, fps=FPS, bus_id="BUS-TEST", camera_id="CAM-1",
            gps=GPS(13.0827, 80.2707), road_segment="SEG-TEST",
        )
        ctx.vehicles = vehicles_fn(f)
        ctx.shared["persons"] = persons_fn(f)
        if f % ROAD_STRIDE == 0:
            ctx.shared["road_defects"] = ([dict(ZEBRA)] if zebra else []) + list(road_extra)
        produced = module.safe_process(ctx)
        events.extend(produced)
        snaps.append({
            "frame": f,
            "events": produced,
            "persons": ctx.shared.get("persons") or [],
            "pedestrians": ctx.shared.get("pedestrians") or [],
            "zebra_crossings": ctx.shared.get("zebra_crossings") or [],
        })
    return events, snaps


def roles(snaps, track_id):
    out = []
    for snap in snaps:
        for p in snap["persons"]:
            if p.get("track_id") == track_id:
                out.append(p["role"])
    return out


def no_pedestrians_anywhere(snaps):
    return all(not snap["pedestrians"] for snap in snaps)


# ------------------------------------------------------------- the 12 scenarios

def test_01_person_walking_on_sidewalk_no_event():
    module = make_module()
    # sidewalk strip nearer the camera than the crossing (feet at y=640)
    events, snaps = run(module, 120, lambda f: [person(1, 150 + 3 * f, 640)], moving_cars)
    assert events == []
    assert no_pedestrians_anywhere(snaps)
    assert set(roles(snaps, 1)) == {"person"}
    assert snaps[-1]["zebra_crossings"], "the crossing itself was detected"


def test_02_person_standing_beside_road_no_event():
    module = make_module()
    # beside the carriageway, right of the crossing's end (x > 1080)
    events, snaps = run(module, 120, lambda f: [person(2, 1150, 500)], moving_cars)
    assert events == []
    assert no_pedestrians_anywhere(snaps)
    assert set(roles(snaps, 2)) == {"person"}


def test_03_person_on_zebra_no_vehicles_no_event():
    module = make_module()
    events, snaps = run(module, 120, lambda f: [person(3, 600, 470)], lambda f: [])
    assert events == []
    assert no_pedestrians_anywhere(snaps)
    # shown as PERSON / CROSSING once the crossing is confirmed, never PEDESTRIAN
    assert "crossing" in roles(snaps, 3)
    assert "pedestrian" not in roles(snaps, 3)
    last = next(p for p in snaps[-1]["persons"] if p["track_id"] == 3)
    assert last["reason"] == "no vehicles present"


def test_04_person_on_zebra_parked_vehicles_no_event():
    module = make_module()
    events, snaps = run(module, 150, lambda f: [person(4, 600, 470)], parked_cars)
    assert events == []
    assert no_pedestrians_anywhere(snaps)
    last = next(p for p in snaps[-1]["persons"] if p["track_id"] == 4)
    assert last["vehicles_present"] is True
    assert last["vehicle_moving"] is False
    assert last["role"] == "crossing"


def test_05_person_on_zebra_with_moving_vehicles_one_event():
    module = make_module()
    events, snaps = run(module, 90, lambda f: [person(5, 600, 470)], moving_cars)
    assert len(events) == 1
    event = events[0]
    meta = event.metadata

    assert event.event_type == "pedestrian_vehicle_risk"
    assert event.source_module == "pedestrian_safety"
    assert meta["event_label"] == "VULNERABLE_PEDESTRIAN_EVENT"
    assert meta["person_track_id"] == 5 and event.track_id == 5
    assert set(meta["vehicle_track_ids"]) <= {11, 12} and meta["vehicle_track_ids"]
    assert meta["zebra_crossing_detected"] is True
    assert meta["vehicle_moving"] is True
    assert meta["crossing_confirmed"] is True
    assert meta["person_on_vehicle"] is False
    assert event.timestamp and event.gps is not None
    assert event.gps.latitude == pytest.approx(13.0827)
    assert 0.0 < event.confidence <= 1.0
    assert event.severity in ("HIGH", "CRITICAL")
    assert event.frame_number is not None

    # temporal confirmation: never on the first qualifying frame
    assert meta["confirmed_frame"] - meta["candidate_start_frame"] >= module.confirm_frames - 1
    assert event.frame_number >= module.confirm_frames

    # role changes PERSON / CROSSING -> PEDESTRIAN only after confirmation
    r = roles(snaps, 5)
    first_ped = r.index("pedestrian")
    assert all(role in ("person", "crossing") for role in r[:first_ped])
    assert snaps[first_ped]["frame"] == event.frame_number
    assert [p["track_id"] for p in snaps[-1]["pedestrians"]] == [5]

    # schema round-trip, and the published person dicts stay JSON-safe
    Event.from_dict(event.to_dict())
    json.dumps(event.to_dict(), default=str)
    json.dumps(snaps[-1]["persons"])


def test_06_person_crosses_zebra_while_vehicles_move_one_event():
    module = make_module()
    walk = lambda f: [person(6, 230 + 9 * f, 470)]           # 230 -> ~1040 px
    events, snaps = run(module, 90, walk, moving_cars)
    assert len(events) == 1
    assert events[0].metadata["crossing_motion"] == "moving_across"
    assert events[0].metadata["crossing_progress_px"] > 0


def test_07_multiple_people_one_event_each_no_spam():
    module = make_module()
    group = lambda f: [person(71, 400, 470), person(72, 520, 480), person(73, 700 + f, 465)]
    events, snaps = run(module, 150, group, moving_cars)
    assert sorted(e.track_id for e in events) == [71, 72, 73]
    assert len({e.event_id for e in events}) == 3
    assert set(events[0].metadata["concurrent_person_track_ids_on_crossing"]) == {71, 72, 73}


def test_08_same_person_on_crossing_for_50_plus_frames_is_one_event():
    module = make_module()
    events, snaps = run(module, 50 + module_confirm_default(), lambda f: [person(8, 640, 470)], moving_cars)
    assert len(events) == 1
    # still one after a much longer stay
    more, _ = run(module, 200, lambda f: [person(8, 640, 470)], moving_cars, start=50 + module_confirm_default())
    assert more == []
    assert module.machine.get(8).phase == CrossingPhase.CONFIRMED


def module_confirm_default():
    return max(5, round(FPS / 3))


def test_09_person_leaves_crossing_event_closes():
    module = make_module()
    on_then_off = lambda f: [person(9, 600, 470 if f < 60 else 650)]
    events, snaps = run(module, 100, on_then_off, moving_cars)
    assert len(events) == 1
    state = module.machine.get(9)
    assert state.phase == CrossingPhase.IDLE
    closed = [c for c in module.closed_episodes if c["person_track_id"] == 9]
    assert len(closed) == 1 and closed[0]["confirmed"] is True
    assert closed[0]["event_id"] == events[0].event_id
    assert closed[0]["closed_frame"] == 60 + module.exit_frames - 1
    assert roles(snaps, 9)[-1] == "person"
    assert snaps[-1]["pedestrians"] == []


def test_10_person_enters_again_later_new_event_allowed():
    """The spec timeline: enter at 100, confirmed ~110, leave at 150, a new
    crossing at 200 may raise a new event."""
    module = make_module()

    def timeline(f):
        on = 100 <= f < 150 or 200 <= f < 260
        return [person(10, 600, 470 if on else 660)]

    events, snaps = run(module, 170, timeline, moving_cars, start=90)
    assert len(events) == 2
    first, second = events
    assert 100 + module.confirm_frames - 1 <= first.frame_number < 150
    assert second.frame_number >= 200
    assert first.event_id != second.event_id


def test_10b_brief_step_off_is_the_same_crossing_not_a_new_event():
    module = make_module()
    flicker = lambda f: [person(15, 600, 470 if not (60 <= f < 63) else 660)]   # 3 frames off
    events, _ = run(module, 150, flicker, moving_cars)
    assert len(events) == 1


def test_11_vehicles_detected_but_stationary_no_event():
    """High detection confidence and a noisy upstream 'speed' value must not
    stand in for real track displacement."""
    module = make_module()
    stationary = lambda f: [
        vehicle(31, 300, 300, conf=0.99, speed=42.0),
        vehicle(32, 900, 540, conf=0.97, speed=35.0),
    ]
    events, snaps = run(module, 150, lambda f: [person(11, 600, 470)], stationary)
    assert events == []
    assert no_pedestrians_anywhere(snaps)
    last = next(p for p in snaps[-1]["persons"] if p["track_id"] == 11)
    assert last["reason"] == "vehicles present but not moving"


def test_12_no_zebra_crossing_detected_no_event():
    module = make_module()
    people = lambda f: [person(121, 600, 470), person(122, 300 + 5 * f % 700, 470)]
    events, snaps = run(module, 150, people, moving_cars, zebra=False, road_extra=[POTHOLE])
    assert events == []
    assert no_pedestrians_anywhere(snaps)
    for snap in snaps:
        assert snap["zebra_crossings"] == []
        for p in snap["persons"]:
            assert p["role"] == "person"
            assert p["zebra_crossing_detected"] is False


# ---------------------------------------------------------------- extra guards

def test_single_qualifying_frame_never_triggers():
    module = make_module()
    # zebra confirmed over frames 0..5, person appears on it for ONE frame at 20
    blip = lambda f: [person(40, 600, 470)] if f == 20 else []
    events, _ = run(module, 60, blip, moving_cars)
    assert events == []


def test_person_inside_bus_on_crossing_area_is_not_a_pedestrian():
    module = make_module()
    # a bus straddling the crossing; passenger feet inside the bus box, above its ground line
    bus = lambda f: [vehicle(50, 380 + 4 * f % 200, 330, w=420, h=200, cls="bus"),
                     vehicle(11, 60 + 4 * f % 900, 300)]
    passenger = lambda f: [person(41, 520 + 4 * f % 200, 470, height=70)]
    events, snaps = run(module, 120, passenger, bus)
    assert events == []
    assert any(p.get("on_vehicle") for s in snaps for p in s["persons"])


def test_person_standing_just_behind_crossing_is_not_on_it():
    """Head/torso inside the crossing's image box, feet on the pavement beyond it."""
    module = make_module()
    events, snaps = run(module, 120, lambda f: [person(42, 600, 395, height=160)], moving_cars)
    assert events == []
    assert set(roles(snaps, 42)) == {"person"}


def test_single_road_model_pass_is_not_a_confirmed_crossing():
    # ego-motion off so this isolates the min_detections rule
    module = make_module(vehicle_motion={"ego_motion_compensation": False})
    events, published = [], []
    for f in range(0, 120):
        ctx = FrameContext(frame=None, frame_number=f, video_time_s=f / FPS, width=W,
                           height=H, fps=FPS, gps=GPS(13.08, 80.27))
        ctx.vehicles = moving_cars(f)
        ctx.shared["persons"] = [person(43, 600, 470)]
        if f % ROAD_STRIDE == 0:
            ctx.shared["road_defects"] = [dict(ZEBRA)] if f == 0 else []
        events.extend(module.safe_process(ctx))
        published.append((ctx.shared["zebra_crossings"], ctx.shared["persons"][0]["role"]))
    assert events == []
    assert all(zebras == [] and role == "person" for zebras, role in published)


def test_same_scene_with_two_passes_does_trigger_without_ego_compensation():
    """Control for the test above: identical scene, crossing seen on every pass."""
    module = make_module(vehicle_motion={"ego_motion_compensation": False})
    events, _ = run(module, 60, lambda f: [person(43, 600, 470)], moving_cars)
    assert len(events) == 1


def _noisy_scene(seed, crossing_jitter, drift, cars_moving, frames=150):
    """Detector noise on everything: crossing box corners jitter by up to
    +/-crossing_jitter px per pass, vehicle boxes by +/-2 px per frame, and the
    whole scene optionally drifts (bus driving) by ``drift`` px/frame."""
    import random
    rnd = random.Random(seed)
    module = make_module()
    events = []
    for f in range(frames):
        dy = drift * f
        ctx = FrameContext(frame=None, frame_number=f, video_time_s=f / FPS, width=W,
                           height=H, fps=FPS, gps=GPS(13.08, 80.27))
        vj = lambda: rnd.randint(-2, 2)
        if cars_moving:
            ctx.vehicles = [vehicle(63, 60 + 6 * f + vj(), 300 + dy + vj())]
        else:
            ctx.vehicles = [vehicle(61, 300 + vj(), 300 + dy + vj()),
                            vehicle(62, 900 + vj(), 540 + dy + vj())]
        ctx.shared["persons"] = [person(57, 600, 470 + dy)]
        if f % ROAD_STRIDE == 0:
            j = lambda: rnd.randint(-crossing_jitter, crossing_jitter)
            z = dict(ZEBRA)
            z["bbox"] = [ZEBRA_BOX[0] + j(), ZEBRA_BOX[1] + dy + j(),
                         ZEBRA_BOX[2] + j(), ZEBRA_BOX[3] + dy + j()]
            ctx.shared["road_defects"] = [z]
        events.extend(module.safe_process(ctx))
    return events


@pytest.mark.parametrize("drift", [0.0, 2.5])
@pytest.mark.parametrize("crossing_jitter", [10, 25])
def test_noisy_crossing_box_parked_cars_never_trigger(drift, crossing_jitter):
    """Crossing-box jitter must not be mistaken for bus motion (which would
    make parked cars look like they move relative to the crossing)."""
    for seed in range(5):
        assert _noisy_scene(seed, crossing_jitter, drift, cars_moving=False) == [], seed


@pytest.mark.parametrize("drift", [0.0, 2.5])
@pytest.mark.parametrize("crossing_jitter", [10, 25])
def test_noisy_crossing_box_moving_traffic_still_triggers_once(drift, crossing_jitter):
    for seed in range(5):
        assert len(_noisy_scene(seed, crossing_jitter, drift, cars_moving=True)) == 1, seed


def test_generic_person_is_never_called_a_school_child():
    module = make_module()
    events, _ = run(module, 60, lambda f: [person(44, 600, 470)], moving_cars)
    meta = events[0].metadata
    assert meta["pedestrian_category"] == "vulnerable pedestrian"
    assert meta["age_group"] is None
    assert "school" not in json.dumps(events[0].to_dict()).lower()


def test_child_class_from_child_aware_detector_is_reported_as_child():
    module = make_module()
    events, _ = run(module, 60, lambda f: [person(45, 600, 470, class_name="child", height=90)], moving_cars)
    assert len(events) == 1
    assert events[0].metadata["age_group"] == "child"
    assert events[0].metadata["pedestrian_category"] == "child pedestrian"
    assert "school" not in json.dumps(events[0].to_dict()).lower()


def test_tracker_id_switch_mid_crossing_does_not_re_alert():
    module = make_module()
    switch = lambda f: [person(46 if f < 60 else 47, 600, 470)]
    events, _ = run(module, 150, switch, moving_cars)
    assert len(events) == 1
    assert module.counters["id_switch_suppressed"] == 1


def test_bus_moving_parked_cars_do_not_count_as_moving_traffic():
    """Ego-motion: the bus drives toward the crossing, so the crossing, the
    parked cars and the waiting person all slide down the image together.
    Relative to the crossing (painted on the road) nothing is moving."""
    module = make_module()
    drift = 1.5                                               # px/frame, whole scene
    events = []
    for f in range(0, 120):
        dy = int(drift * f)
        ctx = FrameContext(frame=None, frame_number=f, video_time_s=f / FPS, width=W,
                           height=H, fps=FPS, gps=GPS(13.08, 80.27))
        ctx.vehicles = [vehicle(61, 300, 300 + dy), vehicle(62, 900, 540 + dy)]
        ctx.shared["persons"] = [person(48, 600, 470 + dy)]
        if f % ROAD_STRIDE == 0:
            z = dict(ZEBRA)
            z["bbox"] = [ZEBRA_BOX[0], ZEBRA_BOX[1] + dy, ZEBRA_BOX[2], ZEBRA_BOX[3] + dy]
            ctx.shared["road_defects"] = [z]
        events.extend(module.safe_process(ctx))
    assert events == []


def test_bus_moving_but_real_traffic_crossing_still_triggers():
    module = make_module()
    drift = 2.0
    events = []
    for f in range(0, 75):
        dy = int(drift * f)
        ctx = FrameContext(frame=None, frame_number=f, video_time_s=f / FPS, width=W,
                           height=H, fps=FPS, gps=GPS(13.08, 80.27))
        ctx.vehicles = [vehicle(63, 60 + 6 * f, 300 + dy)]    # really moving laterally
        ctx.shared["persons"] = [person(49, 600, 470 + dy)]
        if f % ROAD_STRIDE == 0:
            z = dict(ZEBRA)
            z["bbox"] = [ZEBRA_BOX[0], ZEBRA_BOX[1] + dy, ZEBRA_BOX[2], ZEBRA_BOX[3] + dy]
            ctx.shared["road_defects"] = [z]
        events.extend(module.safe_process(ctx))
    assert len(events) == 1
    assert events[0].metadata["ego_motion_compensated"] is True


def test_untracked_person_is_only_a_person():
    module = make_module()
    untracked = lambda f: [{**person(0, 600, 470), "track_id": None}]
    events, snaps = run(module, 90, untracked, moving_cars)
    assert events == []
    assert no_pedestrians_anywhere(snaps)


def test_require_crossing_motion_option_blocks_a_stationary_person():
    module = make_module(temporal={"require_crossing_motion": True})
    events, _ = run(module, 120, lambda f: [person(50, 600, 470)], moving_cars)
    assert events == []
    module = make_module(temporal={"require_crossing_motion": True})
    events, _ = run(module, 120, lambda f: [person(51, 230 + 7 * f, 470)], moving_cars)
    assert len(events) == 1


def test_stats_expose_episode_counters():
    module = make_module()
    run(module, 90, lambda f: [person(52, 600, 470)], moving_cars)
    stats = module.stats()
    assert stats["crossing"]["events_confirmed"] == 1
    assert stats["events"] == 1


def test_annotator_labels_person_crossing_and_pedestrian():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from pipeline.annotator import Annotator

    module = make_module()
    events, _ = run(module, 60, lambda f: [person(53, 600, 470)], moving_cars)
    ctx = FrameContext(frame=np.zeros((H, W, 3), dtype=np.uint8), frame_number=60,
                       video_time_s=2.0, width=W, height=H, fps=FPS, gps=GPS(13.08, 80.27))
    ctx.vehicles = moving_cars(60)
    ctx.shared["persons"] = [person(53, 600, 470), person(54, 150, 650)]
    module.safe_process(ctx)
    by_id = {p["track_id"]: p["role"] for p in ctx.shared["persons"]}
    assert by_id == {53: "pedestrian", 54: "person"}

    annotator = Annotator(enabled=True)
    annotator.push_events(events, 60)
    frame = annotator.draw(ctx)
    assert frame.shape == (H, W, 3) and frame.any()
    assert annotator._alert_text(events[0]).startswith("VULNERABLE PEDESTRIAN EVENT")


def test_perception_publishes_persons_not_pedestrians():
    config = load_config()
    from modules.perception import PedestrianPerception
    perception = PedestrianPerception(config)       # model not loaded
    ctx = FrameContext(frame=None, frame_number=0)
    perception.update(ctx)
    assert ctx.shared["persons"] == []
    assert "pedestrians" not in ctx.shared
