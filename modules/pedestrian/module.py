"""
MODULE 3 -- PEDESTRIAN SAFETY / VULNERABLE PEDESTRIANS   (owner: Person 3)
===========================================================================

SIH26124 asks the platform to "identify vulnerable pedestrian situations
such as school children crossing roads". This module answers exactly that,
and nothing looser.

--------------------------------------------------------------------------
PERSON != PEDESTRIAN
--------------------------------------------------------------------------
The previous version treated every YOLO ``person`` inside a fixed, hand-drawn
image rectangle (``zones.crossing_zone``) as a pedestrian and scored it
against every nearby vehicle. A person on the pavement, at a bus stop, in a
parked area or sitting inside a bus could therefore raise a pedestrian alert,
and the "crossing" was never a real crossing, just a band of the image.

A **vulnerable pedestrian event** is now raised only when ALL of these hold,
together, for several consecutive frames of the same tracked person:

  1. a person is detected and tracked                ``ctx.shared["persons"]``
  2. a zebra crossing is actually DETECTED           Person 1's ROAD_DEFECT_MODEL,
                                                     class ``zebra_crossing``
  3. the person's feet are ON that crossing          foot-band overlap
     (and not on/in a vehicle: bus passenger, rider)
  4. vehicles are present AND actually moving        frame-to-frame displacement
                                                     of vehicle track ids
  5. the person keeps occupying the carriageway      temporal state machine
     (crossing / interacting with the roadway)

If any one is false there is no event, the person is not added to
``ctx.shared["pedestrians"]``, and the annotator shows them as ``PERSON``
(or ``PERSON / CROSSING`` when they are on a detected crossing but the other
conditions are not met).

One continuous crossing produces exactly ONE event (per-track state machine
in ``crossing_logic.CrossingStateMachine``); leaving the crossing closes the
episode, and a later, separate crossing may raise a new one.

--------------------------------------------------------------------------
WHAT IS STILL PERSON 3's CODE
--------------------------------------------------------------------------
``vendor/person3_pedestrian_safety/src/risk_engine.py`` (``RiskEngine``) is
loaded unmodified and still scores the confirmed situation (distance to the
nearest moving vehicle, closing speed, child multiplier) to set the event's
severity. It no longer decides WHETHER an event exists: the five conditions
above do.

--------------------------------------------------------------------------
CHILDREN
--------------------------------------------------------------------------
The shipped detector is the generic COCO model, which only has ``person``.
The module never invents an age: events say ``vulnerable pedestrian``. Only
if a dedicated child-aware checkpoint is configured and it emits class
``child`` does the event carry ``age_group: "child"`` (still not "school
child", which no detector here can know).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.event import Event, EventType, Severity, SourceModule, make_event
from core.frame_context import FrameContext
from core.module_base import BaseModule, ModuleUnavailable
from core.vendor_loader import load_vendor_attr, VendorImportError

from .crossing_logic import (
    CrossingPhase,
    CrossingStateMachine,
    TrackHistory,
    Transition,
    ZebraCrossingTracker,
    bbox_bottom_center,
    bbox_center,
    bbox_diag,
    is_moving,
    person_on_crossing,
    person_on_vehicle,
    point_to_bbox_distance,
    vehicle_occluded_by_person,
)

RISK_ENGINE_SRC = "person3_pedestrian_safety/src/risk_engine.py"

#: Label written into every event so dashboards/logs can tell this apart from
#: any looser "person near vehicle" notion.
EVENT_LABEL = "VULNERABLE_PEDESTRIAN_EVENT"

#: Person 3's risk level -> platform severity.
SEVERITY_BY_LEVEL: Dict[str, str] = {
    "LOW": Severity.LOW,
    "MEDIUM": Severity.MEDIUM,
    "HIGH": Severity.HIGH,
    "CRITICAL": Severity.CRITICAL,
}

#: Roles published on each person detection (read by the annotator).
ROLE_PERSON = "person"            # ordinary detected human
ROLE_CROSSING = "crossing"        # on a detected zebra crossing, not an event
ROLE_PEDESTRIAN = "pedestrian"    # confirmed crossing while vehicles move

#: Class names that a *dedicated* child-aware checkpoint may emit.
CHILD_CLASSES = {"child"}


class PedestrianSafetyModule(BaseModule):
    name = "pedestrian_safety"
    owner = "Person 3"

    def __init__(self, config, evidence=None):
        super().__init__(config.section("modules.pedestrian_safety"))
        self.platform_config = config
        self.evidence = evidence
        self.save_evidence = bool(self.config.get("save_evidence", True))
        self.severity_floor = str(self.config.get("severity_floor", Severity.HIGH)).upper()

        zebra = self.config.get("zebra_crossing") or {}
        self.zebra_classes = {
            str(c).lower() for c in (zebra.get("classes") or ["zebra_crossing"])
        }
        self.zebra_min_conf = float(zebra.get("min_confidence", 0.35))
        self.zebra_min_detections = int(zebra.get("min_detections", 2))
        self.zebra_match_iou = float(zebra.get("match_iou", 0.2))
        self._zebra_hold_cfg = zebra.get("hold_frames")

        person = self.config.get("person") or {}
        self.foot_band_ratio = float(person.get("foot_band_ratio", 0.2))
        self.min_foot_overlap = float(person.get("min_foot_overlap", 0.5))
        self.on_vehicle_margin = float(person.get("on_vehicle_bottom_margin", 0.1))

        motion = self.config.get("vehicle_motion") or {}
        self._motion_window_cfg = motion.get("window_frames")
        self.vehicle_min_px = float(motion.get("min_displacement_px", 6.0))
        self.vehicle_min_ratio = float(motion.get("min_displacement_ratio", 0.12))
        self.vehicle_max_dist = float(motion.get("max_distance_to_crossing", 1.5))
        self.ego_compensation = bool(motion.get("ego_motion_compensation", True))
        self.occlusion_overlap = float(motion.get("occlusion_overlap", 0.05))
        self.max_shape_change = float(motion.get("max_shape_change", 0.3))
        self.ego_min_detections = int(motion.get("ego_motion_min_detections", 3))
        self.ego_significance = float(motion.get("ego_motion_significance", 2.0))

        temporal = self.config.get("temporal") or {}
        self._temporal_cfg = temporal
        self.require_crossing_motion = bool(temporal.get("require_crossing_motion", False))
        self.crossing_progress_ratio = float(temporal.get("crossing_progress_ratio", 0.25))

        self.engine = None
        self._risk: Dict[str, Any] = {}
        self._built_for_fps: Optional[float] = None
        self.zebras: Optional[ZebraCrossingTracker] = None
        self.machine: Optional[CrossingStateMachine] = None
        self.vehicle_hist = TrackHistory()
        self.person_hist = TrackHistory()

        self.counters: Dict[str, int] = {
            "persons_seen": 0,
            "person_frames_on_crossing": 0,
            "episodes_started": 0,
            "events_confirmed": 0,
            "episodes_closed": 0,
            "episodes_resumed": 0,
            "id_switch_suppressed": 0,
        }
        self.closed_episodes: List[Dict[str, Any]] = []

    # -- lifecycle ------------------------------------------------------

    def setup(self) -> None:
        try:
            RiskEngine = load_vendor_attr(RISK_ENGINE_SRC, "RiskEngine")
        except VendorImportError as exc:
            raise ModuleUnavailable(
                f"Person 3 risk_engine.py could not be imported: {exc}"
            ) from exc
        self._risk = self._risk_cfg()
        self.engine = RiskEngine({"risk": self._risk})
        if self.config.get("zones"):
            self.log.info(
                "modules.pedestrian_safety.zones is set but no longer used: a "
                "fixed image rectangle is not a zebra crossing. Crossings come "
                "only from the ROAD_DEFECT_MODEL's zebra_crossing detections."
            )

    def _risk_cfg(self) -> Dict[str, Any]:
        defaults = {
            "thresholds": {"low_max": 20, "medium_max": 45, "high_max": 70},
            "child_multiplier": 1.20,
            "crossing_bonus": 15,
            "road_bonus": 10,
            "approach_bonus": 25,
            "close_bonus": 25,
            "close_distance_px": 70,
            "dangerous_distance_px": 40,
            "approach_threshold_px_per_frame": 0.8,
            "vehicle_speed_px_per_frame": 1.0,
            "max_pair_distance_px": 260,
        }
        defaults.update(self.config.get("risk") or {})
        return defaults

    def _build(self, fps: float) -> None:
        """Temporal thresholds scale with the real video fps (same approach
        as Module 4), unless set explicitly in the config."""
        fps = float(fps or 30.0)
        t = self._temporal_cfg
        stride = int(self.platform_config.get("perception.road_defect.frame_stride", 5) or 5)

        def frames(key: str, default: int) -> int:
            value = t.get(key)
            return int(value) if value is not None else int(default)

        self.confirm_frames = frames("confirm_frames", max(5, round(fps / 3)))
        self.exit_frames = frames("exit_frames", max(5, round(fps / 3)))
        self.track_lost_frames = frames("track_lost_frames", max(10, round(fps)))
        self.reentry_cooldown_frames = frames(
            "reentry_cooldown_frames", max(self.exit_frames, round(fps))
        )
        self.motion_window = int(
            self._motion_window_cfg
            if self._motion_window_cfg is not None else max(5, round(fps / 2))
        )
        zebra_hold = int(
            self._zebra_hold_cfg
            if self._zebra_hold_cfg is not None else max(2 * stride + 1, round(fps / 2))
        )
        self.zebras = ZebraCrossingTracker(
            min_detections=self.zebra_min_detections,
            hold_frames=zebra_hold,
            match_iou=self.zebra_match_iou,
            max_velocity_gap_frames=max(zebra_hold, 3 * stride),
        )
        self.machine = CrossingStateMachine(
            confirm_frames=self.confirm_frames,
            exit_frames=self.exit_frames,
            track_lost_frames=self.track_lost_frames,
            reentry_cooldown_frames=self.reentry_cooldown_frames,
        )
        history_len = max(90, self.motion_window * 3)
        self.vehicle_hist = TrackHistory(maxlen=history_len)
        self.person_hist = TrackHistory(maxlen=history_len)
        self._built_for_fps = fps
        self.log.info(
            "vulnerable-pedestrian rule @ %.1f fps: confirm=%s exit=%s lost=%s "
            "reentry=%s motion_window=%s zebra_hold=%s zebra_min_detections=%s",
            fps, self.confirm_frames, self.exit_frames, self.track_lost_frames,
            self.reentry_cooldown_frames, self.motion_window, zebra_hold,
            self.zebra_min_detections,
        )

    # -- per frame --------------------------------------------------------

    def process(self, ctx: FrameContext) -> List[Event]:
        if self.machine is None:
            self._build(ctx.fps)
        f = ctx.frame_number

        # 1. zebra crossings: only frames where the road model actually ran
        road = ctx.shared.get("road_defects")
        if road is not None:
            self.zebras.update(
                [
                    d for d in road
                    if str(d.get("class_name", "")).lower() in self.zebra_classes
                    and float(d.get("confidence", 0.0)) >= self.zebra_min_conf
                ],
                f,
            )
        active_zebras = self.zebras.active(f)
        max_gap = self.zebras.max_velocity_gap_frames

        # 2. vehicle positions (tracked only: movement needs an identity).
        #    A sample taken while a person who is not riding the vehicle
        #    covers part of it is unreliable: people walking in front of a
        #    waiting car shift its box centre without the car moving, which
        #    on a zebra crossing is the normal case, not an edge case.
        vehicles = [v for v in (ctx.vehicles or []) if v.get("bbox") is not None]
        persons = [p for p in (ctx.shared.get("persons") or []) if p.get("bbox") is not None]
        for v in vehicles:
            if v.get("track_id") is not None:
                occluded = self.occlusion_overlap > 0 and any(
                    vehicle_occluded_by_person(
                        v["bbox"], p["bbox"], self.occlusion_overlap, self.on_vehicle_margin
                    )
                    for p in persons
                )
                self.vehicle_hist.update(v["track_id"], f, v["bbox"], reliable=not occluded)
                if self.engine is not None:
                    cx, cy = bbox_center(v["bbox"])
                    self.engine.update_position(f"V{v['track_id']}", cx, cy, f)

        # 3. persons
        present_ids = []
        tagged: List[Dict[str, Any]] = []
        confirmed_now: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

        for p in persons:
            self.counters["persons_seen"] += 1
            info = self._evaluate_person(p, vehicles, active_zebras, f, max_gap)
            tid = p.get("track_id")
            if tid is not None:
                present_ids.append(tid)
                self.person_hist.update(tid, f, p["bbox"])
                if self.engine is not None:
                    fx, fy = bbox_bottom_center(p["bbox"])
                    self.engine.update_position(f"P{tid}", fx, fy, f)

                transition = self.machine.observe(
                    tid, f, info["on_zebra_crossing"], info["qualifies"]
                )
                state = self.machine.get(tid)
                self._on_transition(transition, state, info, f)
                if info["on_zebra_crossing"]:
                    self.counters["person_frames_on_crossing"] += 1
                    state.person_conf.append(float(p.get("confidence", 0.0)))
                    self._update_progress(state, p, info)
                    state.last_point = bbox_bottom_center(p["bbox"])
                    state.last_height = float(p["bbox"][3] - p["bbox"][1])

                if self.machine.can_confirm(tid) and (
                    not self.require_crossing_motion
                    or state.progress_px >= self.crossing_progress_ratio * max(1.0, state.last_height)
                ):
                    inherited = self._id_switch_source(tid, state, info, f, present_ids, persons)
                    if inherited is not None:
                        self.machine.confirm(tid, f, inherited.event_id)
                        self.counters["id_switch_suppressed"] += 1
                        self.log.info(
                            "person #%s on crossing continues the confirmed crossing of "
                            "lost track #%s (tracker id switch) -- no new event",
                            tid, inherited.track_id,
                        )
                    else:
                        confirmed_now.append((p, info))

                info["crossing_phase"] = state.phase
                info["crossing_confirmed"] = state.phase == CrossingPhase.CONFIRMED
                info["event_id"] = state.event_id
            else:
                info["crossing_phase"] = CrossingPhase.IDLE
                info["crossing_confirmed"] = False
                info["event_id"] = None
                info["reason"] = info.get("reason") or "untracked person"
            tagged.append(info)

        # 4. emit ONE event per newly confirmed crossing
        events: List[Event] = []
        concurrent = [
            t["track_id"] for t in tagged if t.get("on_zebra_crossing") and t.get("track_id") is not None
        ]
        for p, info in confirmed_now:
            event = self._build_event(ctx, p, info, concurrent)
            self.machine.confirm(p["track_id"], f, event.event_id)
            state = self.machine.get(p["track_id"])
            info["crossing_phase"] = state.phase
            info["crossing_confirmed"] = True
            info["event_id"] = event.event_id
            self.counters["events_confirmed"] += 1
            events.append(event)

        # 5. age tracks that were not detected this frame
        for tid in self.machine.mark_missing(f, present_ids):
            self._record_close(self.machine.get(tid), f, "track lost")

        # 6. roles -- PERSON != PEDESTRIAN
        for info in tagged:
            if info["crossing_confirmed"] and info["qualifies"]:
                info["role"] = ROLE_PEDESTRIAN
            elif info["on_zebra_crossing"]:
                info["role"] = ROLE_CROSSING
            else:
                info["role"] = ROLE_PERSON

        ctx.shared["persons"] = tagged
        ctx.shared["pedestrians"] = [t for t in tagged if t["role"] == ROLE_PEDESTRIAN]
        ctx.shared["zebra_crossings"] = [z.as_dict(f, max_gap) for z in active_zebras]

        # 7. housekeeping
        self.zebras.prune(f)
        horizon = max(self.track_lost_frames, self.reentry_cooldown_frames) * 3
        dead = self.vehicle_hist.prune(f, horizon)
        dead_p = self.person_hist.prune(f, horizon)
        self.machine.prune(f, horizon)
        if self.engine is not None:
            for tid in dead:
                self.engine.hist.pop(f"V{tid}", None)
            for tid in dead_p:
                self.engine.hist.pop(f"P{tid}", None)
        return events

    # -- condition evaluation ----------------------------------------------

    def _evaluate_person(self, p, vehicles, active_zebras, frame_number, max_gap) -> Dict[str, Any]:
        box = p["bbox"]
        info: Dict[str, Any] = {
            **p,
            "zebra_crossing_detected": bool(active_zebras),
            "on_zebra_crossing": False,
            "zebra_crossing_id": None,
            "zebra_crossing_bbox": None,
            "zebra_crossing_confidence": None,
            "foot_overlap": 0.0,
            "on_vehicle": False,
            "vehicles_present": bool(vehicles),
            "vehicle_moving": False,
            "moving_vehicles": [],
            "ego_motion_compensated": False,
            "qualifies": False,
            "reason": None,
        }

        if not active_zebras:
            info["reason"] = "no zebra crossing detected"
            return info

        best = None
        for zebra in active_zebras:
            zbox = zebra.predicted_bbox(frame_number, max_gap)
            on, overlap = person_on_crossing(box, zbox, self.foot_band_ratio, self.min_foot_overlap)
            if overlap > info["foot_overlap"]:
                info["foot_overlap"] = round(overlap, 3)
            if on and (best is None or overlap > best[2]):
                best = (zebra, zbox, overlap)
        if best is None:
            info["reason"] = "not on the zebra crossing"
            return info

        carrier = next(
            (v for v in vehicles if person_on_vehicle(box, v["bbox"], self.on_vehicle_margin)),
            None,
        )
        if carrier is not None:
            info["on_vehicle"] = True
            info["on_vehicle_track_id"] = carrier.get("track_id")
            info["reason"] = "person is on/in a vehicle"
            return info

        zebra, zbox, overlap = best
        info.update({
            "on_zebra_crossing": True,
            "zebra_crossing_id": zebra.crossing_id,
            "zebra_crossing_bbox": [int(round(v)) for v in zbox],
            "zebra_crossing_confidence": round(zebra.confidence, 3),
        })

        if not vehicles:
            info["reason"] = "no vehicles present"
            return info

        ego = (0.0, 0.0)
        ego_err = 0.0
        if self.ego_compensation:
            estimate = zebra.velocity_estimate(
                max_gap, self.ego_min_detections, self.ego_significance
            )
            if estimate is None:
                # On a bus camera a parked car and a moving car look the same
                # until the camera's own motion is known. Do not guess.
                info["reason"] = "camera motion not yet estimated from the crossing"
                return info
            ego = (estimate[0], estimate[1])
            ego_err = (estimate[2] ** 2 + estimate[3] ** 2) ** 0.5
            info["ego_motion_compensated"] = True
            info["ego_motion_px_per_frame"] = [round(ego[0], 3), round(ego[1], 3)]
        reach = self.vehicle_max_dist * bbox_diag(zbox) if self.vehicle_max_dist > 0 else None
        moving = []
        for v in vehicles:
            vid = v.get("track_id")
            if vid is None:
                continue
            if reach is not None and point_to_bbox_distance(bbox_center(v["bbox"]), zbox) > reach:
                continue
            disp = self.vehicle_hist.displacement(
                vid, frame_number, self.motion_window, ego,
                max_shape_change=self.max_shape_change if self.max_shape_change > 0 else None,
            )
            margin = self.ego_significance * ego_err * (disp["span_frames"] if disp else 0.0)
            if is_moving(disp, self.vehicle_min_px, self.vehicle_min_ratio, margin):
                moving.append({
                    "track_id": vid,
                    "class_name": v.get("class_name"),
                    "bbox": list(v["bbox"]),
                    "confidence": round(float(v.get("confidence", 0.0)), 3),
                    "displacement_px": round(disp["magnitude"], 2),
                    "px_per_frame": round(disp["px_per_frame"], 3),
                    "speed_px_s": round(float(v.get("speed", 0.0) or 0.0), 2),
                })
        info["moving_vehicles"] = moving
        info["vehicle_moving"] = bool(moving)
        if not moving:
            info["reason"] = "vehicles present but not moving"
            return info

        info["qualifies"] = True
        return info

    def _update_progress(self, state, p, info) -> None:
        """How far the person has moved ALONG the crossing since the episode
        started, measured in crossing-relative coordinates so the bus's own
        motion cancels out."""
        zbox = info.get("zebra_crossing_bbox")
        if not zbox:
            return
        horizontal = (zbox[2] - zbox[0]) >= (zbox[3] - zbox[1])
        fx, fy = bbox_bottom_center(p["bbox"])
        if horizontal:
            length = max(1.0, zbox[2] - zbox[0])
            rel = (fx - zbox[0]) / length
        else:
            length = max(1.0, zbox[3] - zbox[1])
            rel = (fy - zbox[1]) / length
        if state.start_rel is None or state.crossing_id != info.get("zebra_crossing_id"):
            state.start_rel = rel
            state.crossing_id = info.get("zebra_crossing_id")
        state.progress_px = abs(rel - state.start_rel) * length

    def _on_transition(self, transition, state, info, f) -> None:
        if transition == Transition.STARTED:
            self.counters["episodes_started"] += 1
        elif transition == Transition.RESUMED:
            self.counters["episodes_resumed"] += 1
        elif transition == Transition.CLOSED:
            self._record_close(state, f, "left the crossing")

    def _record_close(self, state, f, why: str) -> None:
        self.counters["episodes_closed"] += 1
        record = {
            "person_track_id": state.track_id,
            "confirmed": state.last_episode_confirmed,
            "event_id": state.event_id,
            "start_frame": state.episode_start_frame,
            "confirmed_frame": state.confirmed_frame,
            "closed_frame": f,
            "reason": why,
        }
        self.closed_episodes.append(record)
        del self.closed_episodes[:-200]
        if state.last_episode_confirmed:
            self.log.info(
                "vulnerable-pedestrian crossing closed: person #%s (%s, frames %s-%s)",
                state.track_id, why, state.episode_start_frame, f,
            )

    def _id_switch_source(self, tid, state, info, f, present_ids, persons):
        """If a CONFIRMED crossing on the same crossing was just lost (track
        not detected now) at practically the same spot, this new id is the
        same human after a tracker id switch: inherit, do not re-alert."""
        if state.last_point is None:
            return None
        present = set(present_ids) | {p.get("track_id") for p in persons}
        radius = max(20.0, 0.75 * max(1.0, state.last_height))
        for other in self.machine.states.values():
            if other.track_id == tid or other.track_id in present:
                continue
            was_confirmed = other.phase == CrossingPhase.CONFIRMED or (
                other.last_episode_confirmed
                and other.closed_frame is not None
                and f - other.closed_frame < self.reentry_cooldown_frames
            )
            if not was_confirmed or other.last_point is None:
                continue
            if f - other.last_seen_frame > self.track_lost_frames + self.reentry_cooldown_frames:
                continue
            if other.crossing_id is not None and other.crossing_id != info.get("zebra_crossing_id"):
                continue
            dx = other.last_point[0] - state.last_point[0]
            dy = other.last_point[1] - state.last_point[1]
            if (dx * dx + dy * dy) ** 0.5 <= radius:
                return other
        return None

    # -- event --------------------------------------------------------------

    def _build_event(self, ctx: FrameContext, p, info, concurrent) -> Event:
        tid = p["track_id"]
        state = self.machine.get(tid)
        moving = info["moving_vehicles"]
        foot = bbox_bottom_center(p["bbox"])

        nearest = min(
            moving, key=lambda v: point_to_bbox_distance(foot, v["bbox"])
        )
        distance = point_to_bbox_distance(foot, nearest["bbox"])
        closing = self.engine.closing_speed(f"P{tid}", f"V{nearest['track_id']}") if self.engine else 0.0
        approaching = closing >= float(self._risk["approach_threshold_px_per_frame"])
        class_name = str(p.get("class_name") or "person").lower()
        score, level = (0.0, "LOW")
        if self.engine is not None:
            score, level = self.engine.score(
                distance=distance,
                approaching=approaching,
                in_road=True,
                crossing=True,
                pedestrian_type=class_name,
                vehicle_speed=nearest["px_per_frame"],
            )
        severity = Severity.max(self.severity_floor, SEVERITY_BY_LEVEL.get(level, Severity.LOW))

        is_child = class_name in CHILD_CLASSES
        person_conf = (
            sum(state.person_conf) / len(state.person_conf)
            if state.person_conf else float(p.get("confidence", 0.0))
        )
        zebra_conf = float(info.get("zebra_crossing_confidence") or 0.0)
        vehicle_conf = max(v["confidence"] for v in moving)
        confidence = min(person_conf, zebra_conf, vehicle_conf)
        progress_needed = self.crossing_progress_ratio * max(1.0, state.last_height)

        event = make_event(
            EventType.PEDESTRIAN_RISK,
            SourceModule.PEDESTRIAN,
            confidence=round(confidence, 3),
            severity=severity,
            track_id=tid,
            class_name=class_name,
            bounding_box=[int(v) for v in p["bbox"]],
            metadata={
                "event_label": EVENT_LABEL,
                "pedestrian_category": "child pedestrian" if is_child else "vulnerable pedestrian",
                "age_group": "child" if is_child else None,
                "age_classification": (
                    "detector class 'child'" if is_child
                    else "not available (detector only provides 'person')"
                ),
                "person_track_id": tid,
                "vehicle_track_ids": [v["track_id"] for v in moving],
                "moving_vehicles": moving,
                "zebra_crossing_detected": True,
                "zebra_crossing_id": info.get("zebra_crossing_id"),
                "zebra_crossing_bbox": info.get("zebra_crossing_bbox"),
                "zebra_crossing_confidence": info.get("zebra_crossing_confidence"),
                "zebra_crossing_source": "ROAD_DEFECT_MODEL:zebra_crossing",
                "person_on_zebra_crossing": True,
                "foot_overlap": info.get("foot_overlap"),
                "person_on_vehicle": False,
                "vehicle_moving": True,
                "ego_motion_compensated": info.get("ego_motion_compensated", False),
                "crossing_confirmed": True,
                "crossing_motion": (
                    "moving_across" if state.progress_px >= progress_needed
                    else "occupying_crossing"
                ),
                "crossing_progress_px": round(state.progress_px, 1),
                "candidate_start_frame": state.episode_start_frame,
                "confirmed_frame": ctx.frame_number,
                "confirmation_frames": self.confirm_frames,
                "concurrent_person_track_ids_on_crossing": concurrent,
                "confidence_components": {
                    "person_mean": round(person_conf, 3),
                    "zebra_crossing": round(zebra_conf, 3),
                    "moving_vehicle_max": round(vehicle_conf, 3),
                    "rule": "min of the three (weakest link)",
                },
                # Person 3's scoring, kept for severity
                "vehicle_track_id": nearest["track_id"],
                "vehicle_class_name": nearest.get("class_name"),
                "vehicle_bounding_box": nearest.get("bbox"),
                "distance_px": round(distance, 2),
                "closing_speed_px_per_frame": round(closing, 3),
                "approaching": approaching,
                "in_road": True,
                "crossing": True,
                "risk_score": score,
                "risk_level": level,
                "role": ROLE_PEDESTRIAN,
            },
        )

        if self.save_evidence and self.evidence is not None and ctx.frame is not None:
            event.evidence_image = self._save_evidence(ctx, p, info, moving)
        event.with_context(ctx)
        self.log.info(
            "VULNERABLE PEDESTRIAN EVENT: person #%s on zebra crossing %s, moving "
            "vehicles %s, frames %s-%s, severity %s",
            tid, info.get("zebra_crossing_id"), [v["track_id"] for v in moving],
            state.episode_start_frame, ctx.frame_number, severity,
        )
        return event

    def _save_evidence(self, ctx, p, info, moving) -> Optional[str]:
        try:
            import cv2                                          # noqa: PLC0415
        except Exception:                                       # noqa: BLE001
            crop = self.evidence.crop(ctx.frame, p["bbox"])
            return self.evidence.save_image(crop, "vulnerable_pedestrian", ctx.bus_id, ctx.frame_number)
        image = ctx.frame.copy()
        zbox = info.get("zebra_crossing_bbox")
        if zbox:
            cv2.rectangle(image, (zbox[0], zbox[1]), (zbox[2], zbox[3]), (255, 255, 255), 2)
            cv2.putText(image, "ZEBRA CROSSING", (zbox[0] + 4, max(14, zbox[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for v in moving:
            x1, y1, x2, y2 = [int(c) for c in v["bbox"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(image, f"MOVING {v.get('class_name')} #{v['track_id']}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
        x1, y1, x2, y2 = [int(c) for c in p["bbox"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(image, f"PEDESTRIAN #{p['track_id']}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        return self.evidence.save_image(image, "vulnerable_pedestrian", ctx.bus_id, ctx.frame_number)

    # -- end of video / reporting --------------------------------------------

    def finalize(self) -> List[Event]:
        if self.machine is not None:
            for state in self.machine.states.values():
                if state.phase != CrossingPhase.IDLE:
                    self.machine._close(state, state.last_seen_frame)   # noqa: SLF001
                    self._record_close(state, state.last_seen_frame, "end of video")
        return []

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        data["crossing"] = dict(self.counters)
        return data
