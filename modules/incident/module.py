"""
MODULE 4 -- INCIDENT DETECTION AND TRACKING     (owner: Person 4)
=================================================================

Collision -> hit-and-run -> ANPR, built on **Person 4's own detectors**,
loaded unmodified from ``vendor/person4_incident/``:

    motion_analysis.py     -> MotionAnalyzer        (speed, accel, lateral)
    rash_driving.py        -> RashDrivingDetector   (rule-based, 2+ signals)
    collision_detection.py -> CollisionDetector     (multi-signal, hardened)
    hit_and_run.py         -> HitAndRunDetector     (escape monitoring)

No new ML model.  Everything here is reasoning over the vehicle boxes,
BoT-SORT track ids and speeds that ``modules.perception.vehicle`` already
produced with Person 2's detector.

--------------------------------------------------------------------------
WHY THE PREVIOUS HIT-AND-RUN IMPLEMENTATION WAS REPLACED
--------------------------------------------------------------------------
The old logic emitted 2 hit-and-runs on 266 frames of ordinary dashcam
traffic containing no crash.  Four concrete defects, all fixed by the code
below:

1. **Disappearance was treated as escape.**  ``gone = track not in frame``
   after ~10 frames *raised* the confidence and still emitted.  An id switch
   or a vehicle leaving the camera view was therefore a hit-and-run.
   -> Person 4's detector confirms escape only on a sustained run of frames
      moving away from the incident point.  Disappearance alone never
      confirms; it resolves to ``TRACK_LOST`` and emits nothing.

2. **"Did not stop" was used as "fled".**  The only test was
   ``min_speed_after < 4 px/s``.  Every moving vehicle passes it.
   -> Escape is now measured as displacement from the recorded incident
      point, scaled by the vehicle's own box size, over consecutive frames.

3. **Overlap alone started the watch.**  Any two boxes overlapping with a
   speed wobble counted as a corroborated collision.
   -> Confirmation now requires a *transient* overlap episode that was
      preceded by a separated state, plus approach or impact-like motion
      change, plus a near-field occluder guard.

4. **The flee threshold was inert.**  ``flee_speed_ratio`` was only ever
   used as a fallback value, never compared against anything.
   -> Every threshold below is read from config and actually applied.

--------------------------------------------------------------------------
THE STATE MACHINE (spec 9)
--------------------------------------------------------------------------
    NORMAL
      -> SUSPICIOUS_INTERACTION     (loose detector flagged the pair)
      -> POSSIBLE_COLLISION         (loose confirmation)
      -> POST_COLLISION_MONITORING  (strict confirmation; track locked)
      -> VEHICLE_LEAVING            (consistent movement away)
      -> HIT_AND_RUN_CONFIRMED      (Person 4's detector confirmed)

    Two terminal states emit no hit-and-run:
      COLLISION_ONLY   the vehicle stayed near the incident point
      TRACK_LOST       the track vanished without a departure run
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from core.event import (
    Event,
    EventType,
    HitAndRunState,
    IncidentStatus,
    Severity,
    SourceModule,
    make_event,
)
from core.frame_context import FrameContext
from core.module_base import BaseModule, ModuleUnavailable
from core.vendor_loader import VendorImportError, load_vendor_attr

VENDOR = "person4_incident/"
VENDOR_ROOT = "person4_incident"


# ---------------------------------------------------------------------------
# Shim: platform vehicle dicts -> the TrackedObject shape Person 4 expects
# ---------------------------------------------------------------------------

class _TrackedObject:
    """Person 4's detectors read attributes off a tracked object.  The
    platform carries vehicles as dicts, so this adapts one to the other
    without either side changing."""

    __slots__ = ("track_id", "bbox", "object_type", "confidence",
                 "frame_id", "timestamp")

    def __init__(self, vehicle: Dict[str, Any], frame_id: int, timestamp: str):
        self.track_id = int(vehicle["track_id"])
        bbox = vehicle["bbox"]
        self.bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        self.object_type = vehicle.get("class_name") or "car"
        self.confidence = float(vehicle.get("confidence") or 0.0)
        self.frame_id = int(frame_id)
        self.timestamp = timestamp


# ---------------------------------------------------------------------------
# The explicit hit-and-run state machine
# ---------------------------------------------------------------------------

class _VehicleCase:
    """One vehicle's journey through the state machine after an incident."""

    def __init__(self, track_id: int, vehicle_class: Optional[str]):
        self.track_id = track_id
        self.vehicle_class = vehicle_class
        self.state = HitAndRunState.NORMAL
        self.timeline: List[Dict[str, Any]] = []
        self.incident_point: Optional[tuple] = None
        self.incident_frame: Optional[int] = None
        self.incident_gps: Optional[Dict[str, Any]] = None
        self.ref_scale: float = 1.0
        self.collision_event_id: Optional[str] = None
        self.max_distance: float = 0.0
        self.last_distance: Optional[float] = None
        self.away_run: int = 0
        self.missing_frames: int = 0
        self.monitored_frames: int = 0
        self.evidence: Dict[str, Any] = {}
        self.synthetic = False
        self.closed = False

    def transition(self, state: str, frame_number: int, video_time_s: float,
                   note: str = "") -> bool:
        if state == self.state:
            return False
        self.state = state
        self.timeline.append({
            "state": state,
            "frame": frame_number,
            "video_time_s": round(float(video_time_s or 0.0), 2),
            "note": note,
        })
        return True


class IncidentModule(BaseModule):
    name = "incident"
    owner = "Person 4"

    def __init__(self, config, evidence=None):
        super().__init__(config.section("modules.incident"))
        self.platform_config = config
        self.evidence_store = evidence

        self.rash_cfg = self.config.get("rash", {}) or {}
        self.collision_cfg = self.config.get("collision", {}) or {}
        self.hit_run_cfg = self.config.get("hit_and_run", {}) or {}
        self.evidence_cfg = self.config.get("evidence", {}) or {}

        self.min_track_age = int(self.config.get("minimum_track_age", 5))
        self.anpr_trigger_confidence = float(
            self.config.get("anpr_trigger_confidence", 0.55)
        )

        # vendor classes, resolved in setup()
        self._MotionAnalyzer = None
        self._RashDrivingDetector = None
        self._CollisionDetector = None
        self._HitAndRunDetector = None

        # instances, built lazily on the first frame (they need the real fps)
        self.motion = None
        self.rash = None
        self.collision_strict = None
        self.collision_loose = None
        self.hit_and_run = None
        self._built = False

        self._track_age: Dict[int, int] = {}
        self._cases: Dict[int, _VehicleCase] = {}
        self._rash_cooldown: Dict[int, int] = {}
        self._frame_ring: Deque[Dict[str, Any]] = deque(
            maxlen=int(self.evidence_cfg.get("pre_frames", 10))
        )
        self._after_jobs: List[Dict[str, Any]] = []
        self._incident_seq = 0
        self._last_ctx: Optional[FrameContext] = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        try:
            self._MotionAnalyzer = load_vendor_attr(
                VENDOR + "motion_analysis.py", "MotionAnalyzer"
            )
            self._RashDrivingDetector = load_vendor_attr(
                VENDOR + "rash_driving.py", "RashDrivingDetector"
            )
            # collision_detection.py does `from ai.vehicle.classes import ...`,
            # so Person 4's own project root goes on sys.path for the import.
            from core.vendor_loader import load_vendor_module
            module = load_vendor_module(
                VENDOR + "collision_detection.py", sys_path=VENDOR_ROOT
            )
            self._CollisionDetector = getattr(module, "CollisionDetector")
            self._HitAndRunDetector = load_vendor_attr(
                VENDOR + "hit_and_run.py", "HitAndRunDetector"
            )
        except VendorImportError as exc:
            raise ModuleUnavailable(
                f"Person 4's incident detectors could not be loaded: {exc}"
            ) from exc

        self.log.info(
            "Person 4 detectors loaded (rash=%s, collision=%s, hit_and_run=%s) "
            "-- no model of its own, reasoning over Person 2's tracks",
            self.rash_cfg.get("enabled", True),
            self.collision_cfg.get("enabled", True),
            self.hit_run_cfg.get("enabled", True),
        )

    def _build(self, fps: float) -> None:
        """Instantiate Person 4's detectors once the real fps is known."""
        fps = float(fps or 30.0)
        cfg = self.collision_cfg
        loose = (cfg.get("possible") or {})

        self.motion = self._MotionAnalyzer(
            max_history=int(self.config.get("history_frames", 45)), fps=fps
        )
        # Temporal-persistence defaults, scaled from the real fps rather
        # than a fixed frame count: require roughly a third of a second
        # of sustained abnormal readings (never fewer than 5 frames)
        # before a single-frame spike is even considered, and look back
        # over about a second while checking for that persistence.
        default_persist_frames = max(5, round(fps / 3))
        self.rash = self._RashDrivingDetector(
            speed_threshold=float(self.rash_cfg.get("speed_threshold_px_s", 250.0)),
            acceleration_threshold=float(
                self.rash_cfg.get("acceleration_threshold", 150.0)
            ),
            lateral_threshold=float(self.rash_cfg.get("lateral_threshold", 80.0)),
            min_persist_frames=int(
                self.rash_cfg.get("min_persist_frames", default_persist_frames)
            ),
            window_frames=int(
                self.rash_cfg.get("window_frames", max(1, round(fps)))
            ),
            min_confidence=float(self.rash_cfg.get("min_confidence", 0.5)),
        )

        strict_kwargs = dict(
            fps=fps,
            history_frames=int(cfg.get("history_frames", 30)),
            confirm_peak=float(cfg.get("confirm_peak", 0.62)),
            confirm_sustained=float(cfg.get("confirm_sustained", 0.50)),
            min_persist_frames=int(cfg.get("min_persist_frames", 2)),
            confirm_min_iou=float(cfg.get("contact_min_iou", 0.08)),
            overlap_episode_min=float(cfg.get("overlap_episode_min", 0.06)),
            max_episode_frames=int(cfg.get("max_episode_frames", 30)),
            separated_threshold=float(cfg.get("separated_threshold", 0.06)),
            max_vehicle_fraction=float(cfg.get("max_vehicle_fraction", 0.42)),
            approach_min=float(cfg.get("approach_min", 0.35)),
            motion_min=float(cfg.get("motion_min", 0.30)),
            cooldown_frames=int(cfg.get("cooldown_frames", 45)),
        )
        self.collision_strict = self._CollisionDetector(**strict_kwargs)

        # The "possible collision" tier is the SAME algorithm with a lower
        # confirmation bar -- every false-positive guard (transient episode,
        # prior separation, occluder size, approach/motion) still applies.
        loose_kwargs = dict(strict_kwargs)
        loose_kwargs.update(
            confirm_peak=float(loose.get("confirm_peak", 0.45)),
            confirm_sustained=float(loose.get("confirm_sustained", 0.36)),
            confirm_min_iou=float(loose.get("contact_min_iou", 0.05)),
            approach_min=float(loose.get("approach_min", 0.25)),
            motion_min=float(loose.get("motion_min", 0.20)),
        )
        self.collision_loose = self._CollisionDetector(**loose_kwargs)

        hr = self.hit_run_cfg
        confirmation_s = float(hr.get("hit_and_run_confirmation_time", 0.4))
        self.hit_and_run = self._HitAndRunDetector(
            fps=fps,
            min_escape_frames=max(
                int(hr.get("min_escape_frames", 12)),
                int(round(confirmation_s * fps)),
            ),
            min_escape_distance_px=float(
                hr.get("departure_distance_threshold", 70.0)
            ),
            escape_ratio=float(hr.get("escape_ratio", 1.5)),
            escape_progress=float(hr.get("escape_progress", 0.5)),
            require_consecutive=int(hr.get("require_consecutive", 3)),
            max_missing_frames=int(hr.get("track_loss_timeout", 30)),
            cooldown_frames=int(hr.get("cooldown_frames", 90)),
        )
        self._built = True
        self.log.info(
            "detectors built for %.1f fps "
            "(escape >= %s px over >= %s frames, track-loss tolerance %s frames)",
            fps,
            self.hit_and_run.min_escape_distance_px,
            self.hit_and_run.min_escape_frames,
            self.hit_and_run.max_missing_frames,
        )

    # ------------------------------------------------------------------
    # per frame
    # ------------------------------------------------------------------

    def process(self, ctx: FrameContext) -> List[Event]:
        if not self._built:
            self._build(ctx.fps)
        self._last_ctx = ctx

        vehicles = ctx.vehicles or []
        for vehicle in vehicles:
            track_id = vehicle.get("track_id")
            if track_id is not None:
                self._track_age[track_id] = self._track_age.get(track_id, 0) + 1

        # Track age gate: a box that has only just appeared has no reliable
        # velocity, and every incident signal here is velocity-based.
        objects = [
            _TrackedObject(v, ctx.frame_number, self._timestamp(ctx))
            for v in vehicles
            if v.get("track_id") is not None
            and self._track_age.get(v["track_id"], 0) >= self.min_track_age
            and v.get("bbox")
        ]

        self._remember_frame(ctx)
        self._run_after_jobs(ctx)

        events: List[Event] = []
        frame_size = (ctx.width or 0, ctx.height or 0)

        # ---- 1. rash / abnormal movement ------------------------------
        if self.rash_cfg.get("enabled", True):
            motion_rows = self.motion.update(objects)
            events.extend(self._rash_events(ctx, motion_rows))
        else:
            self.motion.update(objects)

        # ---- 2. collisions --------------------------------------------
        confirmed: List[dict] = []
        if self.collision_cfg.get("enabled", True):
            confirmed = self.collision_strict.update(objects, frame_size=frame_size)
            possible = self.collision_loose.update(objects, frame_size=frame_size)

            confirmed_pairs = {
                self._pair(c["vehicle_track_id"], c["other_track_id"])
                for c in confirmed
            }
            for coll in confirmed:
                events.append(self._collision_event(ctx, coll, confirmed=True))
            for coll in possible:
                pair = self._pair(coll["vehicle_track_id"], coll["other_track_id"])
                if pair in confirmed_pairs:
                    continue          # already reported at the higher tier
                events.append(self._collision_event(ctx, coll, confirmed=False))

        # ---- 2b. optional, clearly-labelled demo trigger ---------------
        injected = self._demo_injection(ctx, objects)
        if injected is not None:
            confirmed = list(confirmed) + [injected["collision"]]
            events.append(injected["event"])

        # ---- 3. hit-and-run -------------------------------------------
        if self.hit_run_cfg.get("enabled", True):
            confirmations = self.hit_and_run.update(objects, confirmed)
            self._advance_states(ctx, objects)
            for hit in confirmations:
                event = self._hit_and_run_event(ctx, hit)
                if event is not None:
                    events.append(event)

        return events

    def finalize(self) -> List[Event]:
        """Close out anything still being monitored when the video ends.
        Nothing is upgraded to a hit-and-run here -- an unfinished
        observation is reported as exactly that."""
        ctx = self._last_ctx
        if ctx is None:
            return []
        for case in self._cases.values():
            if case.closed or case.state in (
                HitAndRunState.HIT_AND_RUN_CONFIRMED,
                HitAndRunState.COLLISION_ONLY,
                HitAndRunState.TRACK_LOST,
            ):
                continue
            case.transition(
                HitAndRunState.COLLISION_ONLY, ctx.frame_number,
                ctx.video_time_s, "video ended before the monitoring window closed",
            )
            case.closed = True
            self.log.info(
                "[HIT_RUN] track %s -> COLLISION_ONLY (video ended, no departure)",
                case.track_id,
            )
        return []

    # ------------------------------------------------------------------
    # rash driving
    # ------------------------------------------------------------------

    def _rash_events(self, ctx: FrameContext, motion_rows) -> List[Event]:
        cooldown = int(
            float(self.rash_cfg.get("cooldown_s", 6.0)) * max(1.0, ctx.fps)
        )
        events: List[Event] = []
        for incident in self.rash.analyze(motion_rows):
            track_id = int(incident["vehicle_track_id"])
            last = self._rash_cooldown.get(track_id)
            if last is not None and ctx.frame_number - last < cooldown:
                continue
            self._rash_cooldown[track_id] = ctx.frame_number

            reasons = incident.get("reasons") or []
            confidence = float(incident.get("confidence") or 0.0)
            # Three corroborating signals is "rash driving"; two is an
            # abnormal manoeuvre and is reported as such, not upgraded.
            if len(reasons) >= 3:
                event_type = EventType.RASH_DRIVING
                severity = Severity.HIGH if confidence >= 0.75 else Severity.MEDIUM
            else:
                event_type = EventType.ABNORMAL_MOVEMENT
                severity = Severity.LOW

            crop = self._vehicle_crop(ctx, track_id)
            evidence_image = self._save(crop, event_type, ctx)

            needs_id = (
                event_type == EventType.RASH_DRIVING
                and bool(self.rash_cfg.get("trigger_anpr_on_high", True))
                and severity == Severity.HIGH
                and confidence >= self.anpr_trigger_confidence
            )
            self.log.info(
                "[INCIDENT] track %s %s (%s) conf=%.2f",
                track_id, event_type, ", ".join(reasons), confidence,
            )
            events.append(make_event(
                event_type,
                SourceModule.INCIDENT,
                confidence=round(confidence, 3),
                severity=severity,
                class_name=incident.get("vehicle_type"),
                track_id=track_id,
                evidence_image=evidence_image,
                metadata={
                    "incident_id": self._incident_id(),
                    "incident_type": event_type,
                    "status": IncidentStatus.SUSPECTED,
                    "vehicle_track_id": track_id,
                    "vehicle_class": incident.get("vehicle_type"),
                    "reasons": reasons,
                    "evidence": ", ".join(reasons),
                    "speed_px_s": incident.get("speed_pixels_per_second"),
                    "acceleration": incident.get("acceleration"),
                    "lateral_movement": incident.get("lateral_movement"),
                    "direction": incident.get("direction"),
                    "detector": "person4.RashDrivingDetector",
                    "requires_identification": needs_id,
                    "anpr_track_id": track_id if needs_id else None,
                },
            ))
        return events

    # ------------------------------------------------------------------
    # collisions
    # ------------------------------------------------------------------

    def _collision_event(self, ctx: FrameContext, coll: dict,
                         confirmed: bool) -> Event:
        track_id = int(coll["vehicle_track_id"])
        other_id = coll.get("other_track_id")
        confidence = float(coll.get("confidence") or 0.0)

        event_type = EventType.COLLISION if confirmed else EventType.POSSIBLE_COLLISION
        severity = Severity.CRITICAL if confirmed else Severity.HIGH
        status = IncidentStatus.CONFIRMED if confirmed else IncidentStatus.SUSPECTED

        case = self._case(track_id, coll.get("vehicle_type"))
        if case.closed:
            # A resolved case is never reopened by a later loose detection --
            # its terminal state is part of the incident record.
            case = None
        if case is not None and confirmed:
            case.transition(
                HitAndRunState.POSSIBLE_COLLISION, ctx.frame_number,
                ctx.video_time_s, "overlap confirmed by approach + impact motion",
            )
        elif case is not None:
            case.transition(
                HitAndRunState.SUSPICIOUS_INTERACTION, ctx.frame_number,
                ctx.video_time_s, "vehicles converged without confirmed contact",
            )

        evidence = self._capture_incident_evidence(ctx, track_id, event_type)
        if confirmed and case is not None:
            case.evidence.update(evidence)
            case.collision_event_id = None      # set below, after the event exists
            self._lock_case(case, ctx, coll)

        # A confirmed collision now also reads the plate -- the offender may
        # never trigger a hit-and-run (e.g. it stays at the scene), so ANPR
        # can no longer wait on that downstream confirmation alone.
        #
        # trigger_anpr_on_possible (default False) additionally requests a
        # plate read on the loose/unconfirmed "possible_collision" tier.
        # This deliberately trades accuracy for recall: it will fire on
        # near-misses and false positives, not just real collisions, so it
        # should only be enabled when getting *some* plate read on a wider
        # net of incidents matters more than avoiding wasted ANPR calls.
        needs_id = (
            (confirmed or bool(self.collision_cfg.get("trigger_anpr_on_possible", False)))
            and bool(self.collision_cfg.get("trigger_anpr", True))
            and confidence >= self.anpr_trigger_confidence
        )

        self.log.info(
            "[INCIDENT] %s track %s <-> %s  iou=%.2f approach=%.2f motion=%.2f conf=%.2f",
            event_type, track_id, other_id, coll.get("iou") or 0.0,
            coll.get("approach") or 0.0, coll.get("motion_change") or 0.0, confidence,
        )
        if needs_id:
            self.log.info("[ANPR] identification requested for track %s (collision)", track_id)

        event = make_event(
            event_type,
            SourceModule.INCIDENT,
            confidence=round(confidence, 3),
            severity=severity,
            class_name=self._pair_class(coll),
            track_id=track_id,
            evidence_image=evidence.get("incident_frame"),
            metadata={
                "incident_id": self._incident_id(),
                "incident_type": event_type,
                "status": status,
                "vehicle_track_ids": [track_id, other_id],
                "vehicle_track_id": track_id,
                "vehicle_class": coll.get("vehicle_type"),
                "other_track_id": other_id,
                "other_type": coll.get("other_type"),
                "iou": coll.get("iou"),
                "distance_px": coll.get("distance"),
                "approach": coll.get("approach"),
                "motion_change": coll.get("motion_change"),
                "impact_frame": coll.get("frame_id"),
                "detector": "person4.CollisionDetector",
                "confirmation_tier": "strict" if confirmed else "loose",
                "evidence": (
                    "transient box overlap preceded by a separated state, "
                    "corroborated by approach and impact-like motion change"
                    if confirmed else
                    "vehicles converged with partial contact evidence -- "
                    "reported as possible, not confirmed"
                ),
                "evidence_frames": evidence,
                # Confirmed collisions request identification directly; the
                # loose/possible tier still leaves it to hit-and-run confirmation.
                "requires_identification": needs_id,
                "anpr_track_id": track_id if needs_id else None,
            },
        )
        if confirmed and case is not None:
            case.collision_event_id = event.event_id
        return event

    def _lock_case(self, case: _VehicleCase, ctx: FrameContext, coll: dict) -> None:
        """Record the incident point and start post-collision monitoring."""
        vehicle = self._vehicle(ctx, case.track_id)
        if vehicle is None:
            return
        x1, y1, x2, y2 = [float(v) for v in vehicle["bbox"]]
        case.incident_point = ((x1 + x2) / 2.0, y2)      # foot point
        case.incident_frame = ctx.frame_number
        case.incident_gps = ctx.gps.as_dict() if ctx.gps else None
        case.ref_scale = max(2.0, x2 - x1, y2 - y1)
        case.max_distance = 0.0
        case.last_distance = 0.0
        case.away_run = 0
        case.missing_frames = 0
        case.monitored_frames = 0
        case.transition(
            HitAndRunState.POST_COLLISION_MONITORING, ctx.frame_number,
            ctx.video_time_s, "track locked, watching for departure",
        )
        self.log.info(
            "[HIT_RUN] track %s entered POST_COLLISION_MONITORING at frame %s",
            case.track_id, ctx.frame_number,
        )
        self._schedule_after_capture(ctx, case)

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------

    def _advance_states(self, ctx: FrameContext, objects) -> None:
        present = {obj.track_id for obj in objects}
        loss_timeout = int(self.hit_run_cfg.get("track_loss_timeout", 30))
        monitor_window = int(self.hit_run_cfg.get("monitor_window_frames", 120))

        # let the annotator draw which vehicles are currently being watched
        ctx.shared["incident_watch"] = {
            case.track_id: case.state
            for case in self._cases.values()
            if not case.closed and case.state in (
                HitAndRunState.POST_COLLISION_MONITORING,
                HitAndRunState.VEHICLE_LEAVING,
            )
        }

        for case in self._cases.values():
            if case.closed or case.incident_point is None:
                continue
            if case.state in (
                HitAndRunState.HIT_AND_RUN_CONFIRMED,
                HitAndRunState.COLLISION_ONLY,
                HitAndRunState.TRACK_LOST,
            ):
                continue

            case.monitored_frames += 1

            if case.track_id not in present:
                case.missing_frames += 1
                if case.missing_frames > loss_timeout:
                    # THE important rule: a vanished track is not an escape.
                    case.transition(
                        HitAndRunState.TRACK_LOST, ctx.frame_number,
                        ctx.video_time_s,
                        f"track absent for {case.missing_frames} frames without "
                        f"a departure run -- NOT reported as hit-and-run",
                    )
                    case.closed = True
                    self.log.info(
                        "[HIT_RUN] track %s -> TRACK_LOST after %s missing frames "
                        "(no hit-and-run: disappearance is not departure)",
                        case.track_id, case.missing_frames,
                    )
                continue

            case.missing_frames = 0
            vehicle = self._vehicle(ctx, case.track_id)
            if vehicle is None:
                continue
            x1, _y1, x2, y2 = [float(v) for v in vehicle["bbox"]]
            foot = ((x1 + x2) / 2.0, y2)
            distance = math.hypot(
                foot[0] - case.incident_point[0],
                foot[1] - case.incident_point[1],
            )
            case.max_distance = max(case.max_distance, distance)
            if case.last_distance is not None and distance > case.last_distance + 1.0:
                case.away_run += 1
            elif case.last_distance is not None and distance < case.last_distance - 1.0:
                case.away_run = max(0, case.away_run - 2)
            case.last_distance = distance

            leaving_after = int(self.hit_run_cfg.get("require_consecutive", 3))
            if (case.away_run >= leaving_after
                    and case.state == HitAndRunState.POST_COLLISION_MONITORING):
                case.transition(
                    HitAndRunState.VEHICLE_LEAVING, ctx.frame_number,
                    ctx.video_time_s,
                    f"moved {distance:.0f} px away from the incident point",
                )
                self.log.info(
                    "[HIT_RUN] track %s moving away from incident point (%.0f px)",
                    case.track_id, distance,
                )

            if case.monitored_frames >= monitor_window:
                case.transition(
                    HitAndRunState.COLLISION_ONLY, ctx.frame_number,
                    ctx.video_time_s,
                    "vehicle stayed near the incident point for the whole window",
                )
                case.closed = True
                self.log.info(
                    "[HIT_RUN] track %s -> COLLISION_ONLY (stayed at the scene)",
                    case.track_id,
                )

    # ------------------------------------------------------------------
    # hit-and-run event
    # ------------------------------------------------------------------

    def _hit_and_run_event(self, ctx: FrameContext, hit: dict) -> Optional[Event]:
        track_id = int(hit["hit_vehicle_track_id"])
        case = self._case(track_id, hit.get("vehicle_type"))
        if case.closed:
            return None

        confidence = float(hit.get("confidence") or 0.0)
        case.transition(
            HitAndRunState.HIT_AND_RUN_CONFIRMED, ctx.frame_number,
            ctx.video_time_s,
            f"departed {hit.get('escape_distance_px')} px at "
            f"{hit.get('escape_speed_px_s')} px/s",
        )
        case.closed = True

        crop = self._vehicle_crop(ctx, track_id)
        vehicle_crop = self._save(crop, "hit_and_run", ctx)
        evidence = dict(case.evidence)
        if vehicle_crop:
            evidence["vehicle_crop"] = vehicle_crop

        needs_id = (
            bool(self.hit_run_cfg.get("trigger_anpr", True))
            and confidence >= self.anpr_trigger_confidence
        )

        self.log.info(
            "[HIT_RUN] track %s departure confirmed -- HIT_AND_RUN "
            "(distance %.0f px, conf %.2f)",
            track_id, float(hit.get("escape_distance_px") or 0.0), confidence,
        )
        if needs_id:
            self.log.info("[ANPR] identification requested for track %s", track_id)

        return make_event(
            EventType.HIT_AND_RUN,
            SourceModule.INCIDENT,
            confidence=round(confidence, 3),
            severity=Severity.CRITICAL,
            class_name=hit.get("vehicle_type"),
            track_id=track_id,
            evidence_image=evidence.get("incident_frame") or vehicle_crop,
            evidence_video=evidence.get("clip"),
            metadata={
                "incident_id": self._incident_id(),
                "incident_type": EventType.HIT_AND_RUN,
                "status": IncidentStatus.SUSPECTED,   # never claimed as certain
                "state": HitAndRunState.HIT_AND_RUN_CONFIRMED,
                "vehicle_track_id": track_id,
                "vehicle_class": hit.get("vehicle_type"),
                "other_track_id": hit.get("other_track_id"),
                "collision_event_id": case.collision_event_id,
                "collision_frame_id": hit.get("collision_frame_id"),
                "incident_location": case.incident_gps,
                "escape_distance_px": hit.get("escape_distance_px"),
                "escape_speed_px_s": hit.get("escape_speed_px_s"),
                "escape_direction": hit.get("escape_direction"),
                "progress": hit.get("progress"),
                "timeline": case.timeline,
                "evidence_frames": evidence,
                "detector": "person4.HitAndRunDetector",
                "synthetic": case.synthetic,
                "evidence": (
                    ("SYNTHETIC demo trigger started this case -- the departure "
                     "measurement below is real. "
                     if case.synthetic else "")
                    + "collision followed by sustained movement away from the "
                      "incident point"
                ),
                "requires_identification": needs_id,
                "anpr_track_id": track_id if needs_id else None,
            },
        )

    # ------------------------------------------------------------------
    # demo trigger  (OFF by default -- see docs/DEMO.md)
    # ------------------------------------------------------------------

    def _demo_injection(self, ctx: FrameContext, objects) -> Optional[Dict[str, Any]]:
        """Inject ONE clearly-labelled synthetic collision so the full
        collision -> hit-and-run -> ANPR chain can be demonstrated on footage
        that contains no real crash.

        What is synthetic: the collision trigger, and nothing else.  The
        vehicle, its track id, its trajectory, the escape measurement, the
        evidence crops, the plate read and the GPS are all real -- which is
        why the vehicle still has to genuinely drive away before a
        hit-and-run is confirmed.  Every event produced here carries
        ``metadata.synthetic = true`` and says so in its evidence string, and
        the dashboard renders it with a SYNTHETIC badge.

        Enable with ``--demo-incident`` or
        ``modules.incident.demo.inject_synthetic_collision: true``.
        """
        demo = self.config.get("demo") or {}
        if not demo.get("inject_synthetic_collision"):
            return None
        if getattr(self, "_demo_done", False):
            return None

        at_frame = int(demo.get("at_frame", 0) or 0)
        if at_frame and ctx.frame_number < at_frame:
            return None

        # Pick a real, well-established, moving vehicle to stand in as the
        # offender -- otherwise the escape monitor has nothing to measure.
        speeds = {
            v.get("track_id"): float(v.get("speed") or 0.0)
            for v in (ctx.vehicles or [])
        }
        candidates = [
            obj for obj in objects
            if self._track_age.get(obj.track_id, 0) >= max(10, self.min_track_age * 2)
        ]
        # A stationary vehicle cannot demonstrate a departure, so prefer one
        # that is actually moving; the escape still has to be earned.
        moving = [
            obj for obj in candidates
            if speeds.get(obj.track_id, 0.0)
            >= float(demo.get("min_offender_speed_px_s", 25.0))
        ]
        pool = moving or candidates
        if not pool:
            return None

        # Prefer a vehicle that will stay in view long enough to be measured:
        # moving, but well inside the frame rather than about to exit it.
        # A track that leaves the camera view resolves to TRACK_LOST -- which
        # is correct behaviour, just useless for a demonstration.
        def headroom(obj) -> float:
            cx = (obj.bbox[0] + obj.bbox[2]) / 2.0
            cy = (obj.bbox[1] + obj.bbox[3]) / 2.0
            w, h = max(1, ctx.width), max(1, ctx.height)
            return min(cx, w - cx) / w + min(cy, h - cy) / h

        offender = max(
            pool,
            key=lambda o: (
                round(headroom(o), 2),
                speeds.get(o.track_id, 0.0),
            ),
        )
        other = next((o for o in objects if o.track_id != offender.track_id), None)

        self._demo_done = True
        collision = {
            "incident_type": "collision",
            "vehicle_track_id": offender.track_id,
            "vehicle_type": offender.object_type,
            "other_track_id": other.track_id if other else None,
            "other_type": other.object_type if other else None,
            "frame_id": ctx.frame_number,
            "timestamp": self._timestamp(ctx),
            "iou": None, "distance": None, "approach": None, "motion_change": None,
            "confidence": 0.0,
            "synthetic": True,
        }

        case = self._case(offender.track_id, offender.object_type)
        case.synthetic = True
        case.transition(
            HitAndRunState.POSSIBLE_COLLISION, ctx.frame_number, ctx.video_time_s,
            "SYNTHETIC demo trigger -- everything after this point is measured "
            "on the real trajectory",
        )
        evidence = self._capture_incident_evidence(ctx, offender.track_id, "collision")
        case.evidence.update(evidence)
        self._lock_case(case, ctx, collision)

        self.log.warning(
            "[DEMO] SYNTHETIC collision injected on real track %s at frame %s -- "
            "escape, ANPR and the dashboard record are real from here on",
            offender.track_id, ctx.frame_number,
        )

        event = make_event(
            EventType.COLLISION,
            SourceModule.INCIDENT,
            confidence=0.0,
            severity=Severity.CRITICAL,
            class_name=offender.object_type,
            track_id=offender.track_id,
            evidence_image=evidence.get("incident_frame"),
            metadata={
                "incident_id": self._incident_id(),
                "incident_type": EventType.COLLISION,
                "status": IncidentStatus.SUSPECTED,
                "synthetic": True,
                "vehicle_track_id": offender.track_id,
                "vehicle_class": offender.object_type,
                "other_track_id": collision["other_track_id"],
                "impact_frame": ctx.frame_number,
                "detector": "demo injection (not a detection)",
                "evidence": (
                    "SYNTHETIC demo trigger -- this collision was injected by "
                    "--demo-incident to exercise the chain. It is NOT a detection."
                ),
                "evidence_frames": evidence,
                "requires_identification": False,
            },
        )
        case.collision_event_id = event.event_id
        return {"collision": collision, "event": event}

    # ------------------------------------------------------------------
    # evidence
    # ------------------------------------------------------------------

    def _remember_frame(self, ctx: FrameContext) -> None:
        if not self.evidence_cfg.get("enabled", True) or ctx.frame is None:
            return
        stride = max(1, int(self.evidence_cfg.get("stride", 3)))
        if ctx.frame_number % stride:
            return
        self._frame_ring.append({
            "frame_number": ctx.frame_number,
            "frame": ctx.frame.copy(),
        })

    def _capture_incident_evidence(self, ctx: FrameContext, track_id: int,
                                   kind: str) -> Dict[str, Any]:
        if not self.evidence_cfg.get("enabled", True):
            return {}
        out: Dict[str, Any] = {}
        out["incident_frame"] = self._save(ctx.frame, kind, ctx)
        if self._frame_ring:
            before = self._frame_ring[0]
            out["frame_before"] = self._save(before["frame"], kind, ctx)
            out["frame_before_number"] = before["frame_number"]
        crop = self._vehicle_crop(ctx, track_id)
        if crop is not None:
            out["vehicle_crop"] = self._save(crop, kind, ctx)
        return {k: v for k, v in out.items() if v is not None}

    def _schedule_after_capture(self, ctx: FrameContext, case: _VehicleCase) -> None:
        if not self.evidence_cfg.get("enabled", True):
            return
        count = int(self.evidence_cfg.get("post_frames", 8))
        if count <= 0:
            return
        self._after_jobs.append({
            "case": case,
            "remaining": count,
            "stride": max(1, int(self.evidence_cfg.get("stride", 3))),
            "frames": [],
        })

    def _run_after_jobs(self, ctx: FrameContext) -> None:
        if not self._after_jobs or ctx.frame is None:
            return
        finished = []
        for job in self._after_jobs:
            if ctx.frame_number % job["stride"]:
                continue
            job["frames"].append(ctx.frame.copy())
            job["remaining"] -= 1
            if job["remaining"] <= 0:
                finished.append(job)
        for job in finished:
            self._after_jobs.remove(job)
            case = job["case"]
            frames = job["frames"]
            if frames:
                case.evidence["frame_after"] = self._save(frames[-1], "hit_and_run", ctx)
                if self.evidence_cfg.get("save_clip", True) and self.evidence_store:
                    clip = self.evidence_store.save_clip(
                        frames, "hit_and_run",
                        fps=max(1.0, (ctx.fps or 25.0) / job["stride"]),
                        bus_id=ctx.bus_id, frame_number=ctx.frame_number,
                    )
                    if clip:
                        case.evidence["clip"] = clip

    def _save(self, image, kind: str, ctx: FrameContext) -> Optional[str]:
        if image is None or self.evidence_store is None:
            return None
        if not self.evidence_cfg.get("enabled", True):
            return None
        return self.evidence_store.save_image(
            image, str(kind), bus_id=ctx.bus_id, frame_number=ctx.frame_number
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _case(self, track_id: int, vehicle_class: Optional[str]) -> _VehicleCase:
        case = self._cases.get(track_id)
        if case is None:
            case = _VehicleCase(track_id, vehicle_class)
            self._cases[track_id] = case
        elif vehicle_class and not case.vehicle_class:
            case.vehicle_class = vehicle_class
        return case

    def _vehicle(self, ctx: FrameContext, track_id: int) -> Optional[Dict[str, Any]]:
        for vehicle in (ctx.vehicles or []):
            if vehicle.get("track_id") == track_id:
                return vehicle
        return None

    def _vehicle_crop(self, ctx: FrameContext, track_id: int):
        vehicle = self._vehicle(ctx, track_id)
        if vehicle is None or ctx.frame is None:
            return None
        from core.evidence import EvidenceStore                  # noqa: PLC0415
        return EvidenceStore.crop(ctx.frame, vehicle.get("bbox"))

    def _incident_id(self) -> str:
        self._incident_seq += 1
        return f"INC-{self._incident_seq:06d}"

    @staticmethod
    def _pair(a, b):
        a, b = int(a), int(b if b is not None else -1)
        return (a, b) if a <= b else (b, a)

    @staticmethod
    def _pair_class(coll: dict) -> str:
        a = coll.get("vehicle_type") or "vehicle"
        b = coll.get("other_type") or "vehicle"
        return f"{a}_{b}"

    @staticmethod
    def _timestamp(ctx: FrameContext) -> str:
        from core.event import _now_iso                          # noqa: PLC0415
        return _now_iso()

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        states: Dict[str, int] = {}
        for case in self._cases.values():
            states[case.state] = states.get(case.state, 0) + 1
        data["hit_and_run_states"] = states
        return data
