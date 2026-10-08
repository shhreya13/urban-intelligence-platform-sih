"""
modules.pedestrian.crossing_logic
=================================

Pure-Python building blocks for the vulnerable-pedestrian rule. No cv2, no
numpy, no model: everything here works on the plain detection dicts already
published on the ``FrameContext``, which is what makes the rule unit-testable
frame by frame.

THE RULE  (SIH26124: "identify vulnerable pedestrian situations such as
school children crossing roads")

    PERSON != PEDESTRIAN

    YOLO detects person
      -> is a zebra crossing detected in the scene?        (ZebraCrossingTracker)
      -> is the person physically on that crossing?        (person_on_crossing)
      -> is the person standing on the road, not on/in a
         vehicle (bus passenger, motorcycle rider)?        (person_on_vehicle)
      -> are vehicles present?
      -> are those vehicles actually moving?               (TrackHistory)
      -> has all of the above held for several
         consecutive frames for the same track?            (CrossingStateMachine)
      -> ONLY THEN: one VULNERABLE PEDESTRIAN EVENT per continuous crossing

Nothing here assumes a fixed image rectangle is a crossing: the only crossing
geometry is what Person 1's ROAD_DEFECT_MODEL actually detected
(class ``zebra_crossing``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

BBox = Sequence[float]
Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# bbox geometry
# ---------------------------------------------------------------------------

def bbox_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def intersection_area(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_center(box: BBox) -> Point:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_bottom_center(box: BBox) -> Point:
    """Ground-contact point of an upright object (a person's feet)."""
    return ((box[0] + box[2]) / 2.0, float(box[3]))


def bbox_diag(box: BBox) -> float:
    return hypot(box[2] - box[0], box[3] - box[1])


def point_to_bbox_distance(point: Point, box: BBox) -> float:
    """0 when the point is inside the box, else Euclidean distance to it."""
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return hypot(dx, dy)


def foot_band(box: BBox, ratio: float) -> List[float]:
    """The bottom ``ratio`` of a person box: the part that touches the road."""
    x1, y1, x2, y2 = box
    height = max(1.0, float(y2) - float(y1))
    return [float(x1), float(y2) - height * ratio, float(x2), float(y2)]


def shift_bbox(box: BBox, dx: float, dy: float) -> List[float]:
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


# ---------------------------------------------------------------------------
# person <-> crossing / vehicle relations
# ---------------------------------------------------------------------------

def person_crossing_overlap(person_box: BBox, crossing_box: BBox, band_ratio: float) -> float:
    """Fraction of the person's foot band that lies on the crossing (0..1).

    Only the feet count. A person on the pavement *behind* a crossing (seen
    from a bus camera) has their head/torso inside the crossing's image box
    but their feet above it, which is exactly the case this must reject.
    """
    band = foot_band(person_box, band_ratio)
    area = bbox_area(band)
    if area <= 0:
        return 0.0
    return intersection_area(band, crossing_box) / area


def person_on_crossing(
    person_box: BBox, crossing_box: BBox, band_ratio: float, min_overlap: float,
) -> Tuple[bool, float]:
    overlap = person_crossing_overlap(person_box, crossing_box, band_ratio)
    foot_x, foot_y = bbox_bottom_center(person_box)
    feet_inside = (
        crossing_box[0] <= foot_x <= crossing_box[2]
        and crossing_box[1] <= foot_y <= crossing_box[3]
    )
    return (feet_inside and overlap >= min_overlap), overlap


def vehicle_occluded_by_person(
    vehicle_box: BBox, person_box: BBox, min_overlap: float, bottom_margin: float,
) -> bool:
    """A person who is NOT on this vehicle (so not its rider/passenger) covers
    part of its box: the vehicle's box, and so its centre, is distorted."""
    area = bbox_area(vehicle_box)
    if area <= 0:
        return False
    if person_on_vehicle(person_box, vehicle_box, bottom_margin):
        return False
    return intersection_area(vehicle_box, person_box) / area >= min_overlap


def person_on_vehicle(person_box: BBox, vehicle_box: BBox, bottom_margin: float) -> bool:
    """True when the person's feet are *within* a vehicle's box, clearly above
    the vehicle's own ground line: a bus passenger seen through a window, a
    motorcycle/scooter rider, someone on a truck bed. A pedestrian walking in
    front of a vehicle has feet below (closer than) the vehicle's bottom edge
    and is not matched."""
    foot_x, foot_y = bbox_bottom_center(person_box)
    vx1, vy1, vx2, vy2 = vehicle_box
    v_height = max(1.0, float(vy2) - float(vy1))
    return (
        vx1 <= foot_x <= vx2
        and vy1 <= foot_y <= (vy2 - bottom_margin * v_height)
    )


# ---------------------------------------------------------------------------
# zebra crossings (from Person 1's ROAD_DEFECT_MODEL)
# ---------------------------------------------------------------------------

@dataclass
class ZebraCrossing:
    crossing_id: int
    bbox: List[float]
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    hits: int = 1
    history: Deque[Tuple[int, float, float]] = field(default_factory=lambda: deque(maxlen=12))

    def velocity(
        self, max_gap_frames: int, min_points: int = 3, significance: float = 2.0,
    ) -> Optional[Tuple[float, float]]:
        est = self.velocity_estimate(max_gap_frames, min_points, significance)
        return None if est is None else (est[0], est[1])

    def velocity_estimate(
        self, max_gap_frames: int, min_points: int = 3, significance: float = 2.0,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Apparent image velocity (px/frame) of the crossing and its standard
        error, least-squares over its recent detections: ``(vx, vy, sx, sy)``.

        A crossing is painted on the road, so this is the camera's own (bus)
        motion at road level: used to stop parked cars from looking like
        moving traffic when the bus itself is moving.

        ``None`` when it cannot be estimated yet (fewer than ``min_points``
        contiguous detections). A component that is not significant against
        the fit's residual scatter (detector box jitter on a stationary bus) is
        reported as 0, and its standard error is returned so callers can widen
        their own thresholds by the uncertainty that remains.
        """
        pts = self._velocity_points(max_gap_frames)
        n = len(pts)
        if n < max(2, min_points):
            return None
        mf = sum(p[0] for p in pts) / n
        var = sum((p[0] - mf) ** 2 for p in pts)
        if var <= 0:
            return None
        slopes, errors = [], []
        for axis in (1, 2):
            mean = sum(p[axis] for p in pts) / n
            slope = sum((p[0] - mf) * (p[axis] - mean) for p in pts) / var
            stderr = 0.0
            if n > 2:
                resid = sum(
                    (p[axis] - (mean + slope * (p[0] - mf))) ** 2 for p in pts
                )
                stderr = (resid / (n - 2)) ** 0.5 / var ** 0.5
                if stderr > 0 and abs(slope) < significance * stderr:
                    slope = 0.0
            slopes.append(slope)
            errors.append(stderr)
        return (slopes[0], slopes[1], errors[0], errors[1])

    def _velocity_points(self, max_gap_frames: int) -> List[Tuple[int, float, float]]:
        if not self.history:
            return []
        latest = self.history[-1][0]
        pts = [h for h in self.history if latest - h[0] <= 4 * max_gap_frames]
        # a gap larger than max_gap_frames between consecutive passes breaks
        # the chain: only keep the most recent contiguous run
        run = [pts[-1]]
        for h in reversed(pts[:-1]):
            if run[0][0] - h[0] > max_gap_frames:
                break
            run.insert(0, h)
        return run

    def predicted_bbox(self, frame_number: int, max_gap_frames: int) -> List[float]:
        """The crossing box carried forward between detection passes (the road
        model runs every ``frame_stride`` frames) using its own velocity."""
        dt = max(0, frame_number - self.last_seen_frame)
        if dt == 0:
            return list(self.bbox)
        vel = self.velocity(max_gap_frames)
        if vel is None:
            return list(self.bbox)
        return shift_bbox(self.bbox, vel[0] * dt, vel[1] * dt)

    def as_dict(self, frame_number: int, max_gap_frames: int) -> Dict[str, Any]:
        return {
            "crossing_id": self.crossing_id,
            "bbox": [int(round(v)) for v in self.predicted_bbox(frame_number, max_gap_frames)],
            "detected_bbox": [int(round(v)) for v in self.bbox],
            "confidence": round(self.confidence, 3),
            "hits": self.hits,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
        }


class ZebraCrossingTracker:
    """Keeps the crossings the road model has *actually detected* alive
    between its strided inference passes, and only reports a crossing once it
    has been detected on ``min_detections`` passes (one-pass flicker is not a
    crossing)."""

    def __init__(
        self,
        min_detections: int = 2,
        hold_frames: int = 15,
        match_iou: float = 0.2,
        max_velocity_gap_frames: int = 15,
    ):
        self.min_detections = max(1, int(min_detections))
        self.hold_frames = max(1, int(hold_frames))
        self.match_iou = float(match_iou)
        self.max_velocity_gap_frames = max(1, int(max_velocity_gap_frames))
        self._crossings: Dict[int, ZebraCrossing] = {}
        self._next_id = 1

    def update(self, detections: Iterable[Dict[str, Any]], frame_number: int) -> None:
        """Feed one road-model pass (only on frames where it actually ran)."""
        unmatched = set(self._crossings.keys())
        for det in sorted(detections, key=lambda d: -float(d.get("confidence", 0.0))):
            box = [float(v) for v in det["bbox"]]
            best_id, best_score = None, 0.0
            for cid in unmatched:
                crossing = self._crossings[cid]
                ref = crossing.predicted_bbox(frame_number, self.max_velocity_gap_frames)
                score = iou(box, ref)
                if score < self.match_iou:
                    cx, cy = bbox_center(box)
                    rx, ry = bbox_center(ref)
                    if hypot(cx - rx, cy - ry) <= 0.35 * bbox_diag(ref):
                        score = max(score, self.match_iou)
                if score >= self.match_iou and score > best_score:
                    best_id, best_score = cid, score
            cx, cy = bbox_center(box)
            if best_id is None:
                crossing = ZebraCrossing(
                    crossing_id=self._next_id, bbox=box,
                    confidence=float(det.get("confidence", 0.0)),
                    first_seen_frame=frame_number, last_seen_frame=frame_number,
                )
                crossing.history.append((frame_number, cx, cy))
                self._crossings[self._next_id] = crossing
                self._next_id += 1
            else:
                unmatched.discard(best_id)
                crossing = self._crossings[best_id]
                crossing.bbox = box
                crossing.confidence = float(det.get("confidence", 0.0))
                crossing.last_seen_frame = frame_number
                crossing.hits += 1
                crossing.history.append((frame_number, cx, cy))

    def active(self, frame_number: int) -> List[ZebraCrossing]:
        return [
            c for c in self._crossings.values()
            if c.hits >= self.min_detections
            and 0 <= frame_number - c.last_seen_frame <= self.hold_frames
        ]

    def prune(self, frame_number: int) -> None:
        stale = [
            cid for cid, c in self._crossings.items()
            if frame_number - c.last_seen_frame > self.hold_frames * 3
        ]
        for cid in stale:
            del self._crossings[cid]


# ---------------------------------------------------------------------------
# frame-to-frame positions per track id
# ---------------------------------------------------------------------------

class TrackHistory:
    """Center positions per tracker id, used to decide from *real frame-to-
    frame displacement* (not detection confidence, not presence) whether a
    track is moving."""

    def __init__(self, maxlen: int = 90):
        self.maxlen = max(4, int(maxlen))
        # (frame, cx, cy, height, width, reliable)
        self._tracks: Dict[int, Deque[Tuple[int, float, float, float, float, bool]]] = {}

    def update(self, track_id: int, frame_number: int, box: BBox, reliable: bool = True) -> None:
        """``reliable=False`` marks a sample whose box is known to be distorted
        (e.g. partly hidden behind a person walking in front of it): it is kept
        for bookkeeping but never used to measure movement."""
        hist = self._tracks.get(track_id)
        if hist is None:
            hist = deque(maxlen=self.maxlen)
            self._tracks[track_id] = hist
        if hist and hist[-1][0] == frame_number:
            hist.pop()
        cx, cy = bbox_center(box)
        hist.append((
            frame_number, cx, cy,
            max(1.0, float(box[3]) - float(box[1])),
            max(1.0, float(box[2]) - float(box[0])),
            bool(reliable),
        ))

    def last_frame(self, track_id: int) -> Optional[int]:
        hist = self._tracks.get(track_id)
        return hist[-1][0] if hist else None

    def displacement(
        self,
        track_id: int,
        frame_number: int,
        window_frames: int,
        ego_velocity: Tuple[float, float] = (0.0, 0.0),
        min_samples: int = 3,
        min_span_ratio: float = 0.6,
        max_shape_change: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """Net displacement over the last ``window_frames`` frames, with the
        camera's own motion removed, from reliable samples only. ``None`` when
        there is not enough clean history to judge (a brand-new or mostly
        occluded track is never assumed to be moving).

        ``max_shape_change`` rejects a measurement whose box changed shape
        disproportionately between the start and end of the window
        (|log w-ratio - log h-ratio| above it): that is one side of the box
        being cut off by occlusion or the frame edge, not movement. An
        approaching vehicle grows proportionally and is not rejected.
        """
        hist = self._tracks.get(track_id)
        if not hist:
            return None
        start = frame_number - window_frames
        samples = [s for s in hist if start <= s[0] <= frame_number and s[5]]
        if len(samples) < min_samples:
            return None
        span = samples[-1][0] - samples[0][0]
        if span < max(1, min_span_ratio * window_frames):
            return None
        # average a few samples at each end so single-frame box jitter of a
        # parked car does not read as movement
        k = max(1, min(3, len(samples) // 3))
        head, tail = samples[:k], samples[-k:]
        f0 = sum(s[0] for s in head) / k
        x0 = sum(s[1] for s in head) / k
        y0 = sum(s[2] for s in head) / k
        f1 = sum(s[0] for s in tail) / k
        x1 = sum(s[1] for s in tail) / k
        y1 = sum(s[2] for s in tail) / k
        if max_shape_change is not None:
            from math import log
            h0 = sum(s[3] for s in head) / k
            w0 = sum(s[4] for s in head) / k
            h1 = sum(s[3] for s in tail) / k
            w1 = sum(s[4] for s in tail) / k
            if abs(log(w1 / w0) - log(h1 / h0)) > max_shape_change:
                return None
        dt = max(1e-6, f1 - f0)
        dx = (x1 - x0) - ego_velocity[0] * dt
        dy = (y1 - y0) - ego_velocity[1] * dt
        mean_h = sum(s[3] for s in samples) / len(samples)
        return {
            "dx": dx, "dy": dy, "magnitude": hypot(dx, dy),
            "span_frames": float(span), "mean_height": mean_h,
            "px_per_frame": hypot(dx, dy) / dt,
        }

    def prune(self, frame_number: int, max_age_frames: int) -> List[int]:
        dead = [
            tid for tid, hist in self._tracks.items()
            if not hist or frame_number - hist[-1][0] > max_age_frames
        ]
        for tid in dead:
            del self._tracks[tid]
        return dead


def is_moving(
    disp: Optional[Dict[str, float]], min_px: float, min_ratio: float,
    uncertainty_px: float = 0.0,
) -> bool:
    """Moving = net displacement above a scale-aware jitter floor, PLUS any
    remaining uncertainty in the camera-motion estimate that was subtracted.
    Unknown history is never 'moving'."""
    if disp is None:
        return False
    threshold = max(float(min_px), float(min_ratio) * disp["mean_height"])
    return disp["magnitude"] >= threshold + max(0.0, float(uncertainty_px))


# ---------------------------------------------------------------------------
# per-person crossing state machine
# ---------------------------------------------------------------------------

class CrossingPhase:
    IDLE = "IDLE"                # not on a detected crossing
    CANDIDATE = "CANDIDATE"      # on a crossing, not (yet) confirmed
    CONFIRMED = "CONFIRMED"      # event emitted for this continuous crossing


@dataclass
class PersonCrossingState:
    track_id: int
    phase: str = CrossingPhase.IDLE
    qualifying_streak: int = 0
    occupancy_frames: int = 0
    off_frames: int = 0
    missing_frames: int = 0
    last_seen_frame: int = -1
    episode_start_frame: Optional[int] = None
    confirmed_frame: Optional[int] = None
    closed_frame: Optional[int] = None
    event_id: Optional[str] = None
    last_episode_confirmed: bool = False
    episodes_confirmed: int = 0
    crossing_id: Optional[int] = None
    start_rel: Optional[float] = None
    progress_px: float = 0.0
    last_point: Optional[Point] = None
    last_height: float = 0.0
    person_conf: Deque[float] = field(default_factory=lambda: deque(maxlen=60))


class Transition:
    NONE = "none"
    STARTED = "started"
    RESUMED = "resumed"          # re-entered within cooldown, same event
    CONFIRMED = "confirmed"      # -> emit exactly one event
    CLOSED = "closed"


class CrossingStateMachine:
    """One state per person track id.

    * CANDIDATE starts the first frame a person's feet are on a detected
      crossing.
    * CONFIRMED needs ``confirm_frames`` *consecutive* frames on which every
      condition held (crossing detected, person on it, moving vehicles).
      That transition is the only thing that emits an event.
    * While CONFIRMED nothing is emitted again, however long the person
      stays on the crossing.
    * The episode CLOSES after ``exit_frames`` consecutive detected frames
      off the crossing, or ``track_lost_frames`` frames without a detection.
    * A later crossing by the same track opens a new episode (and may emit
      a new event), unless it starts within ``reentry_cooldown_frames`` of a
      confirmed close, in which case it is the same crossing resumed.
    """

    def __init__(
        self,
        confirm_frames: int,
        exit_frames: int,
        track_lost_frames: int,
        reentry_cooldown_frames: int,
    ):
        self.confirm_frames = max(2, int(confirm_frames))
        self.exit_frames = max(1, int(exit_frames))
        self.track_lost_frames = max(1, int(track_lost_frames))
        self.reentry_cooldown_frames = max(0, int(reentry_cooldown_frames))
        self.states: Dict[int, PersonCrossingState] = {}

    def get(self, track_id: int) -> PersonCrossingState:
        state = self.states.get(track_id)
        if state is None:
            state = PersonCrossingState(track_id=track_id)
            self.states[track_id] = state
        return state

    def observe(
        self,
        track_id: int,
        frame_number: int,
        on_crossing: bool,
        qualifies: bool,
    ) -> str:
        """Advance one detected person by one frame."""
        s = self.get(track_id)
        s.last_seen_frame = frame_number
        s.missing_frames = 0

        if not on_crossing:
            s.qualifying_streak = 0
            if s.phase == CrossingPhase.IDLE:
                return Transition.NONE
            s.off_frames += 1
            if s.off_frames >= self.exit_frames:
                self._close(s, frame_number)
                return Transition.CLOSED
            return Transition.NONE

        s.off_frames = 0
        transition = Transition.NONE
        if s.phase == CrossingPhase.IDLE:
            recent = (
                s.last_episode_confirmed
                and s.closed_frame is not None
                and frame_number - s.closed_frame < self.reentry_cooldown_frames
            )
            if recent:
                s.phase = CrossingPhase.CONFIRMED
                transition = Transition.RESUMED
            else:
                s.phase = CrossingPhase.CANDIDATE
                s.episode_start_frame = frame_number
                s.confirmed_frame = None
                s.event_id = None
                s.occupancy_frames = 0
                s.qualifying_streak = 0
                s.start_rel = None
                s.progress_px = 0.0
                s.person_conf.clear()
                transition = Transition.STARTED

        s.occupancy_frames += 1
        if s.phase == CrossingPhase.CANDIDATE:
            s.qualifying_streak = s.qualifying_streak + 1 if qualifies else 0
        return transition

    def can_confirm(self, track_id: int) -> bool:
        s = self.states.get(track_id)
        return (
            s is not None
            and s.phase == CrossingPhase.CANDIDATE
            and s.qualifying_streak >= self.confirm_frames
        )

    def confirm(self, track_id: int, frame_number: int, event_id: Optional[str]) -> None:
        s = self.get(track_id)
        s.phase = CrossingPhase.CONFIRMED
        s.confirmed_frame = frame_number
        s.event_id = event_id
        s.episodes_confirmed += 1

    def mark_missing(self, frame_number: int, present_ids: Iterable[int]) -> List[int]:
        """Age every known track that was not detected this frame. Returns
        the ids whose episode closed because the track was lost."""
        present = set(present_ids)
        closed: List[int] = []
        for tid, s in self.states.items():
            if tid in present:
                continue
            s.missing_frames += 1
            if s.phase != CrossingPhase.IDLE and s.missing_frames > self.track_lost_frames:
                self._close(s, frame_number)
                closed.append(tid)
        return closed

    def prune(self, frame_number: int, max_idle_frames: int) -> None:
        dead = [
            tid for tid, s in self.states.items()
            if s.phase == CrossingPhase.IDLE
            and frame_number - s.last_seen_frame > max_idle_frames
        ]
        for tid in dead:
            del self.states[tid]

    @staticmethod
    def _close(s: PersonCrossingState, frame_number: int) -> None:
        s.last_episode_confirmed = s.phase == CrossingPhase.CONFIRMED
        s.phase = CrossingPhase.IDLE
        s.closed_frame = frame_number
        s.qualifying_streak = 0
        s.off_frames = 0
