"""
hit_and_run.py
==============

Person 4 - Hit-and-run detection for the Urban Intelligence Platform.

Chain: COLLISION -> HIT-AND-RUN -> ANPR
        (this module)           (next)

A hit-and-run is a confirmed COLLISION after which one of the vehicles
leaves the incident location instead of stopping, while the other (victim)
does not move away comparably.

What it does
------------
- Locks the COLLIDING vehicle's track ID once a collision is confirmed.
- Records the incident location (the vehicle's bottom-centre/foot at the
  moment of the collision).
- Monitors that same track afterwards:
      * if it moves away from the incident point by a meaningful distance
        (>= min_escape_distance_ratio x its own size) ...
      * with a sustained, mostly-monotonic run of "away" frames ...
      * for at least min_escape_frames ...
      -> a hit-and-run is confirmed and emitted ONCE per locked vehicle.

Deliberate design choices
-------------------------
- Track DISAPPEARANCE alone NEVER confirms a hit-and-run. A vehicle that
  simply vanishes (occlusion / leaves frame / id-switch) without a
  consistent run of "moving away" frames gets NO event. This avoids the
  classic false positive where BoT-SORT drops the id in congestion.
- The victim (other object) position is NOT required after the crash: the
  reference "escape point" is the crash location itself, which is robust
  when the victim also gets pushed or the video cuts.

Output / contract
-----------------
    update(tracked_objects, collisions_this_frame) -> list[dict]

Each confirmed hit-and-run looks like:

{
    "incident_type": "hit_and_run",
    "hit_vehicle_track_id": 7,
    "vehicle_type": "car",
    "collision_frame_id": 1532,
    "collision_timestamp": "...iso...",
    "frame_id": 1610,
    "timestamp": "...iso...",
    "timestamp_seconds": 68.72,
    "escape_distance_px": 218.4,
    "escape_speed_px_s": 22.7,
    "escape_direction": "right",
    "progress": 0.71,
    "confidence": 0.74
}
"""

import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple

BOX = Tuple[int, int, int, int]


class HitAndRunDetector:
    """Stateful hit-and-run detector driven by confirmed collision events."""

    def __init__(
        self,
        fps: float = 30.0,
        min_escape_frames: int = 12,
        min_escape_distance_px: float = 70.0,
        escape_ratio: float = 1.5,
        escape_progress: float = 0.5,
        require_consecutive: int = 3,
        max_missing_frames: int = 30,
        cooldown_frames: int = 90,
        debug: bool = False,
    ):
        self.fps = float(fps)
        self.min_escape_frames = min_escape_frames
        self.min_escape_distance_px = min_escape_distance_px
        self.escape_ratio = escape_ratio
        self.escape_progress = escape_progress
        self.require_consecutive = require_consecutive
        self.max_missing_frames = max_missing_frames
        self.cooldown_frames = cooldown_frames
        self.debug = debug

        # track_id -> foot-point history (frame_id, (x, y), timestamp)
        self._positions: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=60)
        )

        # locked vehicles awaiting escape confirmation
        self._pending: Dict[int, dict] = {}

        # collision frame for each locked vehicle -> cooldown guard
        self._last_lock: Dict[int, int] = {}

        # vehicles already flagged -> never re-flag
        self._confirmed: set = set()

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    def update(
        self,
        tracked_objects,
        collisions_this_frame: List[dict],
    ) -> List[dict]:
        """
        Call once per frame:

            objects = tracker.process_frame(frame)
            collisions = collision.update(objects, frame_size=...)
            hit_and_runs = hit_and_run.update(objects, collisions)

        Returns newly confirmed hit-and-run events for THIS frame.
        """
        events: List[dict] = []

        # 1) Refresh foot-position history for every object we care about.
        for obj in tracked_objects:
            track_id = int(obj.track_id)
            box = (int(obj.bbox[0]), int(obj.bbox[1]),
                   int(obj.bbox[2]), int(obj.bbox[3]))
            cx = (box[0] + box[2]) / 2.0
            foot = (cx, float(box[3]))
            self._positions[track_id].append({
                "frame_id": int(obj.frame_id),
                "foot": foot,
                "bbox": box,
                "type": getattr(obj, "object_type", ""),
                "timestamp": getattr(obj, "timestamp", None),
            })

        # 2) Register any new collisions as candidate hit-and-runs.
        for coll in collisions_this_frame:
            self._lock(coll)

        # 3) Evaluate every pending vehicle.
        for track_id in list(self._pending.keys()):
            result = self._evaluate_pending(track_id)
            if result is not None:
                events.append(result)
                self._confirmed.add(track_id)
                del self._pending[track_id]

        return events

    # ============================================================
    # LOCK ONTO A COLLIDING VEHICLE
    # ============================================================

    def _lock(self, collision: dict):
        track_id = int(collision["vehicle_track_id"])

        if track_id in self._confirmed:
            return

        # Cooldown: ignore re-locking the same vehicle within a short time.
        if frame_id := collision.get("frame_id"):
            if frame_id - self._last_lock.get(track_id, -10 ** 9) \
                    < self.cooldown_frames:
                return

        hist = self._positions.get(track_id, [])
        if not hist:
            return
        hist = list(hist)

        # Incident point = where the vehicle's foot was at the collision.
        # Prefer the position whose frame matches the collision, else latest.
        target_frame = collision.get("frame_id")
        rec = None
        if target_frame is not None:
            rec = next(
                (h for h in reversed(hist)
                 if h["frame_id"] <= target_frame),
                hist[-1],
            )
        else:
            rec = hist[-1]

        box = rec["bbox"]
        ref_scale = max(2.0, (box[2] - box[0]), (box[3] - box[1]))

        self._pending[track_id] = {
            "track_id": track_id,
            "vehicle_type": collision.get("vehicle_type", "vehicle"),
            "other_track_id": collision.get("other_track_id"),
            "other_type": collision.get("other_type"),
            "collision": collision,
            "collision_frame_id": collision.get("frame_id"),
            "collision_timestamp": collision.get("timestamp"),
            "incident_foot": rec["foot"],
            "ref_scale": ref_scale,
            "onset_distance": 0.0,
            "max_distance": 0.0,
            "appearance_frames": 0,
            "away_steps": 0,
            "away_runs": 0,
            "best_run": 0,
            "last_distance": None,
            "last_away_frame": None,
            "last_frame": None,
            "missing_frames": 0,
            "last_timestamp": None,
        }

        self._last_lock[track_id] = collision.get("frame_id", 0)

    # ============================================================
    # EVALUATE A PENDING VEHICLE
    # ============================================================

    def _evaluate_pending(self, track_id: int) -> dict | None:
        pending = self._pending[track_id]

        current_hist = list(self._positions.get(track_id, []))
        if not current_hist:
            pending["missing_frames"] += 1
            if pending["missing_frames"] > self.max_missing_frames:
                self._expire(pending)
            return None

        rec = current_hist[-1]
        foot = rec["foot"]
        frame_id = rec["frame_id"]
        pending["last_frame"] = frame_id
        pending["last_foot"] = foot
        pending["last_timestamp"] = rec.get("timestamp")
        pending["missing_frames"] = 0
        pending["appearance_frames"] += 1

        dist = math.hypot(
            foot[0] - pending["incident_foot"][0],
            foot[1] - pending["incident_foot"][1],
        )
        pending["max_distance"] = max(pending["max_distance"], dist)

        prev = pending["last_distance"]
        if prev is not None and dist > prev + 1.0:
            pending["away_steps"] += 1
            away_before = pending["last_away_frame"]
            if away_before is not None and frame_id - away_before <= 2:
                pending["best_run"] += 1
            else:
                pending["best_run"] = 1
            pending["last_away_frame"] = frame_id
        elif prev is not None and dist < prev - 1.0:
            # moving back toward the scene: weaken the run, no hard reset
            pending["best_run"] = max(0, pending["best_run"] - 2)
        pending["last_distance"] = dist

        if self._confirm_conditions(pending):
            return self._emit(pending)

        return None

    # ============================================================
    # CONFIRMATION RULES
    # ============================================================

    def _confirm_conditions(self, pending: dict) -> bool:
        if pending["appearance_frames"] < self.min_escape_frames:
            return False

        escape_distance = pending["max_distance"] - pending["onset_distance"]
        required_px = max(
            self.min_escape_distance_px,
            self.escape_ratio * pending["ref_scale"],
        )
        if escape_distance < required_px:
            return False

        # Escaping = a sustained, near-continuous progress away from the
        # incident point. A genuine flee shows several consecutive away-frames.
        if pending["best_run"] < self.require_consecutive:
            return False

        return True

    # ============================================================
    # EMIT + LIFECYCLE HELPERS
    # ============================================================

    def _emit(self, pending: dict) -> dict:
        collision = pending["collision"]
        escape_distance = pending["max_distance"] - pending["onset_distance"]

        frame_id = pending["last_frame"] or 0
        escape_window = pending["last_frame"] - (pending["collision_frame_id"] or 0)
        escape_window = max(1, escape_window)

        # px per second (scale-free at the source scale)
        escape_speed = escape_distance / escape_window * self.fps

        dx = pending["last_distance"] if pending["last_distance"] else 0.0

        direction = self._direction(pending)

        confidence = min(
            1.0,
            round(
                0.40
                + 0.20 * min(1.0, escape_distance / (3.0 * pending["ref_scale"]))
                + 0.20 * min(
                    1.0,
                    escape_speed / max(1.0, self.fps * pending["ref_scale"] * 0.6),
                )
                + 0.20 * min(1.0, pending["best_run"] / 10.0),
                2,
            ),
        )
        confidence = max(0.55, confidence)

        ts = pending["last_timestamp"] or collision.get("timestamp")

        return {
            "incident_type": "hit_and_run",
            "hit_vehicle_track_id": pending["track_id"],
            "vehicle_type": pending["vehicle_type"],
            "other_track_id": pending["other_track_id"],
            "other_type": pending["other_type"],
            "collision_frame_id": pending["collision_frame_id"],
            "collision_timestamp": pending["collision_timestamp"],
            "frame_id": frame_id,
            "timestamp": ts,
            "timestamp_seconds": (
                round(frame_id / self.fps, 3) if self.fps > 0 else None
            ),
            "escape_distance_px": round(escape_distance, 2),
            "escape_speed_px_s": round(escape_speed, 2),
            "escape_direction": direction,
            "progress": round(
                min(1.0, escape_distance / (3.0 * pending["ref_scale"])), 3
            ),
            "confidence": confidence,
            "gps": None,     # no GPS feed available -> honest null
        }

    def _direction(self, pending: dict) -> str:
        foot = pending["last_foot"] if "last_foot" in pending else None
        if foot is None:
            return "unknown"
        dx = foot[0] - pending["incident_foot"][0]
        dy = foot[1] - pending["incident_foot"][1]
        if max(abs(dx), abs(dy)) < 1.0:
            return "stationary"
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    def _expire(self, pending: dict):
        track_id = pending["track_id"]
        self._pending.pop(track_id, None)

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        self._positions.clear()
        self._pending.clear()
        self._confirmed.clear()
        self._last_lock.clear()