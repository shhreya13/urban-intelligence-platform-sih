"""
collision_detection.py
======================

Person 4 - Collision detection for the Urban Intelligence Platform.

Detects:

    A. Vehicle <-> Vehicle
    B. Vehicle <-> Pedestrian (person)

What it does:
    - Operates on BoT-SORT TrackedObject lists (one call per video frame).
    - Uses MULTIPLE spatial + temporal signals, never a single weak one:
        1. bounding-box overlap / IoU
        2. containment ratio (how much of the smaller box is covered)
        3. normalized center distance / gap
        4. approaching behaviour over recent frames
        5. impact-like motion change (sudden speed change around contact)
        6. temporal persistence inside a rolling window
    - Debounces with a per-pair cooldown so one crash is NOT reported
      multiple times.

Output / contract
    update(tracked_objects) -> list[dict]

    Each confirmed collision looks like:

    {
        "incident_type": "collision",
        "vehicle_track_id": 7,          # ALWAYS the vehicle
        "vehicle_type": "car",
        "other_track_id": 12,           # other vehicle OR the pedestrian
        "other_type": "motorcycle",     # or "person"
        "frame_id": 1532,
        "timestamp": "...iso...",
        "timestamp_seconds": 65.14,
        "iou": 0.23,
        "distance": 41.2,               # px center distance at peak
        "approach": 0.86,
        "motion_change": 0.55,
        "containment": False,
        "confidence": 0.81
    }

For a Vehicle <-> Pedestrian pair, vehicle_track_id is always the vehicle.
For a Vehicle <-> Vehicle pair, the primary (vehicle_track_id) is chosen as
the non-motorcycle when one of the two is a motorcycle, otherwise the faster
object at contact.
"""

from collections import defaultdict, deque
import math
from typing import Dict, List, Tuple

from ai.vehicle.classes import VEHICLE_CLASSES


BOX = Tuple[int, int, int, int]  # x1, y1, x2, y2


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def box_area(box: BOX) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


def box_center(box: BOX) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_union(a: BOX, b: BOX) -> BOX:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def compute_iou(a: BOX, b: BOX) -> float:
    """Intersection over Union of two boxes (0..1)."""
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    if inter <= 0.0:
        return 0.0

    union = box_area(a) + box_area(b) - inter

    if union <= 0.0:
        return 0.0

    return inter / union


def compute_cover(a: BOX, b: BOX) -> float:
    """
    How much of the SMALLER box is covered by the other (0..1).
    A high value often means one tracked box (usually a person) is fully
    inside a much bigger box (usually a vehicle) - which is usually an
    occlusion/annotation effect, not a real collision.
    """
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    if inter <= 0.0:
        return 0.0

    smaller_area = min(box_area(a), box_area(b))

    if smaller_area <= 0.0:
        return 0.0

    return inter / smaller_area


class CollisionDetector:
    """
    Stateful collision detector.

    Call update(tracked_objects) once per frame with the list of
    TrackedObject records produced by BoT-SORT.
    """

    def __init__(
        self,
        fps: float = 30.0,
        history_frames: int = 30,
        report_threshold: float = 0.30,
        confirm_peak: float = 0.62,
        confirm_sustained: float = 0.50,
        min_persist_frames: int = 2,
        min_iou: float = 0.05,
        confirm_min_iou: float = 0.08,
        overlap_episode_min: float = 0.06,
        max_episode_frames: int = 30,
        separated_threshold: float = 0.06,
        max_vehicle_fraction: float = 0.42,
        approach_min: float = 0.35,
        motion_min: float = 0.30,
        cooldown_frames: int = 45,
        debug: bool = False,
    ):
        self.fps = float(fps)
        self.history_frames = history_frames
        self.report_threshold = report_threshold
        self.confirm_peak = confirm_peak
        self.confirm_sustained = confirm_sustained
        self.min_persist_frames = min_persist_frames
        self.min_iou = min_iou
        self.confirm_min_iou = confirm_min_iou
        self.overlap_episode_min = overlap_episode_min
        self.max_episode_frames = max_episode_frames
        self.max_vehicle_fraction = max_vehicle_fraction
        self.separated_threshold = separated_threshold
        self.approach_min = approach_min
        self.motion_min = motion_min
        self.cooldown_frames = cooldown_frames
        self.debug = debug

        # Track ID -> recent positions (bbox, type, timestamp)
        self._positions: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.history_frames)
        )

        # Normalized pair key (a < b) -> rolling window of per-frame signals
        self._pair_window: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=self.history_frames)
        )

        # Normalized pair key -> last frame a collision was emitted
        self._last_emitted: Dict[Tuple[int, int], int] = {}

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    def update(self, tracked_objects, frame_size=None) -> List[dict]:
        """
        One call per frame.

        frame_size: optional (width, height) tuple of the video frame. When
        provided, boxes that cover a huge fraction of the frame (a near-field
        occluder / close-up object) are ignored as usable collision partners
        because they overlap almost everything and cause false positives.

        Returns a list of newly confirmed collision events for THIS frame
        (usually 0-1, occasionally 2 when two independent crashes happen).
        """
        events: List[dict] = []

        frame_w = frame_h = None
        if frame_size is not None:
            frame_w, frame_h = frame_size
            frame_area = float(max(frame_w, 1) * max(frame_h, 1))

        current = []

        for obj in tracked_objects:
            otype = obj.object_type
            track_id = int(obj.track_id)

            if otype not in VEHICLE_CLASSES and otype != "person":
                continue

            box = (int(obj.bbox[0]), int(obj.bbox[1]),
                   int(obj.bbox[2]), int(obj.bbox[3]))

            box_area_value = box_area(box)
            if box_area_value <= 0:
                continue

            # Near-field occluder guard: a box covering a huge share of the
            # frame is a close-up object, not a meaningful collision partner.
            if frame_area is not None and \
                    box_area_value / frame_area > self.max_vehicle_fraction:
                continue

            cx, cy = box_center(box)
            foot = (cx, float(box[3]))  # bottom-centre is jitter-robust

            self._positions[track_id].append({
                "frame_id": int(obj.frame_id),
                "bbox": box,
                "center": (cx, cy),
                "foot": foot,
                "type": otype,
                "timestamp": getattr(obj, "timestamp", None),
            })

            current.append({
                "track_id": track_id,
                "type": otype,
                "bbox": box,
                "center": (cx, cy),
                "frame_id": int(obj.frame_id),
                "timestamp": getattr(obj, "timestamp", None),
            })

        # --------------------------------------------------------
        # Pair evaluation
        # --------------------------------------------------------

        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                a = current[i]
                b = current[j]

                pair_events = self._evaluate_pair(a, b)
                events.extend(pair_events)

        return events

    # ============================================================
    # SINGLE PAIR EVALUATION
    # ============================================================

    def _evaluate_pair(self, a: dict, b: dict) -> List[dict]:
        ta, tb = a["type"], b["type"]

        # person-person pairs are not incidents
        if ta == "person" and tb == "person":
            return []

        # at least one of the two must be a vehicle
        if ta not in VEHICLE_CLASSES and tb not in VEHICLE_CLASSES:
            return []

        paired = {a["track_id"]: a, b["track_id"]: b}

        iou = compute_iou(a["bbox"], b["bbox"])
        cover = compute_cover(a["bbox"], b["bbox"])

        # Containment (one box fully swallowing the other): heavily suppress.
        containment = cover >= 0.80

        ref = self._reference_scale(a["bbox"], b["bbox"])

        if ref <= 0:
            return []

        center_dist = math.hypot(
            a["center"][0] - b["center"][0],
            a["center"][1] - b["center"][1],
        )
        normalized_dist = center_dist / ref

        # ------------------------------------------------
        # Geometry signal (real box overlap only)
        # ------------------------------------------------
        if iou > 0.0:
            geom = iou
            if containment:
                geom *= 0.30
        else:
            # Boxes are not overlapping: a near-gap must NOT alone confirm
            # a collision, so it contributes only weakly and capped.
            geom = min(0.35, max(0.0, 1.0 - normalized_dist)) * 0.5

        # ------------------------------------------------
        # Approaching behaviour (over last N frames)
        # ------------------------------------------------
        approach = self._approach_evidence(a["track_id"], b["track_id"])

        # ------------------------------------------------
        # Impact-like motion change
        # ------------------------------------------------
        motion_change = self._motion_change(
            a["track_id"],
            b["track_id"],
        )

        # ------------------------------------------------
        # Instant score
        # ------------------------------------------------
        instant = min(
            1.0,
            0.55 * geom
            + 0.25 * approach
            + 0.20 * motion_change,
        )

        pair_key = self._pair_key(a["track_id"], b["track_id"])
        frame_id = int(a.get("frame_id", 0))

        self._pair_window[pair_key].append({
            "frame_id": frame_id,
            "instant": instant,
            "iou": iou,
            "geom": geom,
            "approach": approach,
            "motion": motion_change,
            "center_dist": center_dist,
            "timestamp": a.get("timestamp"),
        })

        # Optional debug reporting for tuning
        if self.debug and (
            iou >= self.min_iou
            or geom >= self.report_threshold
            or instant >= self.report_threshold
        ):
            self._print_pair_debug(
                frame_id, a, b, iou, cover, center_dist,
                normalized_dist, approach, motion_change, instant,
            )

        # ------------------------------------------------
        # Confirmation logic
        # ------------------------------------------------
        event = self._confirm_pair(
            pair_key,
            frame_id,
            paired,
            iou,
            cover,
            center_dist,
            containment,
        )

        if event is not None:
            return [event]

        return []

    # ============================================================
    # CONFIRMATION (temporal persistence + debounce)
    # ============================================================

    def _confirm_pair(
        self,
        pair_key: Tuple[int, int],
        frame_id: int,
        paired: dict,
        current_iou: float,
        current_cover: float,
        current_center_dist: float,
        containment: bool,
    ) -> dict | None:
        window = self._pair_window[pair_key]

        if len(window) < self.min_persist_frames:
            return None

        peak_instant = max(item["instant"] for item in window)
        geom_max = max(item["geom"] for item in window)
        iou_max = max(item["iou"] for item in window)
        iou_min = min(item["iou"] for item in window)
        approach_max = max(item["approach"] for item in window)
        motion_max = max(item["motion"] for item in window)

        n_above_sustained = sum(
            1 for item in window
            if item["instant"] >= self.confirm_sustained
        )

        # 1. Strong contact evidence: boxes actually overlap.
        strong_contact = (
            iou_max >= self.confirm_min_iou
            and geom_max >= self.confirm_min_iou
        )

        # 2. Temporal persistence: either one hard-impact peak, or at least
        #    min_persist_frames of near/above sustained contact.
        peak_ok = peak_instant >= self.confirm_peak and strong_contact
        sustained_ok = (
            n_above_sustained >= self.min_persist_frames
            and strong_contact
            and iou_max >= self.confirm_min_iou + 0.03
        )
        temporal_ok = peak_ok or sustained_ok

        if not temporal_ok:
            return None

        # 3. Episode check: a real collision is a TRANSIENT overlap. Pairs
        #    that are overlapping for a long, continuous stretch are normal
        #    congested/already-stuck traffic, not a fresh crash. Count how
        #    many consecutive frames (ending now) have real overlap.
        episode_length = 0
        for item in reversed(window):
            if item["iou"] >= self.overlap_episode_min:
                episode_length += 1
            else:
                break

        if episode_length > self.max_episode_frames:
            return None

        # 4. The two boxes must have been SEPARATED in the recent window:
        #    a real collision approaches from a gap, while co-moving traffic
        #    or a box that blinks into existence overlapping another car never
        #    shows a pre-contact separated state.
        if not any(
            item["iou"] < self.separated_threshold for item in window
        ):
            return None

        types = {paired[k]["type"] for k in paired}
        has_person = "person" in types

        # 3. Do NOT confirm on a single weak signal: require approaching
        #    behaviour OR impact-like motion change.
        if has_person:
            # Pedestrian cases need impact-like change OR a very strong
            # sustained contact with clear approach.
            motion_ok = (
                motion_max >= self.motion_min
                or (approach_max >= self.approach_min and geom_max >= 0.45)
            )
        else:
            motion_ok = (
                approach_max >= self.approach_min
                or motion_max >= self.motion_min
            )

        if not motion_ok:
            return None

        # 4. Debounce / cooldown.
        if frame_id - self._last_emitted.get(pair_key, -10 ** 9) \
                < self.cooldown_frames:
            return None

        # ------------------------------------------------
        # Build event
        # ------------------------------------------------
        ids = [paired[k] for k in paired]
        primary, secondary = self._order_pair(ids)

        confidence = min(
            1.0,
            round(
                0.45 * peak_instant
                + 0.30 * min(1.0, geom_max)
                + 0.25 * approach_max,
                2,
            ),
        )
        confidence = max(confidence, 0.45)

        self._last_emitted[pair_key] = frame_id

        peak_frame = next(
            (item["frame_id"] for item in window
             if item["instant"] == max(i["instant"] for i in window)),
            frame_id,
        )
        ts = None
        for item in window:
            if item["frame_id"] == peak_frame and item.get("timestamp"):
                ts = item["timestamp"]
                break

        event = {
            "incident_type": "collision",
            "vehicle_track_id": primary["track_id"],
            "vehicle_type": primary["type"],
            "other_track_id": secondary["track_id"],
            "other_type": secondary["type"],
            "frame_id": peak_frame,
            "timestamp": ts,
            "timestamp_seconds": (
                round(peak_frame / self.fps, 3) if self.fps > 0 else None
            ),
            "iou": round(iou_max, 3),
            "distance": round(current_center_dist, 2),
            "approach": round(approach_max, 3),
            "motion_change": round(motion_max, 3),
            "containment": bool(containment),
            "confidence": confidence,
            "gps": None,     # no GPS feed available -> honest null
        }
        return event

    # ============================================================
    # SIGNAL HELPERS
    # ============================================================

    def _reference_scale(self, box_a: BOX, box_b: BOX) -> float:
        """Half-sum of the two smaller box sides - a rough 'radius' scale."""
        wa = box_a[2] - box_a[0]
        ha = box_a[3] - box_a[1]
        wb = box_b[2] - box_b[0]
        hb = box_b[3] - box_b[1]
        return 0.5 * (min(wa, ha) + min(wb, hb))

    def _approach_evidence(self, id_a: int, id_b: int) -> float:
        """
        Fraction of the recent history steps where the two centres moved
        closer to each other. 0.5 means unknown/insufficient history so
        approach never confirms a collision by itself.
        """
        ha = list(self._positions.get(id_a, []))
        hb = list(self._positions.get(id_b, []))

        if len(ha) < 4 or len(hb) < 4:
            return 0.5

        # Align by frame id
        by_a = {h["frame_id"]: h["center"] for h in ha}
        by_b = {h["frame_id"]: h["center"] for h in hb}

        common_frames = sorted(set(by_a) & set(by_b))

        distances = []
        for f in common_frames:
            ca = by_a[f]
            cb = by_b[f]
            distances.append(
                math.hypot(ca[0] - cb[0], ca[1] - cb[1])
            )

        if len(distances) < 3:
            return 0.5

        closings = 0
        for k in range(1, len(distances)):
            if distances[k] < distances[k - 1] - 0.5:
                closings += 1

        return closings / (len(distances) - 1)

    def _motion_change(self, id_a: int, id_b: int) -> float:
        """
        Impact-like change, SCALE-FREE (independent of fps/video resolution).

        Uses the bottom-centre ("feet") point of each box (jitter-robust),
        smooths the position with a 3-frame moving average (kills single-frame
        YOLO box wiggles), then compares the recent per-frame speed with the
        object's own earlier speed.

        - Consistent speed  -> ratio ~1 -> score ~0
        - Sudden stop       -> ratio ~0 -> score ~0.8-0.9
        - Sudden burst      -> ratio >1 -> score grows up to 0.7
        - Already static    -> no meaningful impact -> score 0
        """
        scores = []
        for track_id in (id_a, id_b):
            hist = list(self._positions.get(track_id, []))
            feet = [
                (h["frame_id"], h["foot"])
                for h in hist
                if h.get("foot")
            ]
            if len(feet) < 6:
                scores.append(0.0)
                continue

            feet = feet[-8:]

            n = len(feet)
            smoothed = []
            for k in range(n):
                lo = max(0, k - 1)
                hi = min(n, k + 2)
                part = feet[lo:hi]
                pts = [p for _, p in part]
                px = sum(p[0] for p in pts) / len(pts)
                py = sum(p[1] for p in pts) / len(pts)
                smoothed.append((feet[k][0], (px, py)))

            steps = []
            for k in range(1, len(smoothed)):
                f0, p0 = smoothed[k - 1]
                f1, p1 = smoothed[k]
                df = max(1, f1 - f0)
                steps.append(
                    math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / df
                )

            if len(steps) < 4:
                scores.append(0.0)
                continue

            # Buffer of steps the object actually travelled recently.
            prev_avg = sum(steps[:-1]) / (len(steps) - 1)

            # Smooth the "current" speed over the last two steps so a single
            # outlying frame cannot fake a regime change.
            last = 0.5 * steps[-1] + 0.5 * steps[-2]

            if prev_avg < 0.35:
                # Object is essentially stationary: a "stop" is not an impact.
                scores.append(0.0)
                continue

            ratio = last / prev_avg

            if ratio > 1.0:
                score = min(0.7, (ratio - 1.0) * 0.6)
            else:
                # A sustained deceleration down to ~0 => a hard stop.
                score = min(0.9, (1.0 - ratio) * 1.2)

            scores.append(score)

        return max(scores)

    def _order_pair(self, objects: List[dict]):
        """
        Decide which object becomes vehicle_track_id.

        Rules:
            - When one is a pedestrian, the vehicle is ALWAYS primary.
            - When both are vehicles, a motorcycle is never preferred over
              a car/bus/truck; otherwise the faster one is chosen.
        """
        a, b = objects[0], objects[1]

        a_is_vehicle = a["type"] in VEHICLE_CLASSES
        b_is_vehicle = b["type"] in VEHICLE_CLASSES

        if not a_is_vehicle:
            return b, a
        if not b_is_vehicle:
            return a, b

        if a["type"] == "motorcycle" and b["type"] != "motorcycle":
            return b, a
        if b["type"] == "motorcycle" and a["type"] != "motorcycle":
            return a, b

        # Both non-motorcycle vehicles: faster one is primary.
        speed_a = self._last_displacement(a["track_id"])
        speed_b = self._last_displacement(b["track_id"])
        if speed_b > speed_a:
            return b, a
        return a, b

    def _last_displacement(self, track_id: int) -> float:
        hist = list(self._positions.get(track_id, []))
        if len(hist) < 2:
            return 0.0
        p0 = hist[-2]["center"]
        p1 = hist[-1]["center"]
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    @staticmethod
    def _pair_key(id_a: int, id_b: int) -> Tuple[int, int]:
        return (min(id_a, id_b), max(id_a, id_b))

    # ============================================================
    # DEBUG OUTPUT (used by test_collision.py)
    # ============================================================

    def summary(self, top_n: int = 15) -> List[dict]:
        """
        Aggregate per-pair statistics for tuning: strongest instantaneous
        score and best geometry/approach/motion per unique pair.

        Returns a list sorted by best instant score (descending).
        """
        rows = []
        for (id_a, id_b), window in self._pair_window.items():
            if not window:
                continue
            best = max(window, key=lambda item: item["instant"])
            rows.append({
                "pair": (id_a, id_b),
                "frames": len(window),
                "best_instant": round(best["instant"], 3),
                "best_iou": round(
                    max(item["iou"] for item in window), 3
                ),
                "best_geom": round(
                    max(item["geom"] for item in window), 3
                ),
                "best_approach": round(
                    max(item["approach"] for item in window), 3
                ),
                "best_motion": round(
                    max(item["motion"] for item in window), 3
                ),
            })
        rows.sort(key=lambda r: r["best_instant"], reverse=True)
        return rows[:top_n]

    def _print_pair_debug(
        self,
        frame_id,
        a,
        b,
        iou,
        cover,
        center_dist,
        normalized_dist,
        approach,
        motion_change,
        instant,
    ):
        print(
            "\n" + "-" * 70
            + f"\nFrame {frame_id} | potential contact"
            + "\n  Object A: track_id={} type={} bbox={}"
            .format(a["track_id"], a["type"], a["bbox"])
            + "\n  Object B: track_id={} type={} bbox={}"
            .format(b["track_id"], b["type"], b["bbox"])
            + "\n  distance: {:.1f}px (normalized {:.2f})"
            .format(center_dist, normalized_dist)
            + "\n  IoU: {:.3f} | cover: {:.3f}"
            .format(iou, cover)
            + "\n  approaching: {:.2f} | motion_change: {:.2f}"
            .format(approach, motion_change)
            + "\n  collision_score (instant): {:.3f}"
            .format(instant)
            + "\n" + "-" * 70
        )

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        self._positions.clear()
        self._pair_window.clear()
        self._last_emitted.clear()