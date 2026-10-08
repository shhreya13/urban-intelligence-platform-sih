from collections import defaultdict, deque


class RashDrivingDetector:
    """
    Rule-based rash driving detector.

    Uses:
    - Speed
    - Acceleration
    - Lateral movement (relative to the vehicle's own recent heading)

    Input comes from MotionAnalyzer.

    WHY THIS NO LONGER FIRES ON A SINGLE FRAME
    -------------------------------------------
    A moving vehicle -- even one driving perfectly normally -- will
    occasionally clear two of these thresholds on one noisy frame simply
    because of detector/tracker jitter or a bump in the video. The old
    version of this detector raised an incident the instant that
    happened. This version instead keeps a short rolling history of
    "was this frame abnormal?" per vehicle and only raises an incident
    once abnormal frames have PERSISTED for ``min_persist_frames`` out of
    the last ``window_frames`` -- i.e. sustained, repeated abnormal
    behaviour, not a one-off spike. It also refuses to raise an incident
    at all if the resulting confidence is below ``min_confidence``.

    Normal driving (including normal overtaking, lane changes, braking,
    and accelerating) is the default outcome. Rash driving is only
    raised when there is persistent, multi-signal, high confidence
    evidence.
    """

    def __init__(
        self,
        speed_threshold=250.0,
        acceleration_threshold=150.0,
        lateral_threshold=80.0,
        min_persist_frames=10,
        window_frames=None,
        min_confidence=0.5,
    ):
        self.speed_threshold = speed_threshold
        self.acceleration_threshold = acceleration_threshold
        self.lateral_threshold = lateral_threshold

        # Temporal validation: an abnormal-looking frame only counts once
        # it has repeated at least this many times within the trailing
        # window. Choose this based on the source fps (e.g. roughly a
        # third of a second of sustained abnormal behaviour) rather than
        # blindly reusing a value tuned for a different frame rate.
        self.min_persist_frames = max(1, int(min_persist_frames))
        self.window_frames = max(
            self.min_persist_frames,
            int(window_frames or self.min_persist_frames * 2),
        )
        self.min_confidence = float(min_confidence)

        # Per-vehicle rolling window of "was >=2 signals abnormal on this
        # frame?" used for the persistence check above.
        self._abnormal_history = defaultdict(
            lambda: deque(maxlen=self.window_frames)
        )
        # Tracks that have already had an incident raised for their
        # CURRENT abnormal episode -- cleared once the vehicle's
        # behaviour drops back under the persistence bar, so a later,
        # separate episode of abnormal driving can still be reported.
        self._reported = set()

        # Stores previous motion information for each vehicle
        self.previous_motion = defaultdict(dict)

    def analyze(self, motion_data):
        """
        Analyze motion data for one frame.

        motion_data should be a list like:

        [
            {
                "track_id": 5,
                "object_type": "car",
                "center": (320, 200),
                "speed_pixels_per_second": 300,
                "direction": "right",
                "acceleration": 180,
                "lateral_movement": 20
            }
        ]

        Returns a list of detected rash-driving incidents. Most frames,
        for most vehicles, this returns an empty list -- normal driving
        is the default outcome.
        """

        incidents = []
        seen_ids = set()

        for vehicle in motion_data:

            track_id = vehicle["track_id"]
            seen_ids.add(track_id)

            speed = vehicle.get("speed_pixels_per_second", 0.0)
            acceleration = abs(vehicle.get("acceleration", 0.0))
            lateral_movement = vehicle.get("lateral_movement", 0.0)

            direction = vehicle.get("direction", "unknown")

            reasons = []

            # --------------------------------
            # 1. HIGH SPEED
            # --------------------------------
            if speed >= self.speed_threshold:
                reasons.append("high_speed")

            # --------------------------------
            # 2. SUDDEN ACCELERATION
            # --------------------------------
            if acceleration >= self.acceleration_threshold:
                reasons.append("sudden_acceleration")

            # --------------------------------
            # 3. LARGE LATERAL MOVEMENT
            # --------------------------------
            if lateral_movement >= self.lateral_threshold:
                reasons.append("sudden_lateral_movement")

            # --------------------------------
            # 4. TEMPORAL VALIDATION
            # --------------------------------
            # We require at least TWO suspicious behaviours on a frame
            # before that frame counts toward the persistence window at
            # all -- a single weak signal never accumulates.
            frame_is_abnormal = len(reasons) >= 2
            history = self._abnormal_history[track_id]
            history.append(frame_is_abnormal)
            persisted_frames = sum(1 for flag in history if flag)

            if persisted_frames < self.min_persist_frames:
                # Not enough sustained evidence yet -- this vehicle is
                # still classified as normal driving. Also allow a fresh
                # episode to be reported later once behaviour normalizes.
                self._reported.discard(track_id)
                continue

            if not frame_is_abnormal:
                # The persistence window still has enough abnormal
                # frames in recent history, but *this* frame itself
                # settled back to normal -- don't open a new incident on
                # a normal frame.
                continue

            # --------------------------------
            # 5. DECIDE RASH DRIVING
            # --------------------------------
            confidence = self._calculate_confidence(
                speed,
                acceleration,
                lateral_movement,
                len(reasons),
            )

            if confidence < self.min_confidence:
                # Ambiguous/low-confidence movement -- do not create an
                # event, even though it technically persisted.
                continue

            if track_id in self._reported:
                # Already raised for this ongoing abnormal episode; the
                # caller/module layer owns re-reporting after a cooldown
                # if the behaviour continues.
                continue

            self._reported.add(track_id)

            incidents.append(
                {
                    "incident_type": "rash_driving",
                    "vehicle_track_id": track_id,
                    "vehicle_type": vehicle.get(
                        "object_type", "unknown"
                    ),
                    "direction": direction,
                    "speed_pixels_per_second": round(speed, 2),
                    "acceleration": round(acceleration, 2),
                    "lateral_movement": round(
                        lateral_movement, 2
                    ),
                    "confidence": round(confidence, 2),
                    "reasons": reasons,
                    "persisted_frames": persisted_frames,
                }
            )

        # Vehicles no longer seen this frame can't keep accumulating
        # persistence credit or a stale "already reported" flag.
        stale_ids = set(self._reported) - seen_ids
        for track_id in stale_ids:
            self._reported.discard(track_id)

        return incidents

    def _calculate_confidence(
        self,
        speed,
        acceleration,
        lateral_movement,
        number_of_reasons,
    ):
        """
        Calculate a simple explainable confidence score.
        """

        score = 0.0

        # Speed contribution
        if speed >= self.speed_threshold:
            score += 0.35

        # Acceleration contribution
        if acceleration >= self.acceleration_threshold:
            score += 0.30

        # Lateral movement contribution
        if lateral_movement >= self.lateral_threshold:
            score += 0.20

        # Multiple suspicious behaviours
        if number_of_reasons >= 3:
            score += 0.15

        return min(score, 1.0)

    def reset(self):
        """Reset stored tracking information."""
        self.previous_motion.clear()
        self._abnormal_history.clear()
        self._reported.clear()
