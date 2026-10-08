import time
import math


class SpeedEstimator:
    """
    Estimates relative vehicle movement using tracked
    vehicle center points.

    IMPORTANT:
    This is NOT physical km/h.

    It reports a stabilized relative movement value
    in pixels/second.

    The estimator:
    - uses real elapsed time
    - smooths movement
    - rejects unrealistic jumps
    - ignores very short tracking histories
    """

    def __init__(
        self,
        smoothing_factor=0.7,
        max_reasonable_speed=150.0,
        minimum_movement=1.5
    ):

        self.smoothing_factor = (
            smoothing_factor
        )

        self.max_reasonable_speed = (
            max_reasonable_speed
        )

        self.minimum_movement = (
            minimum_movement
        )

        self.track_history = {}

    def update(self, detections):

        current_time = time.time()

        updated_detections = []

        active_ids = set()

        for detection in detections:

            track_id = detection.get(
                "track_id"
            )

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            center_x = (
                (x1 + x2) / 2.0
            )

            center_y = (
                (y1 + y2) / 2.0
            )

            # Default speed
            speed = 0.0

            # -------------------------------------------------
            # TRACKED OBJECT
            # -------------------------------------------------

            if track_id is not None:

                active_ids.add(track_id)

                if track_id in self.track_history:

                    previous = (
                        self.track_history[
                            track_id
                        ]
                    )

                    previous_x = previous[
                        "x"
                    ]

                    previous_y = previous[
                        "y"
                    ]

                    previous_time = previous[
                        "time"
                    ]

                    previous_speed = previous[
                        "speed"
                    ]

                    dt = (
                        current_time
                        - previous_time
                    )

                    # Prevent invalid time
                    if dt > 0.001:

                        dx = (
                            center_x
                            - previous_x
                        )

                        dy = (
                            center_y
                            - previous_y
                        )

                        displacement = math.sqrt(
                            dx * dx
                            +
                            dy * dy
                        )

                        raw_speed = (
                            displacement
                            / dt
                        )

                        # -------------------------------------------------
                        # REMOVE VERY SMALL JITTER
                        # -------------------------------------------------

                        if (
                            displacement
                            < self.minimum_movement
                        ):

                            raw_speed = 0.0

                        # -------------------------------------------------
                        # REMOVE UNREALISTIC TRACK JUMPS
                        # -------------------------------------------------

                        if (
                            raw_speed
                            > self.max_reasonable_speed
                        ):

                            raw_speed = (
                                previous_speed
                            )

                        # -------------------------------------------------
                        # SMOOTH SPEED
                        # -------------------------------------------------

                        speed = (
                            self.smoothing_factor
                            * previous_speed
                            +
                            (
                                1.0
                                - self.smoothing_factor
                            )
                            * raw_speed
                        )

                    else:

                        speed = (
                            previous_speed
                        )

                # -------------------------------------------------
                # STORE CURRENT POSITION
                # -------------------------------------------------

                self.track_history[
                    track_id
                ] = {

                    "x": center_x,

                    "y": center_y,

                    "time": current_time,

                    "speed": speed
                }

            detection["speed"] = speed

            updated_detections.append(
                detection
            )

        # -------------------------------------------------
        # REMOVE OLD TRACKS
        # -------------------------------------------------

        old_ids = list(
            self.track_history.keys()
        )

        for track_id in old_ids:

            if track_id not in active_ids:

                previous = (
                    self.track_history[
                        track_id
                    ]
                )

                # Keep track briefly instead of
                # deleting immediately.
                if (
                    current_time
                    - previous["time"]
                    > 2.0
                ):

                    del self.track_history[
                        track_id
                    ]

        return updated_detections

    def average_speed(self, detections):

        speeds = []

        for detection in detections:

            speed = detection.get(
                "speed",
                0.0
            )

            # Only valid positive speeds
            if (
                speed >= 0
                and speed <= self.max_reasonable_speed
            ):

                speeds.append(speed)

        if not speeds:

            return 0.0

        # -------------------------------------------------
        # REMOVE EXTREME OUTLIERS
        # -------------------------------------------------

        speeds.sort()

        # If enough values exist, remove
        # the highest and lowest 10%.
        if len(speeds) >= 5:

            trim = max(
                1,
                int(len(speeds) * 0.10)
            )

            if (
                len(speeds)
                > trim * 2
            ):

                speeds = speeds[
                    trim:-trim
                ]

        return (
            sum(speeds)
            / len(speeds)
        )

    def moving_vehicle_count(
        self,
        detections,
        minimum_speed=5.0
    ):

        count = 0

        for detection in detections:

            speed = detection.get(
                "speed",
                0.0
            )

            if speed >= minimum_speed:

                count += 1

        return count

    def slow_vehicle_count(
        self,
        detections,
        maximum_speed=5.0
    ):

        count = 0

        for detection in detections:

            speed = detection.get(
                "speed",
                0.0
            )

            if speed <= maximum_speed:

                count += 1

        return count

    def stopped_vehicle_count(
        self,
        detections,
        maximum_speed=2.0
    ):

        count = 0

        for detection in detections:

            speed = detection.get(
                "speed",
                0.0
            )

            if speed <= maximum_speed:

                count += 1

        return count