import time


class CongestionAnalyzer:
    """
    Traffic congestion analyzer.

    Uses:
    - vehicle density
    - relative speed
    - persistence over time

    The current speed is relative pixels/second.
    Physical km/h requires camera calibration.
    """

    def __init__(
        self,
        density_threshold=0.40,
        speed_threshold=35.0,
        minimum_duration_seconds=8
    ):

        self.density_threshold = (
            density_threshold
        )

        self.speed_threshold = (
            speed_threshold
        )

        self.minimum_duration_seconds = (
            minimum_duration_seconds
        )

        self.congestion_start_time = None

    def update(
        self,
        density_percentage,
        average_speed
    ):

        current_time = time.time()

        # Convert percentage to 0-1
        density = (
            density_percentage / 100.0
        )

        # -------------------------------------------------
        # CONDITION 1
        # -------------------------------------------------
        # High vehicle density
        # -------------------------------------------------

        high_density = (
            density
            >= self.density_threshold
        )

        # -------------------------------------------------
        # CONDITION 2
        # -------------------------------------------------
        # Low relative speed
        # -------------------------------------------------

        low_speed = (
            average_speed
            <= self.speed_threshold
        )

        # -------------------------------------------------
        # CONGESTION LOGIC
        # -------------------------------------------------
        #
        # Strong congestion:
        # high density + low speed
        #
        # Building congestion:
        # moderate density + very low speed
        #
        # -------------------------------------------------

        strong_congestion = (
            high_density
            and low_speed
        )

        moderate_congestion = (
            density >= 0.25
            and average_speed <= 20.0
        )

        congestion_condition = (
            strong_congestion
            or moderate_congestion
        )

        # -------------------------------------------------
        # START TIMER
        # -------------------------------------------------

        if congestion_condition:

            if self.congestion_start_time is None:

                self.congestion_start_time = (
                    current_time
                )

            duration = (
                current_time
                - self.congestion_start_time
            )

        else:

            self.congestion_start_time = None

            duration = 0.0

        # -------------------------------------------------
        # FINAL STATUS
        # -------------------------------------------------

        if congestion_condition:

            if (
                duration
                >= self.minimum_duration_seconds
            ):

                status = "CONGESTED"

            else:

                status = "BUILDING"

        else:

            status = "NORMAL"

        return {
            "status": status,

            "is_congested": (
                status == "CONGESTED"
            ),

            "duration_seconds": duration,

            "density_percentage": (
                density_percentage
            ),

            "average_speed": (
                average_speed
            ),

            "high_density": (
                high_density
            ),

            "low_speed": (
                low_speed
            )
        }