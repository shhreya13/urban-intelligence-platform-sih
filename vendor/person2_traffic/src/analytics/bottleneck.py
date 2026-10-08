import time


class BottleneckAnalyzer:
    """
    Detects traffic bottlenecks using spatial zones.

    A zone becomes a bottleneck when:
    1. It contains many vehicles.
    2. Vehicles inside that zone are moving slowly.
    3. The condition remains for a short period.

    Speed is currently relative pixels/second.
    Physical speed requires camera calibration.
    """

    def __init__(
        self,
        density_threshold=0.60,
        speed_threshold=15.0,
        minimum_duration_seconds=10,
        rows=3,
        columns=3,
        max_vehicles_per_zone=8
    ):

        self.density_threshold = density_threshold

        self.speed_threshold = speed_threshold

        self.minimum_duration_seconds = (
            minimum_duration_seconds
        )

        self.rows = rows
        self.columns = columns

        self.max_vehicles_per_zone = (
            max_vehicles_per_zone
        )

        self.zone_start_times = {}

    def _get_zone(
        self,
        center_x,
        center_y,
        frame_width,
        frame_height
    ):

        column_width = (
            frame_width / self.columns
        )

        row_height = (
            frame_height / self.rows
        )

        column = int(
            center_x / column_width
        )

        row = int(
            center_y / row_height
        )

        # Keep values inside grid
        column = min(
            max(column, 0),
            self.columns - 1
        )

        row = min(
            max(row, 0),
            self.rows - 1
        )

        return row, column

    def _zone_name(
        self,
        row,
        column
    ):

        row_names = [
            "TOP",
            "MIDDLE",
            "BOTTOM"
        ]

        column_names = [
            "LEFT",
            "CENTER",
            "RIGHT"
        ]

        row_name = (
            row_names[row]
            if row < len(row_names)
            else f"ROW-{row + 1}"
        )

        column_name = (
            column_names[column]
            if column < len(column_names)
            else f"COL-{column + 1}"
        )

        return f"{row_name}-{column_name}"

    def analyze(
        self,
        detections,
        frame_width,
        frame_height
    ):

        current_time = time.time()

        # Create empty zones
        zones = {}

        for row in range(self.rows):

            for column in range(self.columns):

                zone_name = self._zone_name(
                    row,
                    column
                )

                zones[zone_name] = {
                    "row": row,
                    "column": column,
                    "vehicle_count": 0,
                    "speeds": [],
                    "average_speed": 0.0,
                    "density": 0.0,
                    "bottleneck": False,
                    "duration": 0.0
                }

        # ---------------------------------------------
        # Assign vehicles to zones
        # ---------------------------------------------

        for detection in detections:

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            center_x = (
                (x1 + x2) / 2
            )

            center_y = (
                (y1 + y2) / 2
            )

            row, column = self._get_zone(
                center_x,
                center_y,
                frame_width,
                frame_height
            )

            zone_name = self._zone_name(
                row,
                column
            )

            zones[zone_name][
                "vehicle_count"
            ] += 1

            speed = detection.get(
                "speed",
                0.0
            )

            zones[zone_name][
                "speeds"
            ].append(speed)

        # ---------------------------------------------
        # Analyze each zone
        # ---------------------------------------------

        bottleneck_zones = []

        for zone_name, zone in zones.items():

            vehicle_count = (
                zone["vehicle_count"]
            )

            speeds = zone["speeds"]

            # Average speed
            if speeds:

                average_speed = (
                    sum(speeds)
                    / len(speeds)
                )

            else:

                average_speed = 0.0

            zone["average_speed"] = (
                average_speed
            )

            # Zone density
            density = min(
                vehicle_count
                / self.max_vehicles_per_zone,
                1.0
            )

            zone["density"] = density

            # Conditions
            high_density = (
                density
                >= self.density_threshold
            )

            low_speed = (
                vehicle_count > 0
                and average_speed
                <= self.speed_threshold
            )

            bottleneck_condition = (
                high_density
                and low_speed
            )

            # -----------------------------------------
            # Start / continue timer
            # -----------------------------------------

            if bottleneck_condition:

                if zone_name not in (
                    self.zone_start_times
                ):

                    self.zone_start_times[
                        zone_name
                    ] = current_time

                duration = (
                    current_time
                    - self.zone_start_times[
                        zone_name
                    ]
                )

                zone["duration"] = duration

                if (
                    duration
                    >= self.minimum_duration_seconds
                ):

                    zone["bottleneck"] = True

                    bottleneck_zones.append(
                        zone_name
                    )

            else:

                # Reset timer
                if zone_name in (
                    self.zone_start_times
                ):

                    del self.zone_start_times[
                        zone_name
                    ]

                zone["duration"] = 0.0

                zone["bottleneck"] = False

        # ---------------------------------------------
        # Find strongest bottleneck
        # ---------------------------------------------

        strongest_zone = None

        strongest_score = 0.0

        for zone_name, zone in zones.items():

            score = (
                zone["density"]
                *
                (
                    1.0
                    -
                    min(
                        zone["average_speed"]
                        /
                        max(
                            self.speed_threshold,
                            1.0
                        ),
                        1.0
                    )
                )
            )

            if score > strongest_score:

                strongest_score = score

                strongest_zone = zone_name

        # ---------------------------------------------
        # Result
        # ---------------------------------------------

        if bottleneck_zones:

            status = "BOTTLENECK"

        elif strongest_zone is not None:

            strongest = zones[
                strongest_zone
            ]

            if (
                strongest["density"]
                >= self.density_threshold
            ):

                status = "BUILDING"

            else:

                status = "NORMAL"

        else:

            status = "NORMAL"

        return {
            "status": status,
            "zones": zones,
            "bottleneck_zones": (
                bottleneck_zones
            ),
            "strongest_zone": strongest_zone,
            "strongest_score": (
                strongest_score
            )
        }