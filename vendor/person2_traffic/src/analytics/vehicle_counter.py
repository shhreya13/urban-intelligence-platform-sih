class VehicleCounter:
    """
    Counts vehicles when their tracking point crosses
    a predefined horizontal counting line.

    Each tracking ID is counted only once per direction.
    """

    def __init__(self, line_start, line_end):
        self.line_start = tuple(line_start)
        self.line_end = tuple(line_end)

        # Previous center position of each track
        self.previous_positions = {}

        # IDs that have already been counted
        self.counted_ids = set()

        # Total counts
        self.counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }

        # Direction counts
        self.direction_counts = {
            "up": 0,
            "down": 0,
        }

    def get_center(self, bbox):
        """
        Get center point of bounding box.
        """

        x1, y1, x2, y2 = bbox

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return center_x, center_y

    def get_line_y(self):
        """
        Get Y coordinate of horizontal counting line.
        """

        y1 = self.line_start[1]
        y2 = self.line_end[1]

        return int((y1 + y2) / 2)

    def update(self, detections):
        """
        Update vehicle positions and detect line crossings.

        Returns:
            List of newly counted vehicles.
        """

        new_counts = []

        line_y = self.get_line_y()

        for detection in detections:

            track_id = detection["track_id"]

            # A vehicle without a tracking ID
            # cannot be reliably counted.
            if track_id is None:
                continue

            class_name = detection["class_name"]

            # Only count known vehicle classes
            if class_name not in self.counts:
                continue

            current_x, current_y = self.get_center(
                detection["bbox"]
            )

            # -----------------------------------------
            # First time seeing this track
            # -----------------------------------------

            if track_id not in self.previous_positions:

                self.previous_positions[track_id] = (
                    current_x,
                    current_y,
                )

                continue

            previous_x, previous_y = (
                self.previous_positions[track_id]
            )

            # -----------------------------------------
            # Determine which side of the line
            # the vehicle was on previously
            # -----------------------------------------

            previous_side = previous_y - line_y
            current_side = current_y - line_y

            # -----------------------------------------
            # Check whether line was crossed
            # -----------------------------------------

            crossed = (
                previous_side < 0
                and current_side >= 0
            ) or (
                previous_side > 0
                and current_side <= 0
            )

            if crossed and track_id not in self.counted_ids:

                # -------------------------------------
                # Determine direction
                # -------------------------------------

                if current_y > previous_y:
                    direction = "down"
                else:
                    direction = "up"

                # -------------------------------------
                # Update count
                # -------------------------------------

                self.counts[class_name] += 1

                self.direction_counts[direction] += 1

                self.counted_ids.add(track_id)

                # -------------------------------------
                # Save event
                # -------------------------------------

                count_event = {
                    "track_id": track_id,
                    "class_name": class_name,
                    "direction": direction,
                    "center": (
                        current_x,
                        current_y,
                    ),
                }

                new_counts.append(count_event)

            # -----------------------------------------
            # Update current position
            # -----------------------------------------

            self.previous_positions[track_id] = (
                current_x,
                current_y,
            )

        return new_counts

    def get_counts(self):
        """
        Return vehicle counts.
        """

        return self.counts.copy()

    def get_total(self):
        """
        Return total number of vehicles counted.
        """

        return sum(self.counts.values())

    def get_direction_counts(self):
        """
        Return direction counts.
        """

        return self.direction_counts.copy()

    def reset(self):
        """
        Reset all counting data.
        """

        self.previous_positions.clear()

        self.counted_ids.clear()

        self.counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }

        self.direction_counts = {
            "up": 0,
            "down": 0,
        }