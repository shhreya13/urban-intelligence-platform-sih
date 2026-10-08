from collections import defaultdict, deque
import math


class MotionAnalyzer:
    """
    Turns raw per-frame bounding-box centers into speed / acceleration /
    lateral-movement estimates for the rash-driving detector.

    IMPORTANT -- why this file measures things over a WINDOW of frames
    instead of just the last two:

    The platform runs on a moving bus. Every vehicle's box drifts across
    the image purely because the bus itself is turning, accelerating, or
    vibrating -- not because the vehicle is doing anything unusual. On
    top of that, detector/tracker jitter moves a box's center by a few
    pixels between any two consecutive frames even when the real vehicle
    is driving in a perfectly straight line at a perfectly steady speed.

    A single-frame measurement cannot tell the difference between "one
    frame of jitter" and "a real, sudden manoeuvre". So every quantity
    below is smoothed over a short window (``smoothing_window`` frames,
    picked from the video's fps) before it is handed to the rash-driving
    detector. This does not by itself decide what counts as rash driving
    -- that logic, plus the requirement that abnormal readings persist
    for many frames, lives in rash_driving.py -- it just makes sure the
    numbers it receives reflect the vehicle's real trend rather than
    single-frame noise or the bus's own motion.
    """

    def __init__(self, max_history=30, fps=30.0, smoothing_window=None):
        self.fps = fps
        self.max_history = max_history

        # Window (in frames) used to smooth speed/direction/lateral
        # estimates. Defaults to roughly a sixth of a second, floored at
        # 3 frames, so it scales sensibly whether the source is 15fps or
        # 60fps instead of hard-coding a frame count tuned for one fps.
        self.smoothing_window = max(3, int(smoothing_window or round(fps / 6)))

        # Track ID -> deque of {"frame_id", "center", ...}
        self.history = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

        # Track ID -> deque of recently computed smoothed speeds, used to
        # measure acceleration over a window rather than a single
        # noisy frame-to-frame delta.
        self._speed_history = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

    def update(self, tracked_objects):
        """
        Receive tracked objects from BoT-SORT
        and analyze their movement.
        """

        results = []

        for obj in tracked_objects:

            # Only analyze vehicles
            if obj.object_type not in {
                "car",
                "motorcycle",
                "bus",
                "truck"
            }:
                continue

            x1, y1, x2, y2 = obj.bbox

            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            self.history[obj.track_id].append({
                "frame_id": obj.frame_id,
                "timestamp": obj.timestamp,
                "center": (center_x, center_y),
                "bbox": obj.bbox,
                "confidence": obj.confidence,
                "object_type": obj.object_type
            })

            speed = self.calculate_speed(obj.track_id)
            self._speed_history[obj.track_id].append(
                (obj.frame_id, speed)
            )

            direction = self.calculate_direction(obj.track_id)

            acceleration = self._calculate_acceleration(obj.track_id)

            lateral_movement = self._calculate_lateral_movement(
                obj.track_id
            )

            results.append({
                "track_id": obj.track_id,
                "object_type": obj.object_type,
                "center": (center_x, center_y),
                "speed_pixels_per_second": speed,
                "direction": direction,
                "acceleration": round(acceleration, 2),
                "lateral_movement": round(lateral_movement, 2),
            })

        return results

    def analyze(self, tracked_objects):
        """
        Alias of update().

        Some existing callers of this module use motion_analyzer.analyze()
        instead of motion_analyzer.update(). This alias keeps both the
        original update() name and the in-the-wild analyze() name working.
        """
        return self.update(tracked_objects)

    def _calculate_acceleration(self, track_id):
        """
        Acceleration in pixels/second^2, measured across the smoothing
        window rather than from a single previous-frame speed delta.

        The old approach (``(speed_now - speed_prev) * fps``) took two
        already-noisy instantaneous speed samples, subtracted them, and
        then MULTIPLIED the result by the fps (e.g. 30x). A couple of
        pixels of ordinary tracker jitter in one frame's speed reading
        was therefore enough to manufacture a huge, entirely fake
        acceleration spike. Measuring the change over several frames
        (and dividing by the real elapsed time between them) means one
        noisy frame can no longer dominate the result.
        """
        speed_history = self._speed_history[track_id]

        if len(speed_history) < 2 or self.fps <= 0:
            return 0.0

        window = min(self.smoothing_window, len(speed_history) - 1)
        older_frame_id, older_speed = speed_history[-1 - window]
        newer_frame_id, newer_speed = speed_history[-1]

        frame_gap = newer_frame_id - older_frame_id
        if frame_gap <= 0:
            return 0.0

        time_gap = frame_gap / self.fps
        return (newer_speed - older_speed) / time_gap

    def _calculate_lateral_movement(self, track_id):
        """
        Lateral (sideways) movement, in pixels, of the vehicle's most
        recent position relative to its own smoothed heading.

        Rather than bucketing the vehicle into "left/right/up/down" and
        picking |dx| or |dy| accordingly (which flips which axis counts
        as "lateral" every time that bucket flickers from a jittery
        frame), this projects the latest frame's displacement onto the
        axis perpendicular to the vehicle's average direction of travel
        over the smoothing window. That gives a continuous measure of
        "how far this vehicle strayed from its own recent path", which
        is stable under jitter and meaningful during real lane changes
        or swerves alike.
        """
        history = self.history[track_id]
        if len(history) < 2:
            return 0.0

        window = min(self.smoothing_window, len(history) - 1)
        older_center = history[-1 - window]["center"]
        newest_center = history[-1]["center"]
        prev_center = history[-2]["center"]

        heading_dx = newest_center[0] - older_center[0]
        heading_dy = newest_center[1] - older_center[1]
        heading_mag = math.hypot(heading_dx, heading_dy)

        if heading_mag < 1e-6:
            # No established heading over the window (near-stationary
            # vehicle) -- fall back to raw last-frame displacement
            # magnitude rather than a divide-by-zero.
            return math.hypot(
                newest_center[0] - prev_center[0],
                newest_center[1] - prev_center[1],
            )

        # Unit vector perpendicular to the smoothed heading.
        perp_x = -heading_dy / heading_mag
        perp_y = heading_dx / heading_mag

        step_dx = newest_center[0] - prev_center[0]
        step_dy = newest_center[1] - prev_center[1]

        lateral = abs(step_dx * perp_x + step_dy * perp_y)
        return lateral

    def calculate_speed(self, track_id):
        """
        Calculate speed in pixels/second, measured over the smoothing
        window (not just the last two frames) so a single jittery frame
        cannot swing the estimate on its own.
        """

        history = self.history[track_id]

        if len(history) < 2:
            return 0.0

        window = min(self.smoothing_window, len(history) - 1)
        previous = history[-1 - window]
        current = history[-1]

        x1, y1 = previous["center"]
        x2, y2 = current["center"]

        # Pixel distance travelled over the window
        distance = math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

        # Frame difference
        frame_difference = (
            current["frame_id"] -
            previous["frame_id"]
        )

        if frame_difference <= 0 or self.fps <= 0:
            return 0.0

        # Convert frames to seconds
        time_difference = frame_difference / self.fps

        speed = distance / time_difference

        return round(speed, 2)

    def calculate_direction(self, track_id):
        """
        Determine basic movement direction from the same smoothed
        window used for speed, instead of the raw last-frame delta, so
        the label doesn't flicker frame to frame on jitter alone.
        """

        history = self.history[track_id]

        if len(history) < 2:
            return "unknown"

        window = min(self.smoothing_window, len(history) - 1)
        previous = history[-1 - window]["center"]
        current = history[-1]["center"]

        dx = current[0] - previous[0]
        dy = current[1] - previous[1]

        # Scale the "stationary" threshold with the window so it still
        # means "didn't really move", regardless of how many frames the
        # window spans.
        threshold = 2 * max(1, window)

        if abs(dx) < threshold and abs(dy) < threshold:
            return "stationary"

        if abs(dx) > abs(dy):

            if dx > 0:
                return "right"

            return "left"

        else:

            if dy > 0:
                return "down"

            return "up"

    def get_history(self, track_id):
        return list(self.history.get(track_id, []))

    def get_all_history(self):
        return {
            track_id: list(history)
            for track_id, history in self.history.items()
        }
