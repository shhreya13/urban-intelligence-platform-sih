from collections import defaultdict, deque
from math import hypot
import time

class RiskEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hist = defaultdict(lambda: deque(maxlen=60))
        self.last_event = {}

    def update_position(self, key, x, y, frame_idx):
        h = self.hist[key]
        h.append((frame_idx, float(x), float(y)))
        return self._speed(h)

    @staticmethod
    def _speed(h):
        if len(h) < 2:
            return 0.0
        f0, x0, y0 = h[-2]
        f1, x1, y1 = h[-1]
        df = max(1, f1 - f0)
        return hypot(x1 - x0, y1 - y0) / df

    def closing_speed(self, pkey, vkey):
        ph = self.hist[pkey]
        vh = self.hist[vkey]
        if len(ph) < 2 or len(vh) < 2:
            return 0.0
        pf0, px0, py0 = ph[-2]
        pf1, px1, py1 = ph[-1]
        vf0, vx0, vy0 = vh[-2]
        vf1, vx1, vy1 = vh[-1]

        d0 = hypot(px0 - vx0, py0 - vy0)
        d1 = hypot(px1 - vx1, py1 - vy1)
        df = max(1, pf1 - pf0)
        return (d0 - d1) / df

    def score(self, *, distance, approaching, in_road, crossing,
              pedestrian_type, vehicle_speed):
        r = self.cfg["risk"]
        s = 0.0

        if distance <= r["dangerous_distance_px"]:
            s += r["close_bonus"]
        elif distance <= r["close_distance_px"]:
            s += r["close_bonus"] * 0.55

        if approaching:
            s += r["approach_bonus"]

        if in_road:
            s += r["road_bonus"]

        if crossing:
            s += r["crossing_bonus"]

        if vehicle_speed >= r["vehicle_speed_px_per_frame"]:
            s += 10

        if pedestrian_type == "child":
            s *= r["child_multiplier"]

        s = min(100.0, s)

        t = r["thresholds"]
        if s <= t["low_max"]:
            level = "LOW"
        elif s <= t["medium_max"]:
            level = "MEDIUM"
        elif s <= t["high_max"]:
            level = "HIGH"
        else:
            level = "CRITICAL"

        return round(s, 1), level

    def should_emit(self, pair_key, cooldown):
        now = time.time()
        if now - self.last_event.get(pair_key, -1e9) >= cooldown:
            self.last_event[pair_key] = now
            return True
        return False
