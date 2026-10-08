"""
test_hit_and_run_synthetic.py
=============================

Deterministic validation of HitAndRunDetector decision logic.

Scenarios (each simulated with synthetic TrackedObject streams):

    A. COLLISION + VEHICLE ESCAPES   -> exactly 1 hit-and-run
       offender hits a victim, keeps driving away, victim brakes/stops.

    B. COLLISION + VEHICLE STOPS     -> 0 hit-and-runs
       offender brakes at the scene (no flee movement).

    C. COLLISION + VEHICLE VANISHES  -> 0 hit-and-runs
       the locked track disappears with NO movement away from the scene
       (occlusion / id-loss), which must NEVER be treated as escape.

    D. DEDUP                         -> still exactly 1 event when the
       scene keeps streaming long after the escape is confirmed.

Run:  python test_hit_and_run_synthetic.py
"""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from ai.vehicle.tracker import TrackedObject  # noqa: E402
from hit_and_run import HitAndRunDetector     # noqa: E402


def box(id_x, obj_type, x1, y1, x2, y2, frame):
    return TrackedObject(
        object_type=obj_type,
        track_id=id_x,
        confidence=0.9,
        bbox=(x1, y1, x2, y2),
        frame_id=frame,
        timestamp="2026-01-01T00:00:00.000",
    )


class Sim:
    def __init__(self, track_id, obj_type, box0, vel):
        self.track_id = track_id
        self.obj_type = obj_type
        self.box = list(box0)
        self.vel = list(vel)

    def step(self, vx=None, vy=None):
        if vx is not None:
            self.vel = [vx, vy]
        self.box[0] += self.vel[0]
        self.box[1] += self.vel[1]
        self.box[2] += self.vel[0]
        self.box[3] += self.vel[1]

    def make(self, frame):
        bx = [int(round(v)) for v in self.box]
        return box(self.track_id, self.obj_type, bx[0], bx[1], bx[2], bx[3], frame)


def iou_of(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(0.01, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def fake_collision(f, victim_braked=True):
    return {
        "incident_type": "collision",
        "vehicle_track_id": 1,
        "vehicle_type": "car",
        "other_track_id": 2,
        "other_type": "car",
        "frame_id": f,
        "timestamp": "2026-01-01T00:00:00.000",
        "timestamp_seconds": f / 30.0,
        "iou": 0.4,
        "distance": 30.0,
        "approach": 0.9,
        "motion_change": 0.8,
        "containment": False,
        "confidence": 0.8,
    }


def run(behaviour, collide_frame, frames=70):
    """behaviour: 'escape' | 'stop' | 'vanish'"""
    offender = Sim(1, "car", (40, 100, 100, 135), (10, 0))
    victim = Sim(2, "car", (200, 100, 270, 135), (6, 0))

    det = HitAndRunDetector(fps=30.0, debug=False)
    events = []
    delivered = False

    for f in range(1, frames + 1):
        if offender is not None:
            offender.step()
        victim.step()
        if f >= collide_frame + 2:
            victim.step(0, 0)  # victim brakes at the scene

        if behaviour == "escape":
            pass  # offender keeps driving away
        elif behaviour == "stop":
            if f >= collide_frame + 2 and offender is not None:
                offender.step(0, 0)
        elif behaviour == "vanish":
            if f >= collide_frame + 4:
                offender = None

        objs = []
        if offender is not None:
            objs.append(offender.make(f))
        objs.append(victim.make(f))

        collisions = []
        if not delivered and f >= collide_frame:
            collisions = [fake_collision(collide_frame)]
            delivered = True

        events.extend(det.update(objs, collisions))

    return events


def main():
    collide = 25
    results = {
        "escape (driver flees)": run("escape", collide),
        "stop (driver brakes at scene)": run("stop", collide),
        "vanish (track lost, no escape)": run("vanish", collide),
    }

    for name, evs in results.items():
        print(f"  {name}: {len(evs)} events")

    ok = (
        len(results["escape (driver flees)"]) == 1
        and len(results["stop (driver brakes at scene)"]) == 0
        and len(results["vanish (track lost, no escape)"]) == 0
    )

    # Dedup sanity: an escape must produce exactly one event total.
    if ok:
        ev = results["escape (driver flees)"][0]
        print(
            f"  escape event: track={ev['hit_vehicle_track_id']} "
            f"dist={ev['escape_distance_px']}px "
            f"speed={ev['escape_speed_px_s']}px/s "
            f"dir={ev['escape_direction']} conf={ev['confidence']}"
        )

    print("=" * 60)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())