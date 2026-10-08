"""
test_collision_synthetic.py
===========================

Deterministic, YOLO-free validation of CollisionDetector decision logic.

Feeds synthetic TrackedObject sequences into the detector and asserts the
desired behaviour:

    A. REAL COLLISION          -> CONFIRMED
       two cars converge from a gap, overlap briefly with a high peak,
       one decelerates hard, then they separate.

    B. CONGESTED CO-MOVEMENT   -> NOT confirmed
       two cars creep forward at constant speed while their boxes overlap
       for 60 frames. Classic gridlock -> must be rejected.

    C. NEAR MISS               -> NOT confirmed
       two fast cars pass close but boxes never overlap.

    D. DETECTION JITTER        -> NOT confirmed
       one track's box jumps around (no genuine overlap progression).

Run:  python test_collision_synthetic.py
"""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from ai.vehicle.tracker import TrackedObject  # noqa: E402
from collision_detection import CollisionDetector  # noqa: E402


def box(id_x, obj_type, x1, y1, x2, y2, frame):
    return TrackedObject(
        object_type=obj_type,
        track_id=id_x,
        confidence=0.9,
        bbox=(x1, y1, x2, y2),
        frame_id=frame,
        timestamp="2026-01-01T00:00:00.000",
    )


class TrackSim:
    """Helper: move a box by a velocity vector each frame."""

    def __init__(self, track_id, obj_type, box0, vel):
        self.track_id = track_id
        self.obj_type = obj_type
        self.box = list(box0)
        self.vel = list(vel)

    def step(self):
        self.box[0] += self.vel[0]
        self.box[1] += self.vel[1]
        self.box[2] += self.vel[0]
        self.box[3] += self.vel[1]

    def set_vel(self, vx, vy):
        self.vel = [vx, vy]

    def make(self, frame):
        bx = [int(round(v)) for v in self.box]
        return box(self.track_id, self.obj_type, bx[0], bx[1], bx[2], bx[3], frame)


def iou_of(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(0.01, area_a + area_b - inter)


def run_scenario(frames):
    det = CollisionDetector(fps=30.0, debug=False)
    events = []
    for frame_objs in frames:
        events.extend(det.update(frame_objs, frame_size=(600, 400)))
    return events


def scenario_a_real_collision():
    # Car A: 60x30 travelling +3 px/frame in x, ~6 px/frame before impact.
    # Car B: stationary car stopped ahead.
    a = TrackSim(1, "car", (100, 100, 160, 130), (4, 0))
    b = TrackSim(2, "car", (300, 90, 360, 125), (0, 0))

    objs = []
    for f in range(0, 60):
        a.step()
        objs.append([a.make(f), b.make(f)])

    # A rear-ends B around frame ~50 and must stop there.
    # (box A leading edge reaches B's box at x1<=300)
    print("  pre-impact overlap:", iou_of(a.box, b.box))

    det = CollisionDetector(fps=30.0, debug=False)
    events = []
    for ob in objs:
        events.extend(det.update(ob, frame_size=(600, 400)))
    return events


def run_frames(frames, name):
    det = CollisionDetector(fps=30.0, debug=False)
    events = []
    for frame_objs in frames:
        events.extend(det.update(frame_objs, frame_size=(600, 400)))
    print(f"  {name}: {len(events)} events -> {[ e['vehicle_track_id'] for e in events]}")
    return events


def scenario_b_congestion():
    # Two cars in adjacent lanes, both moving slowly, permanently overlapping.
    a = TrackSim(1, "car", (200, 100, 280, 130), (0.5, 0))
    b = TrackSim(2, "car", (240, 95, 320, 132), (0.5, 0))
    frames = []
    for f in range(0, 80):
        a.step(); b.step()
        frames.append([a.make(f), b.make(f)])
    return run_frames(frames, "congestion co-move (overlap 80 frames)")


def scenario_c_near_miss():
    # One fast car crossing, never actually overlapping the other.
    a = TrackSim(1, "car", (50, 80, 120, 115), (9, 0))
    b = TrackSim(2, "car", (60, 150, 130, 185), (9, 0))
    frames = []
    for f in range(0, 60):
        a.step(); b.step()
        frames.append([a.make(f), b.make(f)])
    return run_frames(frames, "near miss (parallel fast, no overlap)")


def scenario_d_jitter():
    # Same car bouncing its box; no second object approaching.
    frames = []
    for f in range(0, 80):
        c = 120 + (7 if f % 6 < 3 else 0)
        frames.append([box(1, "car", c, 100, c + 90, 140, f)])
    return run_frames(frames, "single-object jitter (no partner)")


def scenario_a2_real_sideswipe():
    # Hit-and-run profile: offender (car 1) rear-ends a moving victim (car 2);
    # the victim brakes hard, the offender keeps driving away -> transient
    # overlap, offender separates, victim stops.
    a = TrackSim(1, "car", (40, 100, 100, 135), (10, 0))
    b = TrackSim(2, "car", (200, 100, 270, 135), (6, 0))
    frames = []
    impact = None
    for f in range(0, 55):
        a.step()
        b.step()
        if iou_of(a.box, b.box) > 0:
            impact = impact or f
        if impact is not None and f >= impact + 2:
            b.set_vel(0, 0)  # victim brakes after impact
        frames.append([a.make(f), b.make(f)])
    print(f"  side-swipe impact at frame {impact}")
    return run_frames(frames, "hit-and-run (offender keeps going, victim brakes)")


def main():
    results = {}
    names = {
        "sideswipe": scenario_a2_real_sideswipe,
        "congestion": scenario_b_congestion,
        "near_miss": scenario_c_near_miss,
        "jitter": scenario_d_jitter,
    }
    events = {k: fun() for k, fun in names.items()}

    # real rear-end needs a dedicated sequence (with penetration + stop)
    a = TrackSim(1, "car", (100, 100, 160, 130), (4, 0))
    b = TrackSim(2, "car", (300, 90, 360, 125), (0, 0))
    frames = []
    impact_frame = None
    for f in range(0, 52):
        a.step()
        if iou_of(a.box, b.box) > 0:
            impact_frame = impact_frame or f
        if impact_frame and f >= impact_frame + 5:
            a.set_vel(0, 0)  # hard brake after 5 frames of penetration
        frames.append([a.make(f), b.make(f)])
    print("  rear-end impact at frame", impact_frame)
    events["rear_end"] = run_frames(frames, "real rear-end (stops after penetration)")

    expected = {
        "rear_end": 1,
        "sideswipe": 1,
        "congestion": 0,
        "near_miss": 0,
        "jitter": 0,
    }

    ok = True
    print("\n" + "=" * 60)
    for k, want in expected.items():
        got = len(events[k])
        status = "PASS" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  {status}  {k:12s} expected={want}  got={got}")

    print("=" * 60)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())