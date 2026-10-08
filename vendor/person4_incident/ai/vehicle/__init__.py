"""
Person 1 module — Live-camera vehicle/person detection, BoT-SORT tracking,
and counting.

Public interface for other team members (mainly Person 3's Event Engine):

    from ai.vehicle import VehicleDetector, VehicleTracker, ObjectCounter

    detector = VehicleDetector()
    tracker = VehicleTracker(detector)
    counter = ObjectCounter(line=((0, 240), (640, 240)))

    tracked_objects = tracker.process_frame(frame)   # -> List[TrackedObject]
    snapshot = counter.update(tracked_objects)        # -> CountSnapshot

    for obj in tracked_objects:
        event_engine.handle(obj.to_dict())
"""

from ai.vehicle.detector import VehicleDetector, Detection, ModelLoadError
from ai.vehicle.tracker import VehicleTracker, TrackedObject
from ai.vehicle.counter import ObjectCounter, CountSnapshot

__all__ = [
    "VehicleDetector",
    "Detection",
    "ModelLoadError",
    "VehicleTracker",
    "TrackedObject",
    "ObjectCounter",
    "CountSnapshot",
]
