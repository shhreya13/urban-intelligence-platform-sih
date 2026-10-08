"""
Shared perception layer.

These are NOT new models.  They are thin adapters around the weights and the
detector code the team already produced, arranged so that each model runs
**once per frame** and every module reads the same detection lists from the
FrameContext.

    vehicle.py       -> Person 2's VehicleDetector (YOLO11 + BoT-SORT)
                        + Person 2's SpeedEstimator
                        (falls back to COCO 'person' if weights are missing)
    road_defect.py   -> Person 1's ROAD_DEFECT_MODEL weights
    waterlogging.py  -> the dedicated WATERLOGGING_MODEL weights
"""

from .vehicle import VehiclePerception
from .road_defect import RoadDefectPerception
from .waterlogging import WaterloggingPerception
from .pedestrian import PedestrianPerception

__all__ = [
    "VehiclePerception", "RoadDefectPerception", "WaterloggingPerception",
    "PedestrianPerception",
]
