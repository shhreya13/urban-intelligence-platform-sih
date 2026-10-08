"""
classes.py
Centralized vehicle/person detection class configuration.

This file must NOT import detector.py, tracker.py, counter.py,
or inference.py.
"""

from typing import Dict, List


# ============================================================
# COCO CLASS IDs
# ============================================================

COCO_CLASS_MAP: Dict[int, str] = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ============================================================
# SUPPORTED CLASSES
# ============================================================

SUPPORTED_CLASSES: List[str] = [
    "car",
    "motorcycle",
    "bus",
    "truck",
    "person",
]


# ============================================================
# YOLO CLASS IDs
# ============================================================

SUPPORTED_CLASS_IDS: List[int] = [
    0, 2, 3, 5, 7
]


# ============================================================
# VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES: List[str] = [
    "car",
    "motorcycle",
    "bus",
    "truck",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def class_name_from_id(class_id: int) -> str:
    """
    Convert COCO class ID to readable class name.
    """

    return COCO_CLASS_MAP.get(
        int(class_id),
        "unknown"
    )


def is_supported_class(class_id: int) -> bool:
    """
    Check whether a class ID is supported.
    """

    return int(class_id) in COCO_CLASS_MAP