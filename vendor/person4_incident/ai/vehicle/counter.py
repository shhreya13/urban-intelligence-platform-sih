"""
counter.py — Vehicle/person density and session counting.

Responsibilities:
    - Count objects currently visible in the camera frame.
    - Count unique tracked objects observed during the session.
    - Separate vehicles from persons.
    - Do NOT use line crossing.
    - Do NOT use a counting line.

Important:
    A BoT-SORT track ID represents a tracking identity during the
    current camera session. It is not guaranteed to represent a
    permanent real-world identity if an object disappears and later
    re-enters the scene.
"""

from dataclasses import dataclass
from typing import Dict, List, Set

from ai.vehicle.classes import (
    SUPPORTED_CLASSES,
    VEHICLE_CLASSES,
)
from ai.vehicle.tracker import TrackedObject


# ============================================================
# SNAPSHOT
# ============================================================

@dataclass
class CountSnapshot:
    """
    Statistics for the current camera session.
    """

    # --------------------------------------------------------
    # CURRENT FRAME
    # --------------------------------------------------------

    current_by_class: Dict[str, int]

    current_total: int

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    session_by_class: Dict[str, int]

    session_total: int

    # --------------------------------------------------------
    # CURRENT VEHICLE/PERSON DENSITY
    # --------------------------------------------------------

    vehicle_current: int

    person_current: int

    # --------------------------------------------------------
    # SESSION VEHICLE/PERSON TOTALS
    # --------------------------------------------------------

    vehicle_session: int

    person_session: int


# ============================================================
# OBJECT COUNTER
# ============================================================

class ObjectCounter:
    """
    Counts currently visible objects and unique tracked objects
    observed during the current camera session.

    There is NO line-crossing logic in this class.
    """

    def __init__(self) -> None:

        # ----------------------------------------------------
        # Track IDs already observed during this session.
        #
        # Structure:
        #
        # {
        #     "car": {1, 5, 8},
        #     "person": {2, 3},
        #     ...
        # }
        # ----------------------------------------------------

        self.seen_ids: Dict[str, Set[int]] = {
            cls: set()
            for cls in SUPPORTED_CLASSES
        }

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        tracked_objects: List[TrackedObject],
    ) -> CountSnapshot:

        """
        Update counts using the objects detected in the current frame.

        Current count:
            Number of objects visible RIGHT NOW.

        Session count:
            Number of unique BoT-SORT IDs observed during this
            camera session.
        """

        # ----------------------------------------------------
        # CURRENT FRAME COUNTS
        # ----------------------------------------------------

        current_by_class: Dict[str, int] = {
            cls: 0
            for cls in SUPPORTED_CLASSES
        }

        # Track IDs visible in this frame.
        current_ids: Dict[str, Set[int]] = {
            cls: set()
            for cls in SUPPORTED_CLASSES
        }

        # ----------------------------------------------------
        # PROCESS TRACKED OBJECTS
        # ----------------------------------------------------

        for obj in tracked_objects:

            object_type = obj.object_type

            # Ignore unexpected classes.
            if object_type not in current_by_class:
                continue

            track_id = int(obj.track_id)

            # ------------------------------------------------
            # CURRENT FRAME
            # ------------------------------------------------

            # Use the ID set so the same ID cannot be counted
            # twice in the same frame.
            if track_id not in current_ids[object_type]:

                current_ids[object_type].add(track_id)

                current_by_class[object_type] += 1

            # ------------------------------------------------
            # SESSION
            # ------------------------------------------------

            # Add this ID to the session history.
            #
            # If ID 7 appears in 500 frames, it is still counted
            # as ONE observed object.
            self.seen_ids[object_type].add(track_id)

        # ====================================================
        # CURRENT TOTAL
        # ====================================================

        current_total = sum(
            current_by_class.values()
        )

        # ====================================================
        # SESSION COUNTS
        # ====================================================

        session_by_class: Dict[str, int] = {
            cls: len(self.seen_ids[cls])
            for cls in SUPPORTED_CLASSES
        }

        session_total = sum(
            session_by_class.values()
        )

        # ====================================================
        # CURRENT VEHICLES
        # ====================================================

        vehicle_current = sum(
            current_by_class.get(cls, 0)
            for cls in VEHICLE_CLASSES
        )

        # ====================================================
        # CURRENT PERSONS
        # ====================================================

        person_current = current_by_class.get(
            "person",
            0,
        )

        # ====================================================
        # SESSION VEHICLES
        # ====================================================

        vehicle_session = sum(
            session_by_class.get(cls, 0)
            for cls in VEHICLE_CLASSES
        )

        # ====================================================
        # SESSION PERSONS
        # ====================================================

        person_session = session_by_class.get(
            "person",
            0,
        )

        # ====================================================
        # RETURN SNAPSHOT
        # ====================================================

        return CountSnapshot(

            current_by_class=current_by_class,

            current_total=current_total,

            session_by_class=session_by_class,

            session_total=session_total,

            vehicle_current=vehicle_current,

            person_current=person_current,

            vehicle_session=vehicle_session,

            person_session=person_session,
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:

        """
        Reset all session statistics.

        Useful if a new camera session starts without restarting
        the Python process.
        """

        self.seen_ids = {
            cls: set()
            for cls in SUPPORTED_CLASSES
        }