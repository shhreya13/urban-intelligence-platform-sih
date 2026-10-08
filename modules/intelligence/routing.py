"""
Department routing table (Module 6 responsibility 7).

    Pothole            -> Roads/PWD
    Damaged road       -> Roads/PWD
    Congestion         -> Traffic
    Waterlogging       -> Drainage
    Bus delay          -> Transport
    Incidents          -> Traffic Police
    Pedestrian risk    -> Traffic Police

Kept as data, not code, so the routing can be re-pointed for a different
city's departments by editing ``config/platform.yaml`` -> ``departments``.
"""

from __future__ import annotations

from typing import Dict, Optional

ROADS_PWD = "Roads/PWD"
TRAFFIC = "Traffic"
DRAINAGE = "Drainage"
TRANSPORT = "Transport"
TRAFFIC_POLICE = "Traffic Police"
MUNICIPAL = "Municipal"

#: class_name -> department (checked first)
CLASS_ROUTES: Dict[str, str] = {
    "pothole": ROADS_PWD,
    "damaged_road": ROADS_PWD,
    "road_damage_other": ROADS_PWD,
    "road_repair": ROADS_PWD,
    "waterlogging": DRAINAGE,
    "missing_divider": ROADS_PWD,
    "divider": ROADS_PWD,
    "missing_zebra_crossing": TRAFFIC,
    "damaged_zebra_crossing": TRAFFIC,
    "zebra_crossing": TRAFFIC,
    "traffic_signboard": TRAFFIC,
    "bus_delay": TRANSPORT,
}

#: event_type -> department (fallback)
EVENT_ROUTES: Dict[str, str] = {
    "road_defect": ROADS_PWD,
    "traffic_congestion": TRAFFIC,
    "traffic_bottleneck": TRAFFIC,
    "traffic_observation": TRAFFIC,
    "pedestrian_vehicle_risk": TRAFFIC_POLICE,
    "rash_driving": TRAFFIC_POLICE,
    "collision": TRAFFIC_POLICE,
    "possible_collision": TRAFFIC_POLICE,
    "abnormal_movement": TRAFFIC_POLICE,
    "hit_and_run": TRAFFIC_POLICE,
    "vehicle_identification": TRAFFIC_POLICE,
    "bus_delay": TRANSPORT,
}

DEPARTMENT_ROUTES = {"by_class": CLASS_ROUTES, "by_event_type": EVENT_ROUTES}


def route_department(
    event_type: str,
    class_name: Optional[str] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> str:
    overrides = overrides or {}
    if class_name and class_name in overrides:
        return overrides[class_name]
    if event_type in overrides:
        return overrides[event_type]
    if class_name and class_name in CLASS_ROUTES:
        return CLASS_ROUTES[class_name]
    return EVENT_ROUTES.get(event_type, MUNICIPAL)
