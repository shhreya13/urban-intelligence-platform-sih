"""
backend.rbac
============

Roles, departments and what each of them may see or do.  This is the ONE
place that answers "can this user see this event type / open this page /
act on this case".  The web dashboard and the mobile app both read the
result through ``GET /auth/me`` -- neither hard-codes its own rules -- and
every API endpoint enforces the same tables server-side, so changing a URL
by hand never reveals another department's data.

Vocabulary
----------
role        what the user IS  (six values, including FIELD_OFFICER)
department  which unit they BELONG to (five values; a field officer belongs
            to one of the four operating departments)
visibility  which event types a department may read
ownership   which department is responsible for acting on a case
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Set

# --------------------------------------------------------------------- roles

CENTRAL_COMMAND = "CENTRAL_COMMAND"
TRAFFIC_DEPARTMENT = "TRAFFIC_DEPARTMENT"
ROADS_INFRASTRUCTURE = "ROADS_INFRASTRUCTURE"
PUBLIC_SAFETY = "PUBLIC_SAFETY"
TRANSPORT_AUTHORITY = "TRANSPORT_AUTHORITY"
FIELD_OFFICER = "FIELD_OFFICER"

ROLES: List[str] = [
    CENTRAL_COMMAND, TRAFFIC_DEPARTMENT, ROADS_INFRASTRUCTURE,
    PUBLIC_SAFETY, TRANSPORT_AUTHORITY, FIELD_OFFICER,
]
DEPARTMENTS: List[str] = [
    CENTRAL_COMMAND, TRAFFIC_DEPARTMENT, ROADS_INFRASTRUCTURE,
    PUBLIC_SAFETY, TRANSPORT_AUTHORITY,
]
OPERATING_DEPARTMENTS: List[str] = DEPARTMENTS[1:]

ROLE_LABEL: Dict[str, str] = {
    CENTRAL_COMMAND: "Central Command",
    TRAFFIC_DEPARTMENT: "Traffic Department",
    ROADS_INFRASTRUCTURE: "Roads & Infrastructure",
    PUBLIC_SAFETY: "Public Safety",
    TRANSPORT_AUTHORITY: "Transport Authority",
    FIELD_OFFICER: "Field Officer",
}
DEPARTMENT_LABEL: Dict[str, str] = {d: ROLE_LABEL[d] for d in DEPARTMENTS}

# ----------------------------------------------------------- event taxonomy

ROAD_TYPES = {"road_defect"}
TRAFFIC_TYPES = {"traffic_observation", "traffic_congestion", "traffic_bottleneck"}
TRAFFIC_INCIDENT_TYPES = {"rash_driving", "collision", "possible_collision"}
SAFETY_TYPES = {
    "hit_and_run", "collision", "possible_collision", "abnormal_movement",
    "pedestrian_vehicle_risk", "rash_driving",
}
ANPR_TYPES = {"vehicle_identification"}
ENGINE_TYPES = {"persistent_issue"}

ALL_TYPES: FrozenSet[str] = frozenset(
    ROAD_TYPES | TRAFFIC_TYPES | SAFETY_TYPES | ANPR_TYPES | ENGINE_TYPES
)

#: Road-model classes that describe HEALTHY infrastructure.  They are an
#: asset inventory, never a case.  Mirrors intelligence.INFRASTRUCTURE_OK.
INFRASTRUCTURE_OK = {"zebra_crossing", "divider", "traffic_signboard", "road_repair"}

#: department -> event types it may read (CENTRAL_COMMAND sees everything).
VISIBLE_TYPES: Dict[str, Set[str]] = {
    TRAFFIC_DEPARTMENT: TRAFFIC_TYPES | TRAFFIC_INCIDENT_TYPES,
    ROADS_INFRASTRUCTURE: set(ROAD_TYPES),
    PUBLIC_SAFETY: SAFETY_TYPES | ANPR_TYPES,
    # Transport watches how traffic and incidents affect its routes.  It gets
    # incident records read-only and with registration numbers redacted.
    TRANSPORT_AUTHORITY: TRAFFIC_TYPES | (SAFETY_TYPES - {"pedestrian_vehicle_risk"}),
}

#: event_type -> department responsible for acting on the case.
OWNER_BY_TYPE: Dict[str, str] = {
    "road_defect": ROADS_INFRASTRUCTURE,
    "traffic_congestion": TRAFFIC_DEPARTMENT,
    "traffic_bottleneck": TRAFFIC_DEPARTMENT,
    "rash_driving": TRAFFIC_DEPARTMENT,
    "collision": PUBLIC_SAFETY,
    "possible_collision": PUBLIC_SAFETY,
    "abnormal_movement": PUBLIC_SAFETY,
    "hit_and_run": PUBLIC_SAFETY,
    "pedestrian_vehicle_risk": PUBLIC_SAFETY,
}

#: Evidence sub-folders (``outputs/evidence/<folder>/...``) -> event type,
#: so ``GET /evidence/<path>`` can be scoped like the events themselves.
EVIDENCE_FOLDER_TYPE: Dict[str, str] = {
    "road_defect": "road_defect",
    "pedestrian_risk": "pedestrian_vehicle_risk",
    "vulnerable_pedestrian": "pedestrian_vehicle_risk",
    "pedestrian_vehicle_risk": "pedestrian_vehicle_risk",
    "anpr": "vehicle_identification",
    "vehicle_identification": "vehicle_identification",
    "hit_and_run": "hit_and_run",
    "collision": "collision",
    "possible_collision": "possible_collision",
    "abnormal_movement": "abnormal_movement",
    "rash_driving": "rash_driving",
    "incident": "collision",
}


def owner_department(event_type: Optional[str], class_name: Optional[str] = None) -> Optional[str]:
    """Which department acts on this, or None when it is not actionable."""
    if event_type == "road_defect" and class_name in INFRASTRUCTURE_OK:
        return None
    return OWNER_BY_TYPE.get(event_type or "")


# ------------------------------------------------------------------- pages
#
# Page ids are shared with the web router (frontend/index.html ROUTES).  The
# existing Central Command pages keep their original ids.

PAGES: Dict[str, List[str]] = {
    CENTRAL_COMMAND: [
        "dashboard", "live", "map", "events", "incidents", "hitrun", "anpr",
        "evidence", "notifications", "traffic", "road", "issues", "analytics",
        "fleet", "users",
    ],
    TRAFFIC_DEPARTMENT: [
        "traffic_home", "traffic", "events", "map", "evidence", "notifications",
    ],
    ROADS_INFRASTRUCTURE: [
        "roads_home", "road", "issues", "events", "map", "evidence", "notifications",
    ],
    PUBLIC_SAFETY: [
        "safety_home", "incidents", "hitrun", "anpr", "events", "map",
        "evidence", "notifications",
    ],
    TRANSPORT_AUTHORITY: [
        "transport_home", "fleet", "events", "map", "notifications",
    ],
    FIELD_OFFICER: ["field_home", "events", "map", "evidence", "notifications"],
}

HOME_PAGE: Dict[str, str] = {r: PAGES[r][0] for r in ROLES}

#: GIS layers each department may switch on, and which are on by default.
MAP_LAYERS = [
    "buses", "potholes", "damaged_roads", "waterlogging", "traffic",
    "incidents", "anpr", "road_assets", "persistent_issues",
]
LAYERS_ALLOWED: Dict[str, List[str]] = {
    CENTRAL_COMMAND: list(MAP_LAYERS),
    TRAFFIC_DEPARTMENT: ["traffic", "incidents", "buses"],
    ROADS_INFRASTRUCTURE: ["potholes", "damaged_roads", "waterlogging", "road_assets", "persistent_issues"],
    PUBLIC_SAFETY: ["incidents", "anpr", "buses"],
    TRANSPORT_AUTHORITY: ["buses", "traffic", "incidents"],
}
LAYERS_DEFAULT: Dict[str, List[str]] = {
    CENTRAL_COMMAND: list(MAP_LAYERS),
    TRAFFIC_DEPARTMENT: ["traffic", "incidents"],
    ROADS_INFRASTRUCTURE: ["potholes", "damaged_roads", "waterlogging", "road_assets"],
    PUBLIC_SAFETY: ["incidents", "anpr"],
    TRANSPORT_AUTHORITY: ["buses", "traffic"],
}


# ------------------------------------------------------------------- scope

@dataclass(frozen=True)
class Scope:
    """Everything the API needs to know to filter one user's view."""

    user_id: str
    role: str
    department: str
    can_manage_users: bool = False

    # -- identity helpers ---------------------------------------------------
    @property
    def is_central(self) -> bool:
        return self.role == CENTRAL_COMMAND

    @property
    def is_field(self) -> bool:
        return self.role == FIELD_OFFICER

    # -- visibility ---------------------------------------------------------
    @property
    def visible_types(self) -> Optional[Set[str]]:
        """None means 'everything' (Central Command)."""
        if self.is_central:
            return None
        return set(VISIBLE_TYPES.get(self.department, set()))

    def sees_type(self, event_type: Optional[str]) -> bool:
        types = self.visible_types
        return types is None or (event_type in types)

    def sees_any(self, event_types: Iterable[str]) -> bool:
        types = self.visible_types
        return types is None or bool(types & set(event_types))

    def restrict(self, requested: Optional[Iterable[str]] = None) -> Optional[List[str]]:
        """Intersect a requested type list with what the user may see.

        Returns None for 'no restriction', or a (possibly empty) list."""
        types = self.visible_types
        if requested is None:
            return None if types is None else sorted(types)
        requested = set(requested)
        return sorted(requested if types is None else requested & types)

    @property
    def can_view_plates(self) -> bool:
        return self.is_central or self.department == PUBLIC_SAFETY

    @property
    def pages(self) -> List[str]:
        return list(PAGES.get(self.role, []))

    def can_open(self, page: str) -> bool:
        return page in PAGES.get(self.role, [])

    @property
    def map_layers(self) -> List[str]:
        if self.is_central:
            return list(MAP_LAYERS)
        return list(LAYERS_ALLOWED.get(self.department, []))

    @property
    def default_layers(self) -> List[str]:
        if self.is_central:
            return list(MAP_LAYERS)
        return list(LAYERS_DEFAULT.get(self.department, []))

    # -- cases --------------------------------------------------------------
    def sees_case(self, case: Dict) -> bool:
        if self.is_central:
            return True
        if case.get("owner_department") == self.department:
            return True
        if self.is_field:
            return case.get("assignee_user_id") == self.user_id
        return self.sees_type(case.get("event_type"))

    def owns_case(self, case: Dict) -> bool:
        return self.is_central or case.get("owner_department") == self.department

    def as_dict(self) -> Dict:
        return {
            "role": self.role,
            "role_label": ROLE_LABEL.get(self.role, self.role),
            "department": self.department,
            "department_label": DEPARTMENT_LABEL.get(self.department, self.department),
            "home": HOME_PAGE.get(self.role),
            "pages": self.pages,
            "visible_event_types": sorted(self.visible_types) if self.visible_types is not None else sorted(ALL_TYPES),
            "can_view_plates": self.can_view_plates,
            "can_manage_users": self.can_manage_users,
            "map_layers": self.map_layers,
            "default_layers": self.default_layers,
        }


def valid_role_department(role: str, department: str) -> Optional[str]:
    """Return an error message, or None when the pair is allowed."""
    if role not in ROLES:
        return f"unknown role {role!r}"
    if department not in DEPARTMENTS:
        return f"unknown department {department!r}"
    if role == FIELD_OFFICER:
        if department not in OPERATING_DEPARTMENTS:
            return "a field officer must belong to an operating department"
        return None
    if role != department:
        return f"role {role} must belong to department {role}"
    return None
