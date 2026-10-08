"""
Priority scoring (Module 6 responsibility 6).

    severity + persistence + traffic impact + safety risk  ->  priority score

Each of the four inputs is normalised to 0..1, combined with the configured
weights, and reported as 0..100 plus a band.  Every component is returned
alongside the score so the dashboard can explain *why* something is
priority 82 rather than just asserting it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.event import Severity

#: How dangerous each class is to a human being, independent of how often
#: it is seen.  0..1.
SAFETY_RISK: Dict[str, float] = {
    "pothole": 0.55,
    "damaged_road": 0.5,
    "road_damage_other": 0.35,
    "waterlogging": 0.7,
    "missing_divider": 0.75,
    "missing_zebra_crossing": 0.6,
    "damaged_zebra_crossing": 0.5,
    "zebra_crossing": 0.05,
    "divider": 0.05,
    "traffic_signboard": 0.05,
    "road_repair": 0.0,
    "vehicle_vehicle": 0.8,
    "person": 0.85,
}

EVENT_SAFETY_RISK: Dict[str, float] = {
    "hit_and_run": 1.0,
    "collision": 0.95,
    "possible_collision": 0.7,
    "abnormal_movement": 0.5,
    "rash_driving": 0.65,
    "traffic_congestion": 0.25,
    "traffic_bottleneck": 0.25,
    "road_defect": 0.4,
}

EVENT_TRAFFIC_IMPACT: Dict[str, float] = {
    "traffic_congestion": 0.9,
    "traffic_bottleneck": 0.85,
    "road_defect": 0.4,
    "collision": 0.8,
    "hit_and_run": 0.7,
    "possible_collision": 0.5,
    "abnormal_movement": 0.3,
    "rash_driving": 0.35,
}

DEFAULT_WEIGHTS = {
    "severity": 0.35,
    "persistence": 0.25,
    "traffic_impact": 0.20,
    "safety_risk": 0.20,
}


def band(score: float) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 55:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


def compute_priority(
    *,
    event_type: str,
    class_name: Optional[str],
    severity: Optional[str],
    observation_count: int = 1,
    bus_count: int = 1,
    age_days: float = 0.0,
    segment_stats: Optional[Dict[str, Any]] = None,
    incident_risk_at_segment: int = 0,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    segment_stats = segment_stats or {}

    # 1. Severity, 0..1
    severity_score = Severity.rank(severity) / 4.0

    # 2. Persistence: how often, by how many buses, for how long.
    observations = min(1.0, observation_count / 12.0)
    fleet = min(1.0, max(0, bus_count - 1) / 3.0)
    age = min(1.0, age_days / 14.0)
    persistence_score = min(1.0, 0.5 * observations + 0.3 * fleet + 0.2 * age)

    # 3. Traffic impact: base for the event type, lifted by what the traffic
    #    module actually measured on that road segment.
    traffic_score = EVENT_TRAFFIC_IMPACT.get(event_type, 0.3)
    density = float(segment_stats.get("avg_density_percentage") or 0.0) / 100.0
    congested = float(segment_stats.get("congested_ratio") or 0.0)
    traffic_score = min(1.0, traffic_score * (1.0 + 0.6 * density + 0.6 * congested))

    # 4. Safety risk
    safety_score = SAFETY_RISK.get(
        class_name or "", EVENT_SAFETY_RISK.get(event_type, 0.3)
    )
    safety_score = max(safety_score, EVENT_SAFETY_RISK.get(event_type, 0.0))
    if incident_risk_at_segment > 0:
        safety_score = min(1.0, safety_score + 0.15)

    total = 100.0 * (
        weights["severity"] * severity_score
        + weights["persistence"] * persistence_score
        + weights["traffic_impact"] * traffic_score
        + weights["safety_risk"] * safety_score
    )
    total = round(min(100.0, max(0.0, total)), 1)

    return {
        "priority_score": total,
        "priority_band": band(total),
        "components": {
            "severity": round(severity_score, 3),
            "persistence": round(persistence_score, 3),
            "traffic_impact": round(traffic_score, 3),
            "safety_risk": round(safety_score, 3),
        },
        "weights": weights,
    }
