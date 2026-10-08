"""
core.config
===========

Centralised configuration (integration rule 11) with configurable paths
(rule 12).

Resolution order, highest priority first:

    1. CLI overrides passed to ``load_config(overrides=...)``
    2. Environment variables  ``UIP_<SECTION>__<KEY>``  (double underscore
       separates nesting), e.g. ``UIP_VIDEO__SOURCE=data/videos/demo.mp4``
    3. ``config/platform.yaml``
    4. Built-in defaults in this file

Every path in the config is resolved relative to the project root, so the
same file works unchanged on Windows and Linux.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "platform.yaml"

ENV_PREFIX = "UIP_"


# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "platform": {
        "bus_id": "BUS-01",
        "camera_id": "CAM-FRONT",
        "log_level": "INFO",
        "log_file": "outputs/platform.log",
        "device": "cpu",              # "cpu", "0", "cuda:0" ...
    },
    "video": {
        "source": "data/videos/demo.mp4",   # file path, or "0" for webcam
        "loop": False,
        "max_frames": 0,               # 0 = whole video
        "start_frame": 0,
        "resize_width": 0,             # 0 = keep native size
        "display": False,              # cv2.imshow window
        "save_annotated": True,
        "annotated_output": "outputs/runs/annotated.mp4",
        "codec": "avc1,H264,mp4v",
    },
    "gps": {
        # Simulated GPS track for development (no bus hardware in this phase).
        # Replace this block with a real GPS reader later -- everything
        # downstream only sees the common event schema.
        "mode": "simulated",           # simulated | fixed | none
        "start": [13.0827, 80.2707],   # Chennai
        "end": [13.0900, 80.2800],
        "segment_size_m": 60,          # road-segment quantisation
    },
    "perception": {
        "vehicle": {
            "enabled": True,
            "model": "models/yolo11n.pt",
            "confidence_threshold": 0.25,
            "image_size": 640,
            "tracker": "botsort.yaml",
            "frame_stride": 1,
        },
        "road_defect": {
            "enabled": True,
            "model": "models/road_defect_best.pt",
            "confidence_threshold": 0.35,
            "image_size": 640,
            "frame_stride": 5,            # defects are static -- no need
                                          # to run this every frame
        },
        "waterlogging": {
            "enabled": True,
            "model": "models/waterlogging_best.pt",
            # Match the weights' validated card (wl_share/VALIDATION_REPORT.md):
            # imgsz 512, conf 0.35. These are code-level fallbacks used only if
            # platform.yaml omits the key, so keep them in sync with the yaml.
            "confidence_threshold": 0.35,
            "image_size": 512,
            "frame_stride": 5,
        },
        "pedestrian": {
            "enabled": True,
            # Falls back to the shared COCO checkpoint (person class) until
            # a dedicated child-aware model is trained -- see
            # vendor/person3_pedestrian_safety/train.py.
            "model": "models/yolo11n.pt",
            "confidence_threshold": 0.3,
            "iou": 0.50,
            "image_size": 640,
            "tracker": "botsort.yaml",
            "frame_stride": 1,
        },
    },
    "modules": {
        "road_detection": {
            "enabled": True,
            "min_confidence": 0.35,
            "emit_cooldown_s": 3.0,
            "save_evidence": True,
        },
        "traffic": {
            "enabled": True,
            "maximum_vehicles": 20,
            "observation_interval_s": 5.0,
            "congestion": {
                "enabled": True,
                "density_threshold": 0.40,
                "speed_threshold": 35.0,
                "minimum_duration_seconds": 8,
            },
            "bottleneck": {
                "enabled": True,
                "density_threshold": 0.60,
                "speed_threshold": 15.0,
                "minimum_duration_seconds": 10,
                "rows": 3,
                "columns": 3,
                "max_vehicles_per_zone": 8,
            },
            "counting": {
                "enabled": True,
                "line_start": [0, 400],
                "line_end": [1280, 400],
            },
            "free_flow_speed_px_s": 60.0,   # used for the delay estimate
        },
        "pedestrian_safety": {
            "enabled": True,
            "save_evidence": True,
            # PERSON != PEDESTRIAN: an event needs a DETECTED zebra crossing,
            # the person's feet on it, moving vehicles, and temporal
            # confirmation. No fixed image zone is ever used as a crossing.
            # Frame counts not listed here scale from the video fps.
            "zebra_crossing": {
                "classes": ["zebra_crossing"],
                "min_confidence": 0.35,
                "min_detections": 2,
                "match_iou": 0.2,
            },
            "person": {
                "foot_band_ratio": 0.2,
                "min_foot_overlap": 0.5,
                "on_vehicle_bottom_margin": 0.1,
            },
            "vehicle_motion": {
                "min_displacement_px": 6.0,
                "min_displacement_ratio": 0.12,
                "max_distance_to_crossing": 1.5,
                "ego_motion_compensation": True,
                "ego_motion_min_detections": 3,
                "ego_motion_significance": 2.0,
                "occlusion_overlap": 0.05,
                "max_shape_change": 0.3,
            },
            "temporal": {
                "require_crossing_motion": False,
                "crossing_progress_ratio": 0.25,
            },
            "severity_floor": "HIGH",
            # Person 3's additive, explainable risk weights -- unmodified.
            # They set the severity of a confirmed event only.
            "risk": {
                "thresholds": {"low_max": 20, "medium_max": 45, "high_max": 70},
                "child_multiplier": 1.20,
                "crossing_bonus": 15,
                "road_bonus": 10,
                "approach_bonus": 25,
                "close_bonus": 25,
                "close_distance_px": 70,
                "dangerous_distance_px": 40,
                "approach_threshold_px_per_frame": 0.8,
                "vehicle_speed_px_per_frame": 1.0,
                "max_pair_distance_px": 260,
            },
        },
        "incident": {
            "enabled": True,
            "history_frames": 45,
            "crop_buffer_frames": 12,
            "crop_buffer_stride": 2,
            "minimum_track_age": 5,
            "anpr_trigger_confidence": 0.55,
            "rash": {
                "enabled": True,
                "speed_threshold_px_s": 250.0,
                "acceleration_threshold": 150.0,
                "lateral_threshold": 80.0,
                "cooldown_s": 6.0,
                "trigger_anpr_on_high": True,
            },
            "collision": {
                "enabled": True,
                "history_frames": 30,
                "confirm_peak": 0.62,
                "confirm_sustained": 0.50,
                "min_persist_frames": 2,
                "contact_min_iou": 0.08,
                "approach_min": 0.35,
                "motion_min": 0.30,
                "cooldown_frames": 45,
                "overlap_episode_min": 0.06,
                "max_episode_frames": 30,
                "separated_threshold": 0.06,
                "max_vehicle_fraction": 0.42,
                "possible": {
                    "confirm_peak": 0.45,
                    "confirm_sustained": 0.36,
                    "contact_min_iou": 0.05,
                    "approach_min": 0.25,
                    "motion_min": 0.20,
                },
            },
            "hit_and_run": {
                "enabled": True,
                "departure_distance_threshold": 70.0,
                "escape_ratio": 1.5,
                "escape_progress": 0.5,
                "min_escape_frames": 12,
                "hit_and_run_confirmation_time": 0.4,
                "require_consecutive": 3,
                # A track missing for longer than this is TRACK_LOST.
                # Track loss NEVER produces a hit-and-run.
                "track_loss_timeout": 30,
                "monitor_window_frames": 120,
                "cooldown_frames": 90,
                "trigger_anpr": True,
            },
            # OFF by default.  See modules/incident/module.py::_demo_injection
            "demo": {
                "inject_synthetic_collision": False,
                "at_frame": 0,
            },
            "evidence": {
                "enabled": True,
                "pre_frames": 10,
                "post_frames": 8,
                "stride": 3,
                "save_clip": True,
            },
        },
        "anpr": {
            "enabled": True,
            "model": "models/plate_best.pt",
            "gpu": False,
            "max_frames_to_ocr": 5,
            "trigger_event_types": ["hit_and_run", "collision", "rash_driving"],
            "min_plate_confidence": 0.45,
            "min_trigger_severity": "HIGH",
            "save_evidence": True,
        },
        "intelligence": {
            "enabled": True,
            "dedup": {
                "gps_radius_m": 35.0,
                "time_window_s": 300,
                "same_bus_window_s": 60,
            },
            "persistence": {
                "resolve_after_clean_passes": 2,
                "stale_after_days": 30,
            },
            "priority": {
                "weights": {
                    "severity": 0.35,
                    "persistence": 0.25,
                    "traffic_impact": 0.20,
                    "safety_risk": 0.20,
                },
            },
        },
    },
    "backend": {
        "enabled": True,
        "mode": "direct",              # direct (in-process DB) | http
        "url": "http://127.0.0.1:8000",
        "host": "127.0.0.1",
        "port": 8000,
        "database": "outputs/urban_intelligence.db",
        "evidence_dir": "outputs/evidence",
    },
    "output": {
        "events_jsonl": "outputs/runs/events.jsonl",
        "summary_json": "outputs/runs/summary.json",
    },
}


# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _env_overrides() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = raw_key[len(ENV_PREFIX):].lower().split("__")
        cursor = out
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(raw_value)
    return out


class Config:
    """Dict wrapper with dotted access and project-root path resolution."""

    def __init__(self, data: Dict[str, Any], root: Path = PROJECT_ROOT):
        self._data = data
        self.root = Path(root)

    # -- access -------------------------------------------------------------

    def get(self, dotted: str, default: Any = None) -> Any:
        cursor: Any = self._data
        for part in dotted.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def section(self, dotted: str) -> Dict[str, Any]:
        value = self.get(dotted, {})
        return value if isinstance(value, dict) else {}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- paths --------------------------------------------------------------

    def path(self, dotted: str, default: Optional[str] = None) -> Optional[Path]:
        raw = self.get(dotted, default)
        if raw in (None, ""):
            return None
        return self.resolve(raw)

    def resolve(self, raw) -> Path:
        p = Path(str(raw)).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def ensure_dirs(self) -> None:
        for dotted in (
            "video.annotated_output",
            "output.events_jsonl",
            "output.summary_json",
            "platform.log_file",
            "backend.database",
        ):
            p = self.path(dotted)
            if p:
                p.parent.mkdir(parents=True, exist_ok=True)
        evidence = self.path("backend.evidence_dir")
        if evidence:
            evidence.mkdir(parents=True, exist_ok=True)


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    data = copy.deepcopy(DEFAULTS)

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            file_data = yaml.safe_load(handle) or {}
        data = _deep_merge(data, file_data)

    data = _deep_merge(data, _env_overrides())
    data = _deep_merge(data, overrides or {})
    return Config(data)
