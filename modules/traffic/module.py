"""
MODULE 2 -- TRAFFIC INTELLIGENCE                (owner: Person 2)
=================================================================

No new ML model.  This module reads the vehicle detections + tracks that
``modules.perception.vehicle`` already produced with **Person 2's own
detector**, and runs **Person 2's own analytics classes**, loaded unmodified
from ``vendor/person2_traffic/src/analytics/``:

    traffic_density.py  -> TrafficDensity        (density %, LOW/MED/HIGH/CRITICAL)
    congestion.py       -> CongestionAnalyzer    (congestion + duration)
    bottleneck.py       -> BottleneckAnalyzer    (zone grid, bottleneck score)
    peak_traffic.py     -> PeakTrafficAnalyzer   (session peak)
    vehicle_counter.py  -> VehicleCounter        (line crossing counts)

What this adapter adds is only the parts Person 2's standalone `main.py`
printed to a window instead of exporting: turning the analyser output into
common Events, per-road-segment route analytics, delay estimation and
heatmap-ready rows.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from core.event import Event, EventType, Severity, SourceModule, make_event
from core.frame_context import FrameContext
from core.module_base import BaseModule, ModuleUnavailable
from core.vendor_loader import load_vendor_module, VendorImportError

from modules.perception.clock import patch_module_clock

VENDOR = "person2_traffic/src/analytics/"

VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")

SEVERITY_BY_LEVEL = {
    "LOW": Severity.INFO,
    "MEDIUM": Severity.LOW,
    "HIGH": Severity.MEDIUM,
    "CRITICAL": Severity.HIGH,
}


class TrafficModule(BaseModule):
    name = "traffic"
    owner = "Person 2"

    def __init__(self, config, perception=None, clock=None):
        super().__init__(config.section("modules.traffic"))
        self.platform_config = config
        self.perception = perception
        self.clock = clock

        self.observation_interval_s = float(
            self.config.get("observation_interval_s", 5.0)
        )
        self.free_flow_speed = float(self.config.get("free_flow_speed_px_s", 60.0))

        self.density = None
        self.congestion = None
        self.bottleneck = None
        self.peak = None
        self.counter = None

        self._last_observation_t = -1e9
        self._congestion_state = "NORMAL"
        self._congestion_open: Optional[Dict[str, Any]] = None
        self._bottleneck_state = "NORMAL"

        # road segment -> rolling aggregate, the heatmap-ready rows
        self._segments: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "samples": 0,
                "vehicle_total": 0,
                "speed_total": 0.0,
                "density_total": 0.0,
                "max_level": "LOW",
                "congested_samples": 0,
                "latitude": None,
                "longitude": None,
            }
        )
        self._unique_tracks: set = set()

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        if self.perception is None or not self.perception.available:
            raise ModuleUnavailable("vehicle perception unavailable")

        try:
            density_mod = load_vendor_module(VENDOR + "traffic_density.py")
            congestion_mod = load_vendor_module(VENDOR + "congestion.py")
            bottleneck_mod = load_vendor_module(VENDOR + "bottleneck.py")
            peak_mod = load_vendor_module(VENDOR + "peak_traffic.py")
            counter_mod = load_vendor_module(VENDOR + "vehicle_counter.py")
        except VendorImportError as exc:
            raise ModuleUnavailable(f"Person 2 analytics not importable: {exc}") from exc

        if self.clock is not None:
            # Durations must be measured in video time when replaying a file.
            patch_module_clock([congestion_mod, bottleneck_mod, peak_mod], self.clock)

        self.density = density_mod.TrafficDensity(
            maximum_vehicles=int(self.config.get("maximum_vehicles", 20))
        )

        congestion_cfg = self.config.get("congestion", {}) or {}
        if congestion_cfg.get("enabled", True):
            self.congestion = congestion_mod.CongestionAnalyzer(
                density_threshold=float(congestion_cfg.get("density_threshold", 0.40)),
                speed_threshold=float(congestion_cfg.get("speed_threshold", 35.0)),
                minimum_duration_seconds=float(
                    congestion_cfg.get("minimum_duration_seconds", 8)
                ),
            )

        bottleneck_cfg = self.config.get("bottleneck", {}) or {}
        if bottleneck_cfg.get("enabled", True):
            self.bottleneck = bottleneck_mod.BottleneckAnalyzer(
                density_threshold=float(bottleneck_cfg.get("density_threshold", 0.60)),
                speed_threshold=float(bottleneck_cfg.get("speed_threshold", 15.0)),
                minimum_duration_seconds=float(
                    bottleneck_cfg.get("minimum_duration_seconds", 10)
                ),
                rows=int(bottleneck_cfg.get("rows", 3)),
                columns=int(bottleneck_cfg.get("columns", 3)),
                max_vehicles_per_zone=int(
                    bottleneck_cfg.get("max_vehicles_per_zone", 8)
                ),
            )

        self.peak = peak_mod.PeakTrafficAnalyzer()

        counting_cfg = self.config.get("counting", {}) or {}
        if counting_cfg.get("enabled", True):
            self.counter = counter_mod.VehicleCounter(
                line_start=counting_cfg.get("line_start", [0, 400]),
                line_end=counting_cfg.get("line_end", [1280, 400]),
            )
        self._counter_cfg = counting_cfg

    # -- per frame ----------------------------------------------------------

    def process(self, ctx: FrameContext) -> List[Event]:
        vehicles = [
            v for v in (ctx.vehicles or [])
            if v.get("class_name") in VEHICLE_CLASSES
        ]
        events: List[Event] = []

        # Counting line defaults to the middle of the frame if the configured
        # one sits outside this video's resolution.
        if self.counter is not None and ctx.height and not self._counter_cfg.get("_fixed"):
            line_y = self.counter.get_line_y()
            if line_y <= 0 or line_y >= ctx.height:
                self.counter.line_start = (0, int(ctx.height * 0.55))
                self.counter.line_end = (int(ctx.width), int(ctx.height * 0.55))
            self._counter_cfg["_fixed"] = True

        vehicle_count = len(vehicles)
        for vehicle in vehicles:
            if vehicle.get("track_id") is not None:
                self._unique_tracks.add(vehicle["track_id"])

        density_pct = self.density.percentage(vehicle_count)
        traffic_level = self.density.traffic_level(vehicle_count)
        average_speed = self.perception.average_speed(vehicles)

        self.peak.update(vehicle_count)

        crossings = self.counter.update(vehicles) if self.counter is not None else []

        congestion_result = (
            self.congestion.update(density_pct, average_speed)
            if self.congestion is not None
            else {"status": "DISABLED", "is_congested": False, "duration_seconds": 0.0}
        )
        bottleneck_result = (
            self.bottleneck.analyze(vehicles, ctx.width or 1280, ctx.height or 720)
            if self.bottleneck is not None
            else {"status": "DISABLED", "strongest_zone": None, "strongest_score": 0.0}
        )

        counts = {name: 0 for name in VEHICLE_CLASSES}
        for vehicle in vehicles:
            counts[vehicle["class_name"]] += 1

        snapshot = {
            "vehicle_count": vehicle_count,
            "counts_by_class": counts,
            "density_percentage": round(density_pct, 1),
            "traffic_level": traffic_level,
            "average_speed_px_s": round(average_speed, 2),
            "moving_vehicles": self.perception.moving_count(vehicles),
            "slow_vehicles": self.perception.slow_count(vehicles),
            "congestion": congestion_result,
            "bottleneck_status": bottleneck_result.get("status"),
            "bottleneck_zone": bottleneck_result.get("strongest_zone"),
            "bottleneck_score": round(float(bottleneck_result.get("strongest_score", 0.0)), 3),
            "line_crossings": crossings,
            "counted_total": self.counter.get_total() if self.counter else 0,
            "counted_by_class": self.counter.get_counts() if self.counter else {},
            "directions": self.counter.get_direction_counts() if self.counter else {},
            "peak": self.peak.get_result(),
            "unique_vehicles": len(self._unique_tracks),
        }
        # Published so the incident module and the annotator can read the same
        # numbers without recomputing them.
        ctx.shared["traffic"] = snapshot

        self._accumulate_segment(ctx, snapshot)

        events.extend(self._congestion_events(ctx, snapshot, congestion_result))
        events.extend(self._bottleneck_events(ctx, snapshot, bottleneck_result))
        events.extend(self._observation_event(ctx, snapshot))
        return events

    def finalize(self) -> List[Event]:
        events: List[Event] = []
        if self._congestion_open is not None:
            events.append(self._close_congestion(None))
        return [e for e in events if e is not None]

    # -- event builders -----------------------------------------------------

    def _estimated_delay_min(self, average_speed: float, duration_s: float) -> float:
        """Rough delay estimate: how much longer this stretch takes at the
        observed speed compared with free-flow.

        Speed is Person 2's relative px/s value, so this is a *comparative*
        estimate, not a calibrated minute count.  Calibrating
        ``free_flow_speed_px_s`` per camera is what makes it physical.
        """
        if average_speed <= 0.5:
            ratio = 4.0
        else:
            ratio = min(6.0, self.free_flow_speed / average_speed)
        return round(max(0.0, (ratio - 1.0)) * (duration_s / 60.0), 2)

    def _congestion_events(self, ctx, snapshot, result) -> List[Event]:
        status = result.get("status", "NORMAL")
        events: List[Event] = []

        if status == "CONGESTED" and self._congestion_state != "CONGESTED":
            self._congestion_open = {
                "start_time_s": ctx.video_time_s,
                "segment": ctx.road_segment,
                "peak_vehicles": snapshot["vehicle_count"],
            }
        elif status == "CONGESTED" and self._congestion_open is not None:
            self._congestion_open["peak_vehicles"] = max(
                self._congestion_open["peak_vehicles"], snapshot["vehicle_count"]
            )

        if status != "CONGESTED" and self._congestion_state == "CONGESTED":
            closed = self._close_congestion(ctx)
            if closed is not None:
                events.append(closed)

        # Emit once when congestion is confirmed, then let the observation
        # events carry the running duration.
        if status == "CONGESTED" and self._congestion_state != "CONGESTED":
            duration = float(result.get("duration_seconds", 0.0))
            events.append(make_event(
                EventType.TRAFFIC_CONGESTION,
                SourceModule.TRAFFIC,
                confidence=round(min(0.99, 0.5 + snapshot["density_percentage"] / 200.0), 3),
                severity=SEVERITY_BY_LEVEL.get(snapshot["traffic_level"], Severity.MEDIUM),
                class_name="congestion_start",
                metadata={
                    "traffic_level": snapshot["traffic_level"],
                    "vehicle_count": snapshot["vehicle_count"],
                    "road_segment": ctx.road_segment,
                    "congestion_duration": round(duration, 1),
                    "bottleneck_score": snapshot["bottleneck_score"],
                    "estimated_delay": self._estimated_delay_min(
                        snapshot["average_speed_px_s"], duration
                    ),
                    "density_percentage": snapshot["density_percentage"],
                    "average_speed_px_s": snapshot["average_speed_px_s"],
                    "counts_by_class": snapshot["counts_by_class"],
                    "state": "STARTED",
                },
            ))

        self._congestion_state = status
        return events

    def _close_congestion(self, ctx) -> Optional[Event]:
        opened = self._congestion_open
        self._congestion_open = None
        if opened is None:
            return None
        end_t = ctx.video_time_s if ctx is not None else opened["start_time_s"]
        duration = max(0.0, end_t - opened["start_time_s"])
        return make_event(
            EventType.TRAFFIC_CONGESTION,
            SourceModule.TRAFFIC,
            confidence=0.8,
            severity=Severity.LOW,
            class_name="congestion_end",
            road_segment=opened["segment"],
            metadata={
                "state": "ENDED",
                "congestion_duration": round(duration, 1),
                "peak_vehicles": opened["peak_vehicles"],
                "estimated_delay": self._estimated_delay_min(1.0, duration),
            },
        )

    def _bottleneck_events(self, ctx, snapshot, result) -> List[Event]:
        status = result.get("status", "NORMAL")
        if status == "BOTTLENECK" and self._bottleneck_state != "BOTTLENECK":
            self._bottleneck_state = status
            return [make_event(
                EventType.TRAFFIC_BOTTLENECK,
                SourceModule.TRAFFIC,
                confidence=round(min(0.99, 0.4 + snapshot["bottleneck_score"]), 3),
                severity=Severity.MEDIUM,
                class_name="bottleneck",
                metadata={
                    "traffic_level": snapshot["traffic_level"],
                    "vehicle_count": snapshot["vehicle_count"],
                    "road_segment": ctx.road_segment,
                    "bottleneck_score": snapshot["bottleneck_score"],
                    "bottleneck_zone": snapshot["bottleneck_zone"],
                    "zones": result.get("bottleneck_zones", []),
                    "estimated_delay": self._estimated_delay_min(
                        snapshot["average_speed_px_s"], 60.0
                    ),
                },
            )]
        self._bottleneck_state = status
        return []

    def _observation_event(self, ctx, snapshot) -> List[Event]:
        """Periodic heartbeat -- this is what feeds route analytics, the
        heatmap and the time series on the dashboard."""
        if (ctx.video_time_s - self._last_observation_t) < self.observation_interval_s:
            return []
        self._last_observation_t = ctx.video_time_s

        congestion = snapshot["congestion"]
        return [make_event(
            EventType.TRAFFIC_OBSERVATION,
            SourceModule.TRAFFIC,
            confidence=0.9,
            severity=SEVERITY_BY_LEVEL.get(snapshot["traffic_level"], Severity.INFO),
            class_name="traffic_observation",
            metadata={
                "traffic_level": snapshot["traffic_level"],
                "vehicle_count": snapshot["vehicle_count"],
                "unique_vehicles": snapshot["unique_vehicles"],
                "counts_by_class": snapshot["counts_by_class"],
                "density_percentage": snapshot["density_percentage"],
                "average_speed_px_s": snapshot["average_speed_px_s"],
                "moving_vehicles": snapshot["moving_vehicles"],
                "slow_vehicles": snapshot["slow_vehicles"],
                "congestion_status": congestion.get("status"),
                "congestion_duration": round(float(congestion.get("duration_seconds", 0.0)), 1),
                "bottleneck_status": snapshot["bottleneck_status"],
                "bottleneck_score": snapshot["bottleneck_score"],
                "bottleneck_zone": snapshot["bottleneck_zone"],
                "counted_total": snapshot["counted_total"],
                "counted_by_class": snapshot["counted_by_class"],
                "directions": snapshot["directions"],
                "peak_vehicles": snapshot["peak"]["peak_vehicles"],
                "estimated_delay": self._estimated_delay_min(
                    snapshot["average_speed_px_s"],
                    float(congestion.get("duration_seconds", 0.0)),
                ),
                "road_segment": ctx.road_segment,
            },
        )]

    # -- route analytics / heatmap -----------------------------------------

    def _accumulate_segment(self, ctx, snapshot) -> None:
        if not ctx.road_segment:
            return
        row = self._segments[ctx.road_segment]
        row["samples"] += 1
        row["vehicle_total"] += snapshot["vehicle_count"]
        row["speed_total"] += snapshot["average_speed_px_s"]
        row["density_total"] += snapshot["density_percentage"]
        if snapshot["congestion"].get("is_congested"):
            row["congested_samples"] += 1
        levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        if levels.index(snapshot["traffic_level"]) > levels.index(row["max_level"]):
            row["max_level"] = snapshot["traffic_level"]
        if ctx.gps is not None:
            row["latitude"] = ctx.gps.latitude
            row["longitude"] = ctx.gps.longitude

    def route_analytics(self) -> List[Dict[str, Any]]:
        """Heatmap-ready rows: one per road segment travelled."""
        rows = []
        for segment, row in self._segments.items():
            samples = max(1, row["samples"])
            rows.append({
                "road_segment": segment,
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "samples": row["samples"],
                "avg_vehicle_count": round(row["vehicle_total"] / samples, 2),
                "avg_density_percentage": round(row["density_total"] / samples, 1),
                "avg_speed_px_s": round(row["speed_total"] / samples, 2),
                "congested_ratio": round(row["congested_samples"] / samples, 3),
                "max_traffic_level": row["max_level"],
            })
        rows.sort(key=lambda r: r["avg_density_percentage"], reverse=True)
        return rows

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        data["unique_vehicles"] = len(self._unique_tracks)
        data["counted_total"] = self.counter.get_total() if self.counter else 0
        data["peak_vehicles"] = self.peak.get_peak_vehicles() if self.peak else 0
        data["segments"] = len(self._segments)
        return data
