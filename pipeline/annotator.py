"""
pipeline.annotator
==================

Draws the integrated overlay -- one window / one output video showing every
module at once, which is what makes the demo readable:

  green boxes   vehicles (Person 2, with track id + speed)
  orange boxes  road defects (Person 1)
  grey boxes    PERSON: any detected human (pavement, bus stop, inside a
                bus, beside the road...) -- not a pedestrian
  cyan boxes    PERSON / CROSSING: feet on a DETECTED zebra crossing, but
                the full vulnerable-pedestrian rule is not met (no moving
                vehicles yet, or not yet temporally confirmed)
  red boxes     PEDESTRIAN: confirmed crossing on a detected zebra crossing
                while vehicles are moving (Person 3, Module 3); tagged
                VULNERABLE PEDESTRIAN EVENT
  white outline zebra crossings currently tracked from Person 1's model
  amber halo    vehicles under post-collision monitoring (Person 4)
  red banner    incidents, with the plate once ANPR has read it
  left panel    traffic analytics
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

COLOR_VEHICLE = (0, 220, 0)
COLOR_WATCH = (0, 200, 255)
COLOR_DEFECT = (0, 150, 255)
COLOR_ALERT = (0, 0, 255)
COLOR_PEDESTRIAN = (0, 0, 255)     # confirmed pedestrian (event) -- red
COLOR_CROSSING = (255, 220, 0)     # person on a detected crossing, no event
COLOR_PERSON = (180, 180, 180)     # ordinary detected person
COLOR_ZEBRA = (255, 255, 255)      # tracked zebra crossing region
COLOR_PANEL = (18, 18, 18)
COLOR_TEXT = (245, 245, 245)

SEVERITY_COLOR = {
    "CRITICAL": (0, 0, 255),
    "HIGH": (0, 80, 255),
    "MEDIUM": (0, 190, 255),
    "LOW": (0, 220, 180),
    "INFO": (200, 200, 200),
}


class Annotator:
    def __init__(self, enabled: bool = True, banner_frames: int = 90):
        self.enabled = enabled
        self._banner: Deque[Dict[str, Any]] = deque(maxlen=6)
        self.banner_frames = banner_frames
        try:
            import cv2                                          # noqa: PLC0415
            self.cv2 = cv2
        except Exception:                                        # noqa: BLE001
            self.cv2 = None
            self.enabled = False

    # -- alerts -------------------------------------------------------------

    def push_events(self, events: List[Any], frame_number: int) -> None:
        for event in events:
            if event.event_type in (
                "collision", "possible_collision", "hit_and_run",
                "rash_driving", "abnormal_movement", "vehicle_identification",
                "pedestrian_vehicle_risk",
            ):
                self._banner.append({
                    "text": self._alert_text(event),
                    "severity": event.severity,
                    "until": frame_number + self.banner_frames,
                })

    @staticmethod
    def _alert_text(event) -> str:
        meta = event.metadata or {}
        label = event.event_type.replace("_", " ").upper()
        if event.event_type == "vehicle_identification":
            plate = meta.get("registration_number") or "unreadable"
            return f"PLATE  track {event.track_id}: {plate} ({event.confidence:.2f})"
        if event.event_type == "pedestrian_vehicle_risk":
            vehicles = ",".join(f"#{v}" for v in (meta.get("vehicle_track_ids") or []))
            return (
                f"VULNERABLE PEDESTRIAN EVENT  "
                f"{meta.get('pedestrian_category', 'vulnerable pedestrian')} "
                f"#{event.track_id} on zebra crossing  "
                f"moving vehicles {vehicles or '-'}  {event.severity}"
            )
        plate = meta.get("registration_number")
        suffix = f"  plate {plate}" if plate else ""
        return f"{label}  track {event.track_id} conf {event.confidence:.2f}{suffix}"

    # -- drawing ------------------------------------------------------------

    def draw(self, ctx, module_events: Optional[List[Any]] = None):
        if not self.enabled or self.cv2 is None:
            return ctx.frame
        cv2 = self.cv2
        frame = ctx.frame

        for defect in (ctx.shared.get("road_defects") or []):
            x1, y1, x2, y2 = defect["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DEFECT, 2)
            self._label(
                frame,
                f"{defect['class_name']} {defect['confidence']:.2f}",
                x1, y1, COLOR_DEFECT,
            )

        for vehicle in (ctx.vehicles or []):
            x1, y1, x2, y2 = vehicle["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VEHICLE, 2)
            self._label(
                frame,
                f"{vehicle['class_name']} #{vehicle.get('track_id')} "
                f"{vehicle.get('speed', 0.0):.0f}px/s",
                x1, y1, COLOR_VEHICLE,
            )

        for zebra in (ctx.shared.get("zebra_crossings") or []):
            x1, y1, x2, y2 = zebra["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ZEBRA, 1)
            self._label(frame, f"ZEBRA CROSSING {zebra.get('confidence', 0.0):.2f}",
                        x1, y2 + 16, COLOR_ZEBRA)

        # PERSON != PEDESTRIAN. Module 3 tags each person with a role:
        #   "person"     ordinary detected human
        #   "crossing"   feet on a detected zebra crossing, no event
        #   "pedestrian" confirmed crossing while vehicles move (event)
        # Without a role (Module 3 unavailable) the conservative label is
        # PERSON, never PEDESTRIAN.
        for person in (ctx.shared.get("persons") or []):
            x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
            role = person.get("role", "person")
            tid = person.get("track_id")
            if role == "pedestrian":
                color, text, thickness = COLOR_PEDESTRIAN, f"PEDESTRIAN #{tid}", 3
            elif role == "crossing":
                color, text, thickness = COLOR_CROSSING, f"PERSON / CROSSING #{tid}", 2
            else:
                color, text, thickness = COLOR_PERSON, f"PERSON #{tid}", 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            self._label(frame, f"{text} {person.get('confidence', 0.0):.2f}", x1, y1, color)
            if role == "pedestrian":
                self._label(frame, "VULNERABLE PEDESTRIAN EVENT", x1, y2 + 18, COLOR_PEDESTRIAN)

        # vehicles the incident engine is monitoring after a collision
        watched = ctx.shared.get("incident_watch") or {}
        for vehicle in (ctx.vehicles or []):
            state = watched.get(vehicle.get("track_id"))
            if not state:
                continue
            x1, y1, x2, y2 = vehicle["bbox"]
            cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), COLOR_WATCH, 2)
            self._label(frame, state, x1, y2 + 18, COLOR_WATCH)

        self._panel(frame, ctx)
        self._banners(frame, ctx)
        return frame

    def _label(self, frame, text, x, y, color) -> None:
        cv2 = self.cv2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x, max(0, y - th - 8)), (x + tw + 6, y), color, -1)
        cv2.putText(
            frame, text, (x + 3, max(11, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
        )

    def _panel(self, frame, ctx) -> None:
        cv2 = self.cv2
        traffic = ctx.shared.get("traffic") or {}
        height, width = frame.shape[:2]
        panel_w, panel_h = 320, 272
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        congestion = (traffic.get("congestion") or {})
        gps = ctx.gps
        lines = [
            ("AI URBAN INTELLIGENCE", 0.55, COLOR_TEXT),
            (f"{ctx.bus_id or '-'} / {ctx.camera_id or '-'}   t={ctx.video_time_s:.1f}s", 0.42, COLOR_TEXT),
            (
                f"GPS {gps.latitude:.5f},{gps.longitude:.5f}" if gps else "GPS n/a",
                0.42, COLOR_TEXT,
            ),
            (f"segment {ctx.road_segment or '-'}", 0.40, (170, 170, 170)),
            (f"Vehicles: {traffic.get('vehicle_count', 0)}  "
             f"unique {traffic.get('unique_vehicles', 0)}", 0.45, COLOR_TEXT),
            (f"Density: {traffic.get('density_percentage', 0)}%  "
             f"Level: {traffic.get('traffic_level', '-')}", 0.45,
             SEVERITY_COLOR.get(
                 {"LOW": "INFO", "MEDIUM": "LOW", "HIGH": "MEDIUM",
                  "CRITICAL": "HIGH"}.get(traffic.get("traffic_level", "LOW"), "INFO"),
                 COLOR_TEXT)),
            (f"Congestion: {congestion.get('status', '-')} "
             f"({congestion.get('duration_seconds', 0):.0f}s)", 0.45, COLOR_TEXT),
            (f"Bottleneck: {traffic.get('bottleneck_status', '-')} "
             f"{traffic.get('bottleneck_zone') or ''}", 0.45, COLOR_TEXT),
            (f"Avg speed: {traffic.get('average_speed_px_s', 0)} px/s   "
             f"counted {traffic.get('counted_total', 0)}", 0.42, COLOR_TEXT),
            (f"Vehicles: {len(ctx.vehicles or [])}   "
             f"Defects: {len(ctx.shared.get('road_defects') or [])}   "
             f"People: {len(ctx.shared.get('persons') or [])}", 0.42, COLOR_TEXT),
            (f"Zebra: {len(ctx.shared.get('zebra_crossings') or [])}   "
             f"On crossing: {sum(1 for p in (ctx.shared.get('persons') or []) if p.get('role') in ('crossing', 'pedestrian'))}   "
             f"Pedestrians: {len(ctx.shared.get('pedestrians') or [])}", 0.42, COLOR_TEXT),
        ]
        y = 34
        for text, scale, color in lines:
            cv2.putText(
                frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                cv2.LINE_AA,
            )
            y += 23

    def _banners(self, frame, ctx) -> None:
        cv2 = self.cv2
        active = [b for b in self._banner if b["until"] >= ctx.frame_number]
        if not active:
            return
        height, width = frame.shape[:2]
        y = height - 14 - 26 * len(active)
        for banner in active:
            color = SEVERITY_COLOR.get(banner["severity"], COLOR_ALERT)
            overlay = frame.copy()
            cv2.rectangle(overlay, (8, y - 18), (width - 8, y + 6), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(
                frame, banner["text"], (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
            y += 26
