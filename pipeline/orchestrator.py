"""
pipeline.orchestrator
=====================

The one place that decides which module runs and when.

    video
      -> vehicle detection + tracking        (Person 2, shared)
      -> road defect detection               (Person 1, shared, strided)
      -> pedestrian/child detection          (Person 3, shared)
      -> traffic analysis                    (Person 2)
      -> pedestrian safety / risk scoring     (Person 3)
      -> incident analysis                   (Person 4)
      -> ANPR  *only when an incident asks for it*   (Person 5)
      -> event normalisation + intelligence  (Person 6)
      -> sinks: JSONL + database  -> backend API -> frontend

Design commitments:
  * Modules never call each other.  They read the FrameContext and return
    Events.  The orchestrator is the only thing that knows the order.
  * One failing module never stops the run (see core.module_base).
  * Every model runs at most once per frame.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import Config, load_config
from core.event import Event
from core.evidence import EvidenceStore
from core.frame_context import FrameContext
from core.geo import build_gps_provider, road_segment_id
from core.logging_setup import get_logger, setup_logging
from core.module_base import ModuleStatus
from core.sinks import build_sinks
from core.track_buffer import FrameRingBuffer, VehicleCropBuffer

from modules.anpr import ANPRModule
from modules.incident import IncidentModule
from modules.intelligence import IntelligenceEngine
from modules.pedestrian import PedestrianSafetyModule
from modules.perception import (
    RoadDefectPerception,
    VehiclePerception,
    WaterloggingPerception,
    PedestrianPerception,
)
from modules.perception.clock import VideoClock
from modules.road_detection import RoadDetectionModule
from modules.traffic import TrafficModule

from .annotator import Annotator
from .video_source import build_video_source


class Pipeline:
    def __init__(self, config: Optional[Config] = None, **overrides):
        self.config = config or load_config(overrides=overrides or None)
        self.config.ensure_dirs()
        setup_logging(
            self.config.get("platform.log_level", "INFO"),
            str(self.config.path("platform.log_file") or ""),
        )
        self.log = get_logger("pipeline")

        self.run_id = uuid.uuid4().hex[:12]
        self.bus_id = self.config.get("platform.bus_id", "BUS-01")
        self.camera_id = self.config.get("platform.camera_id", "CAM-FRONT")
        self._annotated_path: Optional[Path] = None

        self.clock = VideoClock(live=False)
        self.evidence = EvidenceStore(self.config)

        # --- shared perception (each model loaded once) ---
        self.vehicle_perception = VehiclePerception(self.config, clock=self.clock)
        self.road_perception = RoadDefectPerception(self.config)
        self.waterlogging_perception = WaterloggingPerception(self.config)
        self.pedestrian_perception = PedestrianPerception(self.config)
        self._perception_status: Dict[str, str] = {}

        # --- team modules ---
        self.road_module = RoadDetectionModule(
            self.config,
            perception=self.road_perception,
            evidence=self.evidence,
            waterlogging_perception=self.waterlogging_perception,
        )
        self.traffic_module = TrafficModule(
            self.config, perception=self.vehicle_perception, clock=self.clock
        )
        self.pedestrian_module = PedestrianSafetyModule(self.config, evidence=self.evidence)
        self.incident_module = IncidentModule(self.config, evidence=self.evidence)
        self.anpr_module = ANPRModule(self.config, evidence=self.evidence)

        self.frame_modules = [
            self.road_module,
            self.traffic_module,
            self.pedestrian_module,
            self.incident_module,
            self.anpr_module,          # process() is a no-op by design
        ]

        incident_cfg = self.config.section("modules.incident")
        self.crop_buffer = VehicleCropBuffer(
            per_track=int(incident_cfg.get("crop_buffer_frames", 12)),
            stride=int(incident_cfg.get("crop_buffer_stride", 2)),
        )
        self.clip_buffer = FrameRingBuffer(size=60, stride=2)

        self._writer_codec = None
        self.sinks = build_sinks(self.config)
        self.database = next(
            (getattr(s, "db", None) for s in self.sinks if hasattr(s, "db")), None
        )
        self.intelligence = IntelligenceEngine(self.config, database=self.database)

        self.annotator = Annotator(
            enabled=bool(
                self.config.get("video.save_annotated", True)
                or self.config.get("video.display", False)
            )
        )

        self.frames_processed = 0
        self.events_total = 0
        self.events_by_type: Dict[str, int] = {}
        self._writer = None
        self._started_at = 0.0

    # ---------------------------------------------------------------- setup

    def setup(self) -> None:
        self.log.info("=" * 66)
        self.log.info("AI URBAN INTELLIGENCE PLATFORM -- integrated pipeline")
        self.log.info("run %s   bus %s   camera %s", self.run_id, self.bus_id, self.camera_id)
        self.log.info("=" * 66)

        for name, perception in (
            ("vehicle", self.vehicle_perception),
            ("road_defect", self.road_perception),
            ("waterlogging", self.waterlogging_perception),
            ("pedestrian", self.pedestrian_perception),
        ):
            try:
                perception.setup()
                self._perception_status[name] = "READY"
            except Exception as exc:                             # noqa: BLE001
                self._perception_status[name] = f"UNAVAILABLE: {exc}"
                self.log.warning(
                    "perception.%s unavailable -- %s. Modules that need it will "
                    "be skipped, the rest of the pipeline continues.", name, exc,
                )

        for module in self.frame_modules:
            module.safe_setup()

        self.intelligence.setup()

        ready = [m.name for m in self.frame_modules if m.status == ModuleStatus.READY]
        skipped = [
            f"{m.name} ({m.status})"
            for m in self.frame_modules if m.status != ModuleStatus.READY
        ]
        self.log.info("modules ready : %s", ", ".join(ready) or "none")
        if skipped:
            self.log.warning("modules skipped: %s", ", ".join(skipped))

    # ------------------------------------------------------------------ run

    def run(self) -> Dict[str, Any]:
        self.setup()
        source = build_video_source(self.config)
        self.clock.live = source.is_camera

        gps_provider = build_gps_provider(self.config, fps=source.fps)
        segment_size = float(self.config.get("gps.segment_size_m", 60.0))
        total_frames = source.expected_frames

        if self.database is not None:
            self.database.start_run(
                self.run_id, self.bus_id, self.camera_id, source.raw_source
            )

        self._open_writer(source)
        display = bool(self.config.get("video.display", False))
        progress_every = max(1, int(source.fps * 5))

        self._started_at = time.time()
        self.log.info("processing ...")

        try:
            for ctx in source.frames():
                self._prepare_context(ctx, gps_provider, total_frames, segment_size)
                events = self._process_frame(ctx)
                self._dispatch(events)

                if self.annotator.enabled:
                    self.annotator.push_events(events, ctx.frame_number)
                    frame = self.annotator.draw(ctx)
                    if self._writer is not None:
                        self._writer.write(frame)
                    if display:
                        import cv2                               # noqa: PLC0415
                        cv2.imshow("AI Urban Intelligence Platform", frame)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            self.log.info("stopped by user (q)")
                            break

                self.frames_processed += 1
                if self.frames_processed % progress_every == 0:
                    self._log_progress(total_frames)
        except KeyboardInterrupt:
            self.log.warning("interrupted -- finalising what has been processed")
        finally:
            summary = self._finish(source)
        return summary

    # -------------------------------------------------------------- per-frame

    def _prepare_context(self, ctx: FrameContext, gps_provider, total_frames, segment_size):
        ctx.bus_id = self.bus_id
        ctx.camera_id = self.camera_id
        ctx.wall_time = time.time()
        ctx.gps = gps_provider.position(ctx.frame_number, total_frames)
        ctx.road_segment = road_segment_id(ctx.gps, segment_size)
        self.clock.set_video_time(ctx.video_time_s)

    def _process_frame(self, ctx: FrameContext) -> List[Event]:
        # 1. shared perception -- one inference per model per frame
        self._safe_perception("vehicle", self.vehicle_perception, ctx)
        self._safe_perception("road_defect", self.road_perception, ctx)
        self._safe_perception("waterlogging", self.waterlogging_perception, ctx)
        self._safe_perception("pedestrian", self.pedestrian_perception, ctx)

        # 2. buffers that make event-triggered ANPR / clips possible
        self.crop_buffer.update(ctx)
        self.clip_buffer.update(ctx)

        # 3. analysis modules
        events: List[Event] = []
        for module in self.frame_modules:
            events.extend(module.safe_process(ctx))

        # 4. event-triggered ANPR
        events.extend(self._run_anpr(events, ctx))

        # 5. intelligence engine (dedup / persistence / priority / routing)
        events.extend(self.intelligence.process_events(events))
        return events

    def _safe_perception(self, name: str, perception, ctx: FrameContext) -> None:
        if not getattr(perception, "available", False):
            return
        try:
            perception.update(ctx)
        except Exception as exc:                                 # noqa: BLE001
            self.log.error("perception.%s failed on frame %s: %s", name, ctx.frame_number, exc)

    def _run_anpr(self, events: List[Event], ctx: FrameContext) -> List[Event]:
        """ANPR runs here and nowhere else -- only for events that asked for
        identification, and only once per track."""
        produced: List[Event] = []
        for event in list(events):
            if not self.anpr_module.should_trigger(event):
                continue
            track_id = event.metadata.get("anpr_track_id", event.track_id)
            crops = self.crop_buffer.get(track_id)
            identification = self.anpr_module.identify(event, crops, ctx)
            if identification is not None:
                identification.with_context(ctx)
                produced.append(identification)
        return produced

    # -------------------------------------------------------------- dispatch

    def _dispatch(self, events: List[Event]) -> None:
        if not events:
            return
        for event in events:
            # Which run (and therefore which annotated_<run_id>.mp4) produced
            # this event.  Lets the backend link evidence to the exact
            # recording instead of matching it after the fact.
            if isinstance(event.metadata, dict):
                event.metadata.setdefault("run_id", self.run_id)
            self.events_total += 1
            self.events_by_type[event.event_type] = (
                self.events_by_type.get(event.event_type, 0) + 1
            )
        for sink in self.sinks:
            try:
                sink.emit_many(events)
            except Exception as exc:                             # noqa: BLE001
                self.log.error("sink %s failed: %s", sink.name, exc)

    # ---------------------------------------------------------------- finish

    def _finish(self, source) -> Dict[str, Any]:
        tail: List[Event] = []
        for module in self.frame_modules:
            tail.extend(module.safe_finalize())
        if tail:
            tail.extend(self.intelligence.process_events(tail))
        tail.extend(self.intelligence.flush_passes())
        self._dispatch(tail)

        for module in self.frame_modules:
            module.safe_close()
        if self._writer is not None:
            self._writer.release()
            self._make_browser_playable()
        source.release()
        if self.config.get("video.display", False):
            try:
                import cv2                                       # noqa: PLC0415
                cv2.destroyAllWindows()
            except Exception:                                    # noqa: BLE001
                pass

        summary = self.summary()
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:                                    # noqa: BLE001
                pass

        summary_path = self.config.path("output.summary_json")
        if summary_path:
            import json
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._print_summary(summary)
        return summary

    def summary(self) -> Dict[str, Any]:
        elapsed = max(1e-6, time.time() - self._started_at)
        if self.database is not None:
            try:
                self.database.finish_run(
                    self.run_id, self.frames_processed, self.events_total, {}
                )
            except Exception:                                    # noqa: BLE001
                pass
        return {
            "run_id": self.run_id,
            "bus_id": self.bus_id,
            "camera_id": self.camera_id,
            "frames_processed": self.frames_processed,
            "elapsed_seconds": round(elapsed, 2),
            "fps": round(self.frames_processed / elapsed, 2),
            "events_total": self.events_total,
            "events_by_type": dict(sorted(self.events_by_type.items())),
            "perception": dict(self._perception_status),
            "modules": [m.stats() for m in self.frame_modules],
            "intelligence": self.intelligence.stats(),
            "issues": [
                {
                    "issue_id": i["issue_id"],
                    "class_name": i.get("class_name"),
                    "status": i.get("status"),
                    "observations": i.get("observation_count"),
                    "buses": i.get("buses"),
                    "priority_score": i.get("priority_score"),
                    "priority_band": i.get("priority_band"),
                    "department": i.get("department"),
                    "road_segment": i.get("road_segment"),
                }
                for i in self.intelligence.issues()[:25]
            ],
            "route_analytics": (
                self.traffic_module.route_analytics()
                if self.traffic_module.status == ModuleStatus.READY else []
            ),
            "outputs": {
                "events_jsonl": str(self.config.path("output.events_jsonl")),
                "annotated_video": (
                    str(self._annotated_output_path())
                    if self._writer is not None else None
                ),
                "database": str(self.config.path("backend.database")),
                "evidence_dir": str(self.config.path("backend.evidence_dir")),
            },
        }

    # ----------------------------------------------------------------- misc

    def _annotated_output_path(self) -> Optional[Path]:
        """The annotated-video path, made unique per run.

        video.annotated_output in config is a single shared filename
        (outputs/runs/annotated.mp4) -- every run used to overwrite the
        same file, so the dashboard's Live Analysis page could only ever
        show the most recent run's video. Insert the run_id into the
        filename so each run gets its own file and all of them stay
        listed and selectable.
        """
        if self._annotated_path is not None:
            return self._annotated_path
        base = self.config.path("video.annotated_output")
        if base is None:
            return None
        self._annotated_path = base.with_name(f"{base.stem}_{self.run_id}{base.suffix}")
        return self._annotated_path

    def _open_writer(self, source) -> None:
        if not self.config.get("video.save_annotated", True):
            return
        try:
            import cv2                                           # noqa: PLC0415
            path = self._annotated_output_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            # H.264 first: browsers cannot decode OpenCV's default MPEG-4
            # Part 2 ("mp4v") stream, and the dashboard's Live Analysis page
            # plays this file.  Fall back to mp4v where avc1 is unavailable.
            codecs = [c.strip() for c in str(
                self.config.get("video.codec", "avc1,H264,mp4v")).split(",") if c.strip()]
            self._writer = None
            for codec in codecs:
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*codec),
                    source.fps, (source.width, source.height),
                )
                if writer.isOpened():
                    self._writer = writer
                    self._writer_codec = codec
                    if codec == "mp4v":
                        self.log.warning(
                            "annotated video is being written as MPEG-4 Part 2 "
                            "(mp4v) -- most browsers cannot play it, so the "
                            "dashboard's Live page may show a blank player. "
                            "Install an OpenCV build with H.264 support, or "
                            "transcode: ffmpeg -i %s -c:v libx264 out.mp4", path.name,
                        )
                    else:
                        self.log.info("annotated video codec: %s", codec)
                    break
                writer.release()
            if self._writer is None:
                self.log.warning("could not open video writer at %s", path)
        except Exception as exc:                                 # noqa: BLE001
            self.log.warning("annotated video disabled: %s", exc)
            self._writer = None

    def _make_browser_playable(self) -> None:
        """If the annotated video had to fall back to mp4v, transcode it to
        H.264 with ffmpeg when ffmpeg is available -- otherwise the
        dashboard's Live Analysis page shows an empty player.  Silently
        skipped when ffmpeg is not installed; the warning at open time
        already told the user what to do."""
        if getattr(self, "_writer_codec", None) != "mp4v":
            return
        import shutil, subprocess                                # noqa: PLC0415
        if not shutil.which("ffmpeg"):
            return
        path = self._annotated_output_path()
        if path is None or not path.exists():
            return
        target = path.with_suffix(".h264.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                 str(target)],
                check=True, timeout=600,
            )
            target.replace(path)
            self.log.info("annotated video transcoded to H.264 for the dashboard")
        except Exception as exc:                                 # noqa: BLE001
            self.log.warning("could not transcode the annotated video: %s", exc)
            target.unlink(missing_ok=True)

    def _log_progress(self, total_frames: int) -> None:
        elapsed = max(1e-6, time.time() - self._started_at)
        fps = self.frames_processed / elapsed
        if total_frames:
            pct = 100.0 * self.frames_processed / total_frames
            self.log.info(
                "  %s/%s frames (%.0f%%)  %.1f fps  %s events",
                self.frames_processed, total_frames, pct, fps, self.events_total,
            )
        else:
            self.log.info(
                "  %s frames  %.1f fps  %s events",
                self.frames_processed, fps, self.events_total,
            )

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        log = self.log
        log.info("=" * 66)
        log.info("RUN COMPLETE  %s", summary["run_id"])
        log.info(
            "frames %s in %.1fs (%.1f fps)  events %s",
            summary["frames_processed"], summary["elapsed_seconds"],
            summary["fps"], summary["events_total"],
        )
        for event_type, count in summary["events_by_type"].items():
            log.info("   %-24s %s", event_type, count)
        log.info("-" * 66)
        for stats in summary["modules"]:
            log.info(
                "   %-14s %-12s %5s events  %6.1f ms/frame  %s",
                stats["module"], stats["status"], stats["events"],
                stats["avg_ms_per_frame"],
                stats["detail"][:28] if stats["detail"] else "",
            )
        intel = summary["intelligence"]
        log.info("-" * 66)
        log.info(
            "   intelligence   issues=%s  opened=%s  merged=%s  reopened=%s  resolved=%s",
            intel.get("issues_total", 0), intel.get("issues_opened", 0),
            intel.get("duplicates_merged", 0), intel.get("issues_reopened", 0),
            intel.get("issues_resolved", 0),
        )
        for issue in summary["issues"][:8]:
            log.info(
                "   %-9s %-22s %-8s obs=%-3s buses=%s  P=%s (%s) -> %s",
                issue["issue_id"], issue["class_name"], issue["status"],
                issue["observations"], len(issue["buses"] or []),
                issue["priority_score"], issue["priority_band"], issue["department"],
            )
        log.info("=" * 66)
        for name, path in summary["outputs"].items():
            if path:
                log.info("   %-16s %s", name, path)
        log.info("=" * 66)
