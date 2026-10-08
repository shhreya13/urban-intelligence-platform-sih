"""
backend.recordings
==================

Evidence video for an event = the annotated recording the pipeline already
wrote (``outputs/runs/annotated_<run_id>.mp4``), positioned at the event's
frame.  The bounding boxes and labels in that recording were drawn by the
pipeline at inference time; nothing here runs a model or draws a box.

Linking an event to its recording
---------------------------------
1. ``run_id`` -- the orchestrator now stamps ``metadata.run_id`` on every
   event, so new runs link exactly.
2. Events recorded before that change carry no run id.  For those the index
   narrows the candidate recordings by physical consistency (the event's
   ``frame_number / fps`` must equal its ``video_time_s``, the frame must
   exist, the bounding box must fit inside the frame, the run's time window
   must contain the event) and, if more than one recording still qualifies,
   compares the event's own evidence crop with the same region of each
   candidate frame.  Anything below ``min_match_score`` is left unlinked
   rather than guessed.

Results are cached in the ``evidence_links`` table.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LINK_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_links (
    event_id        TEXT PRIMARY KEY,
    video_name      TEXT,
    frame_index     INTEGER,
    fps             REAL,
    method          TEXT,
    score           REAL,
    recordings_sig  TEXT,
    resolved_at     REAL
);
"""

_RUN_RE = re.compile(r"_([0-9a-f]{8,32})\.mp4$", re.IGNORECASE)


def _epoch(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    try:
        value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        try:
            value = datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


class RecordingIndex:
    def __init__(self, database, config, log=None):
        self.db = database
        self.log = log
        self.runs_dir = (config.path("output.events_jsonl") or Path("outputs/runs/x")).parent
        self.evidence_dir = config.path("backend.evidence_dir") or Path("outputs/evidence")
        self.frames_dir = config.path("workflow.frame_cache_dir", "outputs/cache/frames")
        section = config.section("workflow")
        self.min_match_score = float(section.get("min_match_score", 0.5))
        self.clip_before_s = float(section.get("clip_before_s", 3.0))
        self.clip_after_s = float(section.get("clip_after_s", 4.0))
        self._probe_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def initialise(self) -> "RecordingIndex":
        self.db.executescript(LINK_SCHEMA)
        return self

    # -- recordings ---------------------------------------------------------

    def recordings(self) -> List[Dict[str, Any]]:
        if not self.runs_dir.exists():
            return []
        out = []
        for path in sorted(self.runs_dir.glob("*.mp4")):
            info = self._probe(path)
            if info:
                out.append(info)
        return out

    def signature(self) -> str:
        return "|".join(sorted(r["name"] for r in self.recordings()))

    def _probe(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = path.name
        cached = self._probe_cache.get(key)
        if cached and cached["_sig"] == (stat.st_size, stat.st_mtime):
            return cached
        info: Dict[str, Any] = {
            "name": path.name, "path": str(path), "size_mb": round(stat.st_size / 1e6, 2),
            "fps": None, "frames": None, "width": None, "height": None,
            "run_id": None, "_sig": (stat.st_size, stat.st_mtime),
        }
        match = _RUN_RE.search(path.name)
        if match:
            info["run_id"] = match.group(1)
        try:
            import cv2                                          # noqa: PLC0415
            capture = cv2.VideoCapture(str(path))
            info["fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0) or None
            info["frames"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
            info["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
            info["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
            capture.release()
        except Exception as exc:                                 # noqa: BLE001
            if self.log:
                self.log.debug("could not probe %s: %s", path.name, exc)
        self._probe_cache[key] = info
        return info

    def by_name(self, name: str) -> Optional[Dict[str, Any]]:
        path = (self.runs_dir / name)
        if not path.exists() or path.suffix.lower() != ".mp4":
            return None
        return self._probe(path)

    # -- event -> recording -------------------------------------------------

    def resolve(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the evidence-video descriptor for an event, or None."""
        if not event or event.get("frame_number") is None:
            return None
        event_id = event["event_id"]
        signature = self.signature()
        cached = self.db.query_one("SELECT * FROM evidence_links WHERE event_id = ?", (event_id,))
        if cached and (cached["method"] != "none" or cached["recordings_sig"] == signature):
            if cached["method"] == "none":
                return None
            info = self.by_name(cached["video_name"])
            if info is not None:
                return self._descriptor(event, info, cached["frame_index"], cached["method"], cached["score"])

        with self._lock:
            link = self._link(event)
        self.db.execute(
            "INSERT OR REPLACE INTO evidence_links(event_id, video_name, frame_index, fps, "
            "method, score, recordings_sig, resolved_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                event_id,
                link["info"]["name"] if link else None,
                link["frame_index"] if link else None,
                link["info"]["fps"] if link else None,
                link["method"] if link else "none",
                link.get("score") if link else None,
                signature, time.time(),
            ),
        )
        if not link:
            return None
        return self._descriptor(event, link["info"], link["frame_index"], link["method"], link.get("score"))

    def _descriptor(self, event, info, frame_index, method, score) -> Dict[str, Any]:
        fps = info.get("fps") or 25.0
        at = event.get("video_time_s")
        if at is None:
            at = frame_index / fps
        return {
            "name": info["name"],
            "url": f"/videos/{info['name']}",
            "run_id": info.get("run_id"),
            "frame_index": frame_index,
            "fps": round(fps, 3),
            "event_time_s": round(float(at), 3),
            "start_s": round(max(0.0, float(at) - self.clip_before_s), 3),
            "end_s": round(float(at) + self.clip_after_s, 3),
            "width": info.get("width"),
            "height": info.get("height"),
            "link_method": method,
            "match_score": round(score, 3) if score is not None else None,
        }

    def _run_id(self, event: Dict[str, Any]) -> Optional[str]:
        return event.get("run_id") or (event.get("metadata") or {}).get("run_id")

    def _link(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        frame = int(event["frame_number"])
        recordings = self.recordings()

        run_id = self._run_id(event)
        if run_id:
            for info in recordings:
                if info.get("run_id") == run_id or run_id in info["name"]:
                    return {"info": info, "frame_index": frame, "method": "run_id"}
            return None      # a run id we cannot find must not fall back to a guess

        candidates = [info for info in recordings if self._consistent(event, info)]
        candidates = self._narrow_by_runs(event, candidates)
        if not candidates:
            return None
        if len(candidates) == 1 and not event.get("evidence_image"):
            return {"info": candidates[0], "frame_index": frame, "method": "consistency", "score": None}

        scored = []
        for info in candidates:
            score = self._similarity(event, info, frame)
            if score is not None:
                scored.append((score, info.get("frames") or 0, info["name"], info))
        if not scored:
            if len(candidates) == 1:
                return {"info": candidates[0], "frame_index": frame, "method": "consistency", "score": None}
            return None
        scored.sort(key=lambda s: (round(s[0], 2), s[1], s[2]), reverse=True)
        best = scored[0]
        if best[0] < self.min_match_score:
            return None
        return {"info": best[3], "frame_index": frame, "method": "matched", "score": best[0]}

    @staticmethod
    def _consistent(event: Dict[str, Any], info: Dict[str, Any]) -> bool:
        fps, frames = info.get("fps"), info.get("frames")
        if not fps or not frames:
            return False
        frame = int(event["frame_number"])
        if frame >= frames:
            return False
        video_time = event.get("video_time_s")
        if video_time is not None:
            tolerance = max(0.04, 1.0 / fps)
            if abs(frame / fps - float(video_time)) > tolerance:
                return False
        box = event.get("bounding_box")
        if box and info.get("width") and info.get("height"):
            if box[2] > info["width"] + 2 or box[3] > info["height"] + 2:
                return False
        return True

    def _narrow_by_runs(self, event, candidates):
        if len(candidates) <= 1:
            return candidates
        ts = _epoch(event.get("timestamp"))
        if ts is None:
            return candidates
        runs = self.db.query("SELECT run_id, bus_id, started_at, finished_at FROM runs")
        in_window = set()
        for run in runs:
            start = _epoch(run.get("started_at"))
            end = _epoch(run.get("finished_at")) or time.time()
            if start is None:
                continue
            if run.get("bus_id") and event.get("bus_id") and run["bus_id"] != event["bus_id"]:
                continue
            if start - 5 <= ts <= end + 5:
                in_window.add(run["run_id"])
        if not in_window:
            return candidates
        narrowed = [c for c in candidates if c.get("run_id") in in_window]
        return narrowed or candidates

    def _similarity(self, event, info, frame_index) -> Optional[float]:
        crop_path = event.get("evidence_image")
        box = event.get("bounding_box")
        if not crop_path or not box:
            return None
        try:
            import cv2                                          # noqa: PLC0415
            import numpy as np                                  # noqa: PLC0415
        except ImportError:
            return None
        crop = cv2.imread(str(self.evidence_dir / crop_path), cv2.IMREAD_GRAYSCALE)
        frame = self._read_frame(info["path"], frame_index)
        if crop is None or frame is None or crop.size == 0:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        pad_x, pad_y = int((x2 - x1) * 0.12), int((y2 - y1) * 0.12)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        region = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        region = cv2.resize(region, (crop.shape[1], crop.shape[0]))
        a = crop.astype("float32").ravel()
        b = region.astype("float32").ravel()
        if a.std() < 1e-3 or b.std() < 1e-3:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _read_frame(path: str, frame_index: int):
        try:
            import cv2                                          # noqa: PLC0415
        except ImportError:
            return None
        capture = cv2.VideoCapture(path)
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
            ok, frame = capture.read()
            return frame if ok else None
        finally:
            capture.release()

    # -- annotated frame extraction ----------------------------------------

    def frame_file(self, video_name: str, frame_index: int) -> Optional[Path]:
        """The annotated frame, exactly as the pipeline drew it, as a JPEG."""
        info = self.by_name(video_name)
        if info is None:
            return None
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        target = self.frames_dir / f"{Path(video_name).stem}_{int(frame_index):06d}.jpg"
        if target.exists():
            return target
        frame = self._read_frame(info["path"], frame_index)
        if frame is None:
            return None
        try:
            import cv2                                          # noqa: PLC0415
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not ok:
                return None
            target.write_bytes(buffer.tobytes())
            return target
        except Exception as exc:                                 # noqa: BLE001
            if self.log:
                self.log.warning("could not extract frame %s from %s: %s", frame_index, video_name, exc)
            return None
