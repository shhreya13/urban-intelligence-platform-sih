"""
core.module_base
================

The clean common interface every team module is wrapped in.

    module.setup()                 -> called once, may raise
    module.process(ctx)            -> List[Event]
    module.finalize()              -> List[Event]   (end-of-video flush)
    module.close()

Rules enforced here (integration spec 13/14/15):
  * A module that cannot load its model is marked UNAVAILABLE and is simply
    skipped -- it never takes the pipeline down.
  * An exception inside process() is caught, logged with the module name,
    counted, and the pipeline continues with the remaining modules.
  * A module that keeps failing is disabled so a broken module cannot spam
    the log for a whole video.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional

from .event import Event
from .frame_context import FrameContext
from .logging_setup import get_logger


class ModuleStatus:
    UNINITIALISED = "UNINITIALISED"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"      # dependency/model missing -> skipped
    DISABLED = "DISABLED"            # switched off in config
    FAILED = "FAILED"                # too many runtime errors -> skipped


class ModuleUnavailable(Exception):
    """Raised by setup() when a model / dependency is missing.

    This is a *soft* failure: the pipeline logs it and carries on without
    the module instead of crashing."""


class BaseModule:
    name = "base"
    #: which Person owns the original implementation (for logging / docs)
    owner = "-"

    MAX_CONSECUTIVE_ERRORS = 10

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.log = get_logger(f"module.{self.name}")
        self.status = ModuleStatus.UNINITIALISED
        self.status_detail = ""
        self.error_count = 0
        self._consecutive_errors = 0
        self.frames_processed = 0
        self.events_emitted = 0
        self.total_time_s = 0.0

    # ---- lifecycle to be implemented by subclasses ------------------------

    def setup(self) -> None:
        """Load models / allocate resources.  Raise ModuleUnavailable if the
        module cannot run at all in this environment."""

    def process(self, ctx: FrameContext) -> List[Event]:
        return []

    def finalize(self) -> List[Event]:
        """Called once when the video ends -- lets a module flush anything it
        was still accumulating (e.g. an ongoing congestion window)."""
        return []

    def close(self) -> None:
        pass

    # ---- safe wrappers used by the orchestrator ---------------------------

    def safe_setup(self) -> bool:
        if self.config.get("enabled", True) is False:
            self.status = ModuleStatus.DISABLED
            self.status_detail = "disabled in config"
            self.log.info("disabled in config -- skipping")
            return False
        try:
            self.setup()
            self.status = ModuleStatus.READY
            self.log.info("ready (owner: %s)", self.owner)
            return True
        except ModuleUnavailable as exc:
            self.status = ModuleStatus.UNAVAILABLE
            self.status_detail = str(exc)
            self.log.warning("UNAVAILABLE -- %s. Pipeline continues without it.", exc)
            return False
        except Exception as exc:                                # noqa: BLE001
            self.status = ModuleStatus.UNAVAILABLE
            self.status_detail = f"{type(exc).__name__}: {exc}"
            self.log.error("setup failed: %s", self.status_detail)
            self.log.debug("%s", traceback.format_exc())
            return False

    def safe_process(self, ctx: FrameContext) -> List[Event]:
        if self.status != ModuleStatus.READY:
            return []
        started = time.perf_counter()
        try:
            events = self.process(ctx) or []
            self._consecutive_errors = 0
        except Exception as exc:                                # noqa: BLE001
            self.error_count += 1
            self._consecutive_errors += 1
            self.log.error(
                "frame %s failed: %s: %s", ctx.frame_number, type(exc).__name__, exc
            )
            self.log.debug("%s", traceback.format_exc())
            if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                self.status = ModuleStatus.FAILED
                self.status_detail = (
                    f"{self._consecutive_errors} consecutive errors, last: {exc}"
                )
                self.log.error(
                    "disabled after %s consecutive errors -- rest of the "
                    "pipeline keeps running", self._consecutive_errors
                )
            return []
        finally:
            self.total_time_s += time.perf_counter() - started
            self.frames_processed += 1

        for event in events:
            event.with_context(ctx)
        self.events_emitted += len(events)
        return events

    def safe_finalize(self) -> List[Event]:
        if self.status not in (ModuleStatus.READY, ModuleStatus.FAILED):
            return []
        try:
            return self.finalize() or []
        except Exception as exc:                                # noqa: BLE001
            self.log.error("finalize failed: %s", exc)
            return []

    def safe_close(self) -> None:
        try:
            self.close()
        except Exception as exc:                                # noqa: BLE001
            self.log.warning("close failed: %s", exc)

    # ---- reporting --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        avg_ms = (
            (self.total_time_s / self.frames_processed) * 1000.0
            if self.frames_processed else 0.0
        )
        return {
            "module": self.name,
            "owner": self.owner,
            "status": self.status,
            "detail": self.status_detail,
            "frames": self.frames_processed,
            "events": self.events_emitted,
            "errors": self.error_count,
            "avg_ms_per_frame": round(avg_ms, 2),
        }
