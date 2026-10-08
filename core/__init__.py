"""Core layer of the AI Urban Intelligence Platform.

Everything the modules are allowed to depend on lives here:
the common event schema, the frame context, the module interface,
configuration, logging, GPS helpers and the output sinks.

Modules depend on `core`.  Modules never depend on each other.
"""

from .config import Config, load_config, PROJECT_ROOT
from .event import Event, EventType, GPS, Severity, SourceModule, make_event
from .frame_context import FrameContext
from .logging_setup import get_logger, setup_logging
from .module_base import BaseModule, ModuleStatus, ModuleUnavailable

__all__ = [
    "Config", "load_config", "PROJECT_ROOT",
    "Event", "EventType", "GPS", "Severity", "SourceModule", "make_event",
    "FrameContext",
    "get_logger", "setup_logging",
    "BaseModule", "ModuleStatus", "ModuleUnavailable",
]
