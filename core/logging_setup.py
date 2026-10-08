"""
core.logging_setup
==================

One place that configures logging for the whole platform, so that when
something breaks you can always tell WHICH module it was.

Every log line is prefixed with the logger name, e.g.::

    12:31:02 INFO  module.road_detection  | ready (owner: Person 1)
    12:31:19 ERROR module.anpr            | frame 412 failed: RuntimeError: ...
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_ROOT = "uip"


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        if name.startswith(_ROOT + "."):
            name = name[len(_ROOT) + 1:]
        record.short_name = name[:24].ljust(24)
        return super().format(record)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    global _CONFIGURED
    logger = logging.getLogger(_ROOT)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    fmt = _Formatter(
        "%(asctime)s %(levelname)-5s %(short_name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # Ultralytics is extremely chatty at INFO.
    for noisy in ("ultralytics", "urllib3", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("YOLO_VERBOSE", "False")

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"{_ROOT}.{name}")
