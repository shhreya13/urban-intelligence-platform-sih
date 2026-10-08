"""
core.sinks
==========

Where finished events go.  A sink is anything with ``emit(event)`` and
``close()``.  The orchestrator holds a list of them, so adding "publish to
MQTT when the bus is online" later means writing one class -- nothing in the
modules changes.

Included sinks:
  * ``JsonlSink``     -- append one JSON object per line (always on, the
                         cheapest possible audit trail).
  * ``DatabaseSink``  -- write straight into the platform SQLite DB
                         (backend.mode = "direct", the default).
  * ``HttpSink``      -- POST /events to a running backend
                         (backend.mode = "http", for a distributed setup).
  * ``MemorySink``    -- keep everything in a list (tests / demo summaries).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .event import Event
from .logging_setup import get_logger


class BaseSink:
    name = "sink"

    def emit(self, event: Event) -> None:
        raise NotImplementedError

    def emit_many(self, events: List[Event]) -> None:
        for event in events:
            self.emit(event)

    def close(self) -> None:
        pass


class MemorySink(BaseSink):
    name = "memory"

    def __init__(self) -> None:
        self.events: List[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


class JsonlSink(BaseSink):
    name = "jsonl"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8")
        self.log = get_logger("sink.jsonl")

    def emit(self, event: Event) -> None:
        self._handle.write(event.to_json() + "\n")
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:                                        # noqa: BLE001
            pass


class DatabaseSink(BaseSink):
    """Writes into the same SQLite database the backend API reads from, so a
    pipeline run shows up live on the dashboard with no extra plumbing."""

    name = "database"

    def __init__(self, db_path: str | Path):
        from backend.db import Database          # local import: keeps core/ standalone
        self.db = Database(db_path)
        self.db.initialise()
        self.log = get_logger("sink.database")
        self.written = 0

    def emit(self, event: Event) -> None:
        self.db.insert_event(event)
        self.written += 1

    def close(self) -> None:
        self.db.close()


class HttpSink(BaseSink):
    """POSTs each event to a backend that is already running elsewhere."""

    name = "http"

    def __init__(self, base_url: str, timeout: float = 3.0, ingest_token: Optional[str] = None):
        import requests                                   # noqa: PLC0415
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.log = get_logger("sink.http")
        self.failures = 0
        # The backend requires this header when auth.ingest_token is set
        # (UIP_AUTH__INGEST_TOKEN on both machines).
        self.headers = {"X-Ingest-Token": str(ingest_token)} if ingest_token else {}

    def emit(self, event: Event) -> None:
        try:
            self._requests.post(
                f"{self.base_url}/events",
                json=event.to_dict(),
                headers=self.headers,
                timeout=self.timeout,
            )
        except Exception as exc:                                 # noqa: BLE001
            self.failures += 1
            if self.failures <= 3:
                self.log.warning("POST /events failed (%s) -- event kept in jsonl", exc)


def build_sinks(config) -> List[BaseSink]:
    """Assemble the sink list from configuration."""
    log = get_logger("sink")
    sinks: List[BaseSink] = []

    jsonl_path = config.path("output.events_jsonl")
    if jsonl_path:
        sinks.append(JsonlSink(jsonl_path))

    backend = config.section("backend")
    if backend.get("enabled", True):
        mode = str(backend.get("mode", "direct")).lower()
        if mode == "http":
            try:
                sinks.append(HttpSink(
                    backend.get("url", "http://127.0.0.1:8000"),
                    ingest_token=config.get("auth.ingest_token"),
                ))
            except Exception as exc:                             # noqa: BLE001
                log.warning("HTTP sink unavailable (%s) -- falling back to database", exc)
                mode = "direct"
        if mode == "direct":
            try:
                sinks.append(DatabaseSink(config.path("backend.database")))
            except Exception as exc:                             # noqa: BLE001
                log.warning("database sink unavailable (%s) -- events still go to jsonl", exc)

    return sinks
