"""
scripts/import_events.py
========================

Load an existing ``events.jsonl`` (the pipeline's own output) into the
platform database, WITHOUT running any model.

Why this exists
---------------
The pipeline normally writes SQLite directly while it runs.  When all you
have is the output of an earlier run (``outputs/runs/events.jsonl`` + the
annotated recordings + ``outputs/evidence/``), for example a fresh checkout
or a machine without the model weights, this rebuilds the database from
that output so the dashboards, the mobile app and the event workflow have
data to work on.

What it writes
--------------
* ``events``              every line of the JSONL, unchanged
* ``vehicles/segments``   derived by ``Database.insert_event`` as usual
* ``issues``              one row per Module 6 issue id, rebuilt from the
                          engine's own ``persistent_issue`` snapshots plus the
                          events that carry that issue id
* ``issue_observations``  one row per event attached to an issue

Nothing is re-detected, re-scored or re-deduplicated: issue ids, priority,
department and severity all come from what the intelligence engine already
wrote.  Managed events (EV-1001 ...) are created by the backend's workflow
sync the next time it starts (or immediately with ``--sync``).

Usage::

    python scripts/import_events.py                  # default paths from config
    python scripts/import_events.py --jsonl other.jsonl --db outputs/demo.db
    python scripts/import_events.py --sync           # also build EV- events now
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config          # noqa: E402
from core.event import Event                 # noqa: E402
from backend.db import Database              # noqa: E402

SEVERITY_RANK = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def _read(path: Path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  line {number}: skipped ({exc})")
    return rows


def _rebuild_issues(rows):
    """issue_id -> issue row, from the engine's snapshots + attached events."""
    snapshots = {}
    attached = defaultdict(list)
    for row in rows:
        issue_id = row.get("issue_id")
        if not issue_id:
            continue
        if row.get("event_type") == "persistent_issue":
            snapshots[issue_id] = row            # later lines win
        else:
            attached[issue_id].append(row)

    issues = {}
    for issue_id in sorted(set(snapshots) | set(attached)):
        snap = snapshots.get(issue_id) or {}
        meta = snap.get("metadata") or {}
        events = sorted(attached.get(issue_id, []), key=lambda r: r.get("timestamp") or "")
        if not events and not snap:
            continue
        first = events[0] if events else snap
        last = events[-1] if events else snap
        buses = []
        for source in [*(meta.get("buses") or []), *(e.get("bus_id") for e in events)]:
            if source and source not in buses:
                buses.append(source)
        severity = max(
            [e.get("severity") or "INFO" for e in events] + [snap.get("severity") or "INFO"],
            key=lambda s: SEVERITY_RANK.get(s, 0),
        )
        latest_scored = next(
            (e for e in reversed(events) if e.get("priority_score") is not None), snap
        )
        gps = (snap.get("gps") or first.get("gps") or {})
        first_epoch = Event.from_dict(first).epoch if first else None
        last_epoch = Event.from_dict(last).epoch if last else None
        state = meta.get("issue_state") or "OPENED"
        status = {"OPENED": "OPEN", "UPDATED": "OPEN", "REOPENED": "REOPENED",
                  "RESOLVED": "RESOLVED"}.get(state, "OPEN")
        issues[issue_id] = {
            "issue_id": issue_id,
            "issue_type": meta.get("issue_type") or first.get("event_type"),
            "class_name": snap.get("class_name") or first.get("class_name"),
            "status": status,
            "severity": severity,
            "first_seen": meta.get("first_seen") or first.get("timestamp"),
            "last_seen": last.get("timestamp") or meta.get("last_seen"),
            "first_seen_epoch": first_epoch,
            "last_seen_epoch": last_epoch,
            "observation_count": max(len(events), int(meta.get("observations") or 0)),
            "bus_count": len(buses) or 1,
            "buses": buses,
            "latitude": gps.get("latitude"),
            "longitude": gps.get("longitude"),
            "road_segment": snap.get("road_segment") or first.get("road_segment"),
            "priority_score": latest_scored.get("priority_score") or meta.get("priority_score") or 0.0,
            "priority_band": ((latest_scored.get("metadata") or {}).get("priority_band")
                              or meta.get("priority_band") or "LOW"),
            "department": snap.get("department") or first.get("department"),
            "clean_passes": int(meta.get("clean_passes") or 0),
            "reopened_count": int(meta.get("reopened_count") or 0),
            "repair_observed": 1 if meta.get("repair_observed") else 0,
            "resolved_at": None,
            "last_event_id": last.get("event_id"),
            "metadata": {"imported_from": "events.jsonl"},
        }
    return issues, attached


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jsonl", help="events file (default: output.events_jsonl)")
    parser.add_argument("--db", help="database (default: backend.database)")
    parser.add_argument("--sync", action="store_true",
                        help="also create the managed events (EV-...) now")
    args = parser.parse_args(argv)

    overrides = {"backend": {"database": args.db}} if args.db else None
    config = load_config(overrides=overrides) if overrides else load_config()
    jsonl = Path(args.jsonl) if args.jsonl else config.path("output.events_jsonl")
    if not jsonl or not jsonl.exists():
        print(f"no events file at {jsonl}")
        return 1
    db_path = config.path("backend.database")
    database = Database(db_path).initialise()

    rows = _read(jsonl)
    print(f"read {len(rows)} events from {jsonl}")
    stored = 0
    for row in rows:
        try:
            database.insert_event(Event.from_dict(row))
            stored += 1
        except Exception as exc:                                 # noqa: BLE001
            print(f"  skipped {row.get('event_id')}: {exc}")
    print(f"stored {stored} events in {db_path}")

    issues, attached = _rebuild_issues(rows)
    existing = {r["issue_id"] for r in database.query("SELECT issue_id FROM issue_observations")}
    for issue_id, issue in issues.items():
        database.upsert_issue(issue)
        if issue_id in existing:
            continue
        for row in attached.get(issue_id, []):
            database.add_issue_observation(issue_id, Event.from_dict(row))
    print(f"rebuilt {len(issues)} issues")

    if args.sync:
        from backend.auth import AuthStore
        from backend.recordings import RecordingIndex
        from backend.workflow import WorkflowStore
        auth = AuthStore(database, config).initialise()
        recordings = RecordingIndex(database, config).initialise()
        workflow = WorkflowStore(database, config, recordings=recordings, auth=auth).initialise()
        stats = workflow.sync(force=True)
        print(f"workflow sync: {stats}")
    database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
