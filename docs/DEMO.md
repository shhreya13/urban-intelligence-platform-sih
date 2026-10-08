# Demo procedure

Three ways to demonstrate the platform, in increasing order of how much they
show — and each is explicit about what is real.

---

## 1. The honest run (default)

```bash
python run_pipeline.py --source data/videos/<your_clip>.mp4 --resize-width 960
python run_backend.py            # http://127.0.0.1:8000
```

Everything on the dashboard is a detection. On footage without a crash the
Hit-and-Run page will be empty, and that is the correct answer — the empty
state on that page explains why.

**This is the mode to use with real bus footage.**

---

## 2. `--demo-incident` — exercise the chain on real footage

```bash
python run_pipeline.py --source data/videos/<clip>.mp4 --demo-incident
```

Injects **one** synthetic collision onto a real, currently-tracked vehicle.
From that point everything is measured for real: the state machine locks the
real track, the escape distance is measured on its real trajectory, the
evidence crops are real frames, and ANPR reads the real vehicle.

Because the departure is real, the outcome is honest — on most footage the
case resolves to `TRACK_LOST` (the tracker dropped the vehicle) or
`COLLISION_ONLY` (it did not travel far enough). That is the anti-false-positive
machinery doing its job, and it is worth showing.

Every event carries `metadata.synthetic = true` and a **SYNTHETIC** badge on
the dashboard.

---

## 3. `scripts/demo_hit_and_run.py` — the full investigation walkthrough

```bash
python scripts/demo_hit_and_run.py --background data/videos/street.mp4
python run_backend.py            # then open http://127.0.0.1:8000/#hitrun
```

A scripted two-vehicle scenario drawn onto real road frames, driven through
the **real** modules: Person 4's `CollisionDetector` and `HitAndRunDetector`,
the real state machine, the real evidence store, the real ANPR trigger, the
real intelligence engine and the real database.

What is scripted: the two vehicles' positions. Nothing else.

It produces the complete chain, which is what makes the investigation page
worth showing:

```
possible_collision  → collision (CRITICAL)
  → POST_COLLISION_MONITORING → VEHICLE_LEAVING → HIT_AND_RUN_CONFIRMED
  → ANPR requested → vehicle_identification
  → persistent issue HNR-001, priority 70, routed to Traffic Police
```

---

## Suggested five-minute walkthrough

1. **Dashboard** — KPI row, the GIS map, events by type, traffic over the run.
2. **Live Analysis** — play the annotated video; click an event in the feed
   and the video seeks to it.
3. **Incidents** — filter to Hit-and-run; click the row.
4. **Hit-and-Run** — the investigation page: state timeline with video
   timestamps, before / impact / after / vehicle / plate evidence strip,
   escape distance and speed, the detector that produced it.
5. **Vehicles / ANPR** — point out that ANPR ran once, only because the
   incident asked, and that a low-confidence read is published as `UNKNOWN`
   rather than a guess.
6. **Persistent Issues** — the same problem seen by several buses is one
   issue with a rising priority score.
7. **Traffic** and **Road Assets** — charts and registers built from the same
   event stream.

## What to say about the empty states

If a page is empty, that is information, not a bug. The Hit-and-Run page
says so directly: a hit-and-run requires a confirmed collision *followed by a
sustained departure from the incident point*, and a vehicle that merely
disappears never qualifies.

---

## 4. `--dashboard`: the multi-department platform, no model

```bash
./demo.sh --dashboard          # Windows: demo.bat --dashboard
```

Loads the pipeline output already in `outputs/` (149 detections, the
annotated recordings, the evidence crops), builds the 24 managed events and
starts both apps. Nothing is re-detected and no box is ever redrawn: the
frames on screen are cut from the pipeline's own annotated recordings.

| App | Address |
|---|---|
| Web dashboard | <http://127.0.0.1:8000/> |
| Mobile Command & Field app | <http://127.0.0.1:8000/mobile/> |

Accounts: `admin@`, `traffic@`, `roads@`, `safety@`, `transport@`, `field@`
at `urbanintel.local`, password `UrbanIntel@2026`.

### Five-minute platform walkthrough

1. **Central Command** (`admin@`): one screen over every department, with 24
   managed events, the GIS map with all layers, fleet, users.
2. **A department** (`roads@`): the same backend, a smaller world. Only
   road events, only road map layers, no ANPR and no user management. Type
   `#users` in the address bar to show the guard sending you back.
3. **The phone** (`roads@` at `/mobile/`): open the pothole EV-1007, view
   the annotated detection frame, play the evidence video and jump to the
   event frame.
4. **The workflow**: verify, assign to the field officer, then sign in as
   `field@`, start work, mark resolved with a note.
5. **Back to Central Command** on the web, without refreshing. The event
   is RESOLVED with the full activity log, seconds later.

The 20 numbered steps with their expected results are in
[ACCEPTANCE_TEST.md](ACCEPTANCE_TEST.md).

### What to say about the platform's empty states

Traffic owns no managed events because the processed footage contains
traffic observations but no confirmed congestion or bottleneck. Buses 02 and
03 have never reported because only BUS-01 ran. Those pages say so in
words. Nothing is filled in with invented data.
