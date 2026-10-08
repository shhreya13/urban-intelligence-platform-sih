# Acceptance test: 20 steps

A scripted run-through of the whole platform, from a clean database to a
pothole resolved on a phone and seen as RESOLVED on Central Command's web
screen. It takes about 10 minutes and needs no model, no GPU and no
internet.

Every expected result below was observed on the shipped data
(`outputs/runs/events.jsonl`, 149 detections from BUS-01 on 17 Sep 2026).
Event numbers are stable: the same import always produces EV-1001 to
EV-1024.

## Before you start

```bash
python -m pip install -r requirements.txt
rm -f outputs/urban_intelligence.db          # Windows: del outputs\urban_intelligence.db
./demo.sh --dashboard                        # Windows: demo.bat --dashboard
```

Leave that running. Open two browser windows:

* **Web**: <http://127.0.0.1:8000/>
* **Mobile**: <http://127.0.0.1:8000/mobile/> in a phone-sized window
  (in Chrome or Edge press F12, then Ctrl+Shift+M and pick a phone), or on
  a real phone as described in step 9.

All six accounts use the password `UrbanIntel@2026`.

| # | Do this | Expect |
|---|---|---|
| 1 | Open the web app. | The login screen. No dashboard data is visible behind it. Refreshing does not get past it. |
| 2 | Sign in as `admin@urbanintel.local` (Central Command). | The command dashboard, headed "Fleet-wide situational overview", with 149 events and 24 open. The sidebar has all four groups: Overview, Operations, Infrastructure, Administration. The header shows the name, the role and a notification bell with a count. |
| 3 | Open **Event Management**. | 24 managed events, EV-1001 to EV-1024, all DETECTED. The summary shows 18 Public Safety and 6 Roads & Infrastructure. |
| 4 | Open **Users & Departments**. | Six accounts with their roles and departments. Central Command can create users, deactivate them and reset passwords. |
| 5 | Sign out. Sign in as `traffic@urbanintel.local`. | The Traffic Department overview. The sidebar has only Overview, Traffic Analytics, Traffic Map, Traffic Issues, Evidence and Notifications: no Roads, Safety, Users or Fleet. The heatmap and vehicle density are drawn from the 16 traffic observations, while the issue panels say "No congestion or bottleneck has been confirmed in the processed footage" and route delays say "Nothing recorded yet". The run contains no congestion or bottleneck events, and nothing is invented to fill the page. |
| 6 | In the address bar, replace the page with `#users` and press Enter. | You are sent straight back to the Traffic home with "Not available for your role". The page never renders. |
| 7 | Sign out. Sign in as `safety@urbanintel.local`. | The Public Safety home, 18 cases. Incidents, hit-and-run and ANPR are available. Road assets is not in the sidebar, and typing `#road` redirects. |
| 8 | Sign out. Sign in as `transport@urbanintel.local`, open an event. | Fleet home with BUS-01 to BUS-03. Events open read-only: the lifecycle buttons are all disabled, with "View only: ... acts on this event" underneath. |
| 9 | Switch to the mobile window and sign in as `roads@urbanintel.local`. (On a real phone: start the server with `python run_backend.py --host 0.0.0.0` and open `http://<computer's LAN IP>:8000/mobile/` on the same Wi-Fi.) | The Command & Field login screen, then the Roads home. The bottom bar has five tabs: Roads, Nearby, Issues, Alerts, Me. The Alerts tab carries an unread badge. |
| 10 | Tap **Issues**, then open **EV-1007 Pothole**. | The event screen: Event ID EV-1007, type POTHOLE, location 13.08381, 80.27212 on SEG-13.08390-80.27197, BUS-01 / CAM-FRONT, 50% confidence, detected 17 Sep 2026, status DETECTED. |
| 11 | Tap the evidence picture, or **VIEW DETECTION FRAME**. | Full-screen frame 135 of `annotated_7baca60ac0b4.mp4` with the pipeline's own `pothole 0.50` box and its telemetry overlay, exactly as the pipeline drew it at inference time. Nothing is drawn by the dashboard. The caption names the recording, the frame and the time. |
| 12 | Close it and tap **PLAY EVIDENCE VIDEO**, then **JUMP TO EVENT FRAME**. | The annotated recording plays, opening about 3 s before the event; the jump lands at about 4.5 s, inside the event frame, with the box on screen. Seeking works without downloading the whole file. |
| 13 | Tap **VERIFY**. | The status becomes VERIFIED, the progress bar advances one step, VERIFY is ticked and disabled, ASSIGN becomes available, and the activity log gains "Verified by Roads Engineer". |
| 14 | Tap **ASSIGN**. | A bottom sheet lists people in Roads & Infrastructure with their open task counts: Field Officer (Roads) and Roads Engineer. The field officer is preselected. |
| 15 | Add the note "Please inspect and patch today" and tap **CONFIRM ASSIGNMENT**. | The status becomes ASSIGNED, the assignee is shown on the event, and START WORK becomes available. |
| 16 | Sign out on the phone. Sign in as `field@urbanintel.local`. | The Tasks home shows EV-1007 as an assigned task, and the Alerts badge has gone up. Alerts contains "EV-1007 assigned to you" with the note. |
| 17 | Open EV-1007 and tap **START WORK**. | The status becomes IN PROGRESS. (As a check that steps cannot be skipped, MARK RESOLVED was disabled before this.) |
| 18 | Tap **MARK RESOLVED**, type "Pothole filled with cold mix and compacted" and confirm. | The status becomes RESOLVED, the progress bar is complete, the resolution note appears on the event, and the buttons are replaced by "Resolved. It re-opens automatically if a bus detects it again." |
| 19 | Go back to the web window (still signed in as Central Command) and wait a few seconds without refreshing. Open **Event Management**, choose the **Resolved** filter and open EV-1007. | The event is RESOLVED. The activity log shows all five entries with who and when: DETECTED by BUS-01 / CAM-FRONT, VERIFIED by Roads Engineer, ASSIGNED to Field Officer (Roads) with the note, IN_PROGRESS by Field Officer (Roads), RESOLVED by Field Officer (Roads) with the resolution note. The evidence tabs show the same annotated frame and recording as the phone. |
| 20 | While still signed in as Central Command, open the same event ID as a Roads user in a private window (`roads@urbanintel.local`), then try a Public Safety event ID such as EV-1001 in the address bar. | Roads sees EV-1007 normally and gets "event not found" for EV-1001, so another department's event ID cannot be opened or acted on, and its existence is not confirmed. |

## What each step demonstrates

| Steps | Requirement |
|---|---|
| 1, 2 | Shared login; no data without a session |
| 2 to 8 | Central Command versus department dashboards; role-aware navigation and map layers |
| 5 | Honest empty states: nothing is invented where the run has no data |
| 6, 7, 20 | URL tampering blocked on the client and, more importantly, on the server |
| 8 | Read-only department (Transport Authority) |
| 9, 10 | Mobile app, per-role bottom navigation, notification badges, event cards |
| 11, 12 | Evidence is the pipeline's own annotated output; no new detection, no redrawn boxes |
| 13 to 18 | Full lifecycle with assignment, notes, notifications and an audit trail; steps cannot be skipped |
| 16 | Field officer sees only what is assigned to them, and is notified personally |
| 19 | Web and mobile share one backend and one database; live refresh without reloading |

## Automated equivalent

The same behaviour is covered by tests, so this can be checked without a
browser:

```bash
python -m pytest tests/test_platform_rbac_workflow.py -q     # 28 tests
python -m pytest tests -q                                    # whole suite
```

The whole suite reports 90 passed, 1 skipped and 2 failed. The skip is the
model smoke test (`ultralytics` not installed). The 2 failures,
`test_incident_engine.py::test_case5_hit_and_run_requests_identification`
and `::test_case6_low_confidence_ocr_returns_unknown`, were already failing
in the project as received and are unrelated to the platform work.
