# Traffic Intelligence Module (Person 2)

Standalone traffic & road analytics module for the AI Urban Intelligence Platform.

Pipeline (final target):

```
LAPTOP WEBCAM -> VEHICLE DETECTION -> BoT-SORT TRACKING -> ROAD ROI
-> VEHICLE COUNTING -> TRAFFIC ANALYTICS -> LIVE OUTPUT / JSON
```

## Status

- [x] Phase 1: Laptop webcam capture
- [ ] Phase 2: Vehicle detection
- [ ] Phase 3: Detection quality test
- [ ] Phase 4: BoT-SORT tracking
- [ ] Phase 5: Road ROI
- [ ] Phase 6: Vehicle counting
- [ ] Phase 7: Counting accuracy validation
- [ ] Phase 8: Vehicle classification
- [ ] Phase 9: Traffic density
- [ ] Phase 10: Traffic level
- [ ] Phase 11: Speed estimation
- [ ] Phase 12: Congestion detection
- [ ] Phase 13: Bottleneck detection
- [ ] Phase 14: Peak traffic analysis
- [ ] Phase 15: Delay estimation
- [ ] Phase 16: Traffic engine
- [ ] Phase 17: Live visualization (full overlay)
- [ ] Phase 18: Full configuration wiring
- [ ] Phase 19: JSON output
- [ ] Phase 20: Testing
- [ ] Phase 21: Performance/Jetson Nano optimization
- [ ] Phase 22: Benchmarking

## Setup

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

Press `q` in the video window to quit.

## Configuration

All tunable parameters live in `config/traffic_config.yaml`. Nothing should be
hardcoded in the source files — if you find yourself typing a number directly
into a `.py` file, it probably belongs in the config instead.
