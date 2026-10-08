import argparse
import json
from pathlib import Path
from collections import defaultdict
import cv2
import yaml
from ultralytics import YOLO

from src.geometry import bottom_center, normalize_polygon, inside_polygon, distance
from src.risk_engine import RiskEngine

COCO_VEHICLE_NAMES = {"bicycle", "car", "motorcycle", "bus", "truck"}

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--ped-model", required=True)
    p.add_argument("--vehicle-model", required=True)
    p.add_argument("--output", default="outputs/pedestrian_safety.mp4")
    p.add_argument("--events", default="outputs/events.jsonl")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default=None, help="cpu, 0, etc.")
    return p.parse_args()

def extract_tracks(result, allowed_names):
    objects = []
    if result.boxes is None or len(result.boxes) == 0:
        return objects

    names = result.names
    ids = result.boxes.id
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    track_ids = ids.cpu().numpy().astype(int) if ids is not None else list(range(len(xyxy)))

    for box, c, k, tid in zip(xyxy, conf, cls, track_ids):
        name = names[int(k)]
        if name not in allowed_names or c < 0.1:
            continue
        objects.append({
            "id": int(tid),
            "name": name,
            "conf": float(c),
            "box": box.tolist(),
        })
    return objects

def main():
    args = get_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    runtime = cfg["runtime"]

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    event_path = Path(args.events)
    event_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    ped_model = YOLO(args.ped_model)
    vehicle_model = YOLO(args.vehicle_model)

    road_poly = normalize_polygon(cfg["zones"]["road_zone"], w, h)
    crossing_poly = normalize_polygon(cfg["zones"]["crossing_zone"], w, h)

    engine = RiskEngine(cfg)
    frame_idx = 0

    with event_path.open("w", encoding="utf-8") as events_file:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Pedestrian/child tracking
            pr = ped_model.track(
                frame,
                persist=True,
                tracker=runtime["tracker"],
                conf=runtime["confidence"],
                iou=runtime["iou"],
                imgsz=runtime["imgsz"],
                verbose=False,
                device=args.device,
            )[0]

            # Vehicle tracking. For final system use Person 2's model.
            vr = vehicle_model.track(
                frame,
                persist=True,
                tracker=runtime["tracker"],
                conf=runtime["confidence"],
                iou=runtime["iou"],
                imgsz=runtime["imgsz"],
                verbose=False,
                device=args.device,
            )[0]

            pedestrians = extract_tracks(pr, {"person", "child"})
            vehicles = extract_tracks(vr, COCO_VEHICLE_NAMES)

            # Draw zones
            overlay = frame.copy()
            cv2.fillPoly(overlay, [road_poly], (0, 120, 255))
            cv2.fillPoly(overlay, [crossing_poly], (255, 160, 0))
            frame = cv2.addWeighted(overlay, 0.10, frame, 0.90, 0)
            cv2.polylines(frame, [road_poly], True, (0, 140, 255), 2)
            cv2.polylines(frame, [crossing_poly], True, (255, 180, 0), 2)

            # Update histories
            for p in pedestrians:
                c = bottom_center(p["box"])
                p["point"] = c
                p["in_road"] = inside_polygon(c, road_poly)
                p["crossing"] = inside_polygon(c, crossing_poly)
                p["speed"] = engine.update_position(
                    f"P{p['id']}", c[0], c[1], frame_idx
                )

            for v in vehicles:
                c = bottom_center(v["box"])
                v["point"] = c
                v["speed"] = engine.update_position(
                    f"V{v['id']}", c[0], c[1], frame_idx
                )

            # Evaluate pedestrian/vehicle pairs
            for p in pedestrians:
                for v in vehicles:
                    d = distance(p["point"], v["point"])
                    if d > cfg["risk"]["max_pair_distance_px"]:
                        continue

                    closing = engine.closing_speed(f"P{p['id']}", f"V{v['id']}")
                    approaching = closing >= cfg["risk"]["approach_threshold_px_per_frame"]

                    score, level = engine.score(
                        distance=d,
                        approaching=approaching,
                        in_road=p["in_road"],
                        crossing=p["crossing"],
                        pedestrian_type=p["name"],
                        vehicle_speed=v["speed"],
                    )

                    pair_key = f"P{p['id']}-V{v['id']}"
                    if level in {"HIGH", "CRITICAL"} and engine.should_emit(
                        pair_key, runtime["event_cooldown_seconds"]
                    ):
                        event = {
                            "event": "pedestrian_vehicle_risk",
                            "pedestrian_id": f"P{p['id']}",
                            "vehicle_id": f"V{v['id']}",
                            "pedestrian_type": p["name"],
                            "distance_px": round(d, 2),
                            "closing_speed_px_per_frame": round(closing, 3),
                            "approaching": approaching,
                            "in_road": p["in_road"],
                            "crossing": p["crossing"],
                            "risk_score": score,
                            "risk_level": level,
                            "timestamp_s": round(frame_idx / fps, 2),
                        }
                        events_file.write(json.dumps(event) + "\n")
                        events_file.flush()

                    # Draw pair status
                    if level in {"HIGH", "CRITICAL"}:
                        cv2.line(
                            frame,
                            tuple(map(int, p["point"])),
                            tuple(map(int, v["point"])),
                            (0, 0, 255),
                            3,
                        )
                        cv2.putText(
                            frame,
                            f"{level} {score:.0f}",
                            (int(p["point"][0]), max(30, int(p["point"][1]) - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

            # Draw pedestrian boxes
            for p in pedestrians:
                x1, y1, x2, y2 = map(int, p["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"P{p['id']} {p['name']} {p['conf']:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # Draw vehicle boxes
            for v in vehicles:
                x1, y1, x2, y2 = map(int, v["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"V{v['id']} {v['name']} {v['speed']:.1f}px/f"
                cv2.putText(frame, label, (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

            cv2.putText(frame, "PERSON 3 - PEDESTRIAN SAFETY",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_idx}",
                        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()

    print(f"Saved annotated video: {out_path}")
    print(f"Saved risk events: {event_path}")

if __name__ == "__main__":
    main()
