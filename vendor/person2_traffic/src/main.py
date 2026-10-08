import cv2
import yaml
import time

from detection.vehicle_detector import VehicleDetector
from analytics.traffic_density import TrafficDensity
from analytics.congestion import CongestionAnalyzer
from analytics.speed_estimator import SpeedEstimator
from analytics.bottleneck import BottleneckAnalyzer
from analytics.peak_traffic import PeakTrafficAnalyzer


def load_config():

    config_path = "config/traffic_config.yaml"

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def draw_detections(frame, detections):

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        class_name = detection["class_name"]

        confidence = detection["confidence"]

        track_id = detection["track_id"]

        speed = detection.get("speed", 0.0)

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Center point
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        cv2.circle(
            frame,
            (center_x, center_y),
            4,
            (255, 0, 0),
            -1
        )

        # Label
        if track_id is not None:

            label = (
                f"{class_name} "
                f"ID:{track_id} "
                f"{confidence:.2f} "
                f"{speed:.1f}px/s"
            )

        else:

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )[0]

        text_width = text_size[0]
        text_height = text_size[1]

        # Label background
        cv2.rectangle(
            frame,
            (
                x1,
                max(
                    0,
                    y1 - text_height - 10
                )
            ),
            (
                x1 + text_width + 5,
                y1
            ),
            (0, 255, 0),
            -1
        )

        # Label text
        cv2.putText(
            frame,
            label,
            (
                x1 + 2,
                max(15, y1 - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return frame


def get_vehicle_counts(detections):

    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    for detection in detections:

        class_name = detection["class_name"]

        if class_name in counts:

            counts[class_name] += 1

    return counts


def draw_statistics(
    frame,
    detections,
    density_analyzer,
    congestion_result,
    bottleneck_result,
    peak_result,
    average_speed,
    moving_count,
    slow_count,
    display_fps,
    inference_fps
):

    counts = get_vehicle_counts(
        detections
    )

    total = sum(
        counts.values()
    )

    # Density
    density_percentage = (
        density_analyzer.percentage(
            total
        )
    )

    # Traffic level
    traffic_level = (
        density_analyzer.traffic_level(
            total
        )
    )

    congestion_status = (
        congestion_result["status"]
    )

    congestion_duration = (
        congestion_result[
            "duration_seconds"
        ]
    )

    bottleneck_status = (
        bottleneck_result["status"]
    )

    strongest_zone = (
        bottleneck_result["strongest_zone"]
    )

    if strongest_zone is None:
        strongest_zone = "NONE"

    peak_vehicles = (
        peak_result["peak_vehicles"]
    )

    peak_time = (
        peak_result["peak_time"]
    )

    # -------------------------------------------------
    # DASHBOARD
    # -------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (600, 690),
        (0, 0, 0),
        -1
    )

    # Title
    cv2.putText(
        frame,
        "TRAFFIC ANALYTICS",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # FPS
    cv2.putText(
        frame,
        f"Display FPS: {display_fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"AI FPS: {inference_fps:.1f}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Vehicle counts
    cv2.putText(
        frame,
        f"Cars: {counts['car']}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Motorcycles: {counts['motorcycle']}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Buses: {counts['bus']}",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Trucks: {counts['truck']}",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # Total
    cv2.putText(
        frame,
        f"Total Vehicles: {total}",
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Density
    cv2.putText(
        frame,
        f"Traffic Density: {density_percentage:.1f}%",
        (20, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # Traffic level
    cv2.putText(
        frame,
        f"Traffic Level: {traffic_level}",
        (20, 325),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Congestion
    cv2.putText(
        frame,
        f"Congestion: {congestion_status}",
        (20, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Duration: {congestion_duration:.1f}s",
        (20, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Speed
    cv2.putText(
        frame,
        f"Avg Relative Speed: {average_speed:.1f}px/s",
        (20, 425),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Moving
    cv2.putText(
        frame,
        f"Moving Vehicles: {moving_count}",
        (20, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Slow
    cv2.putText(
        frame,
        f"Slow Vehicles: {slow_count}",
        (20, 475),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Movement
    if average_speed <= 15:
        movement = "LOW SPEED"
    else:
        movement = "MOVING"

    cv2.putText(
        frame,
        f"Movement: {movement}",
        (20, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Bottleneck
    cv2.putText(
        frame,
        f"Bottleneck: {bottleneck_status}",
        (20, 535),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # Hot zone
    cv2.putText(
        frame,
        f"Hot Zone: {strongest_zone}",
        (20, 565),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # -------------------------------------------------
    # PEAK TRAFFIC
    # -------------------------------------------------

    cv2.putText(
        frame,
        f"Peak Vehicles: {peak_vehicles}",
        (20, 600),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Peak Time: {peak_time}",
        (20, 630),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    return frame


def main():

    print("=" * 60)
    print("AI URBAN INTELLIGENCE PLATFORM")
    print("TRAFFIC & ROAD ANALYTICS")
    print("=" * 60)

    # -------------------------------------------------
    # CONFIG
    # -------------------------------------------------

    config = load_config()

    camera_config = config["camera"]

    detection_config = config["detection"]

    camera_source = camera_config["source"]

    camera_width = camera_config["width"]

    camera_height = camera_config["height"]

    model_path = detection_config["model"]

    confidence_threshold = (
        detection_config[
            "confidence_threshold"
        ]
    )

    image_size = detection_config["image_size"]

    frame_skip = detection_config.get(
        "frame_skip",
        0
    )

    # -------------------------------------------------
    # CONGESTION CONFIG
    # -------------------------------------------------

    congestion_config = config.get(
        "congestion",
        {}
    )

    congestion_enabled = (
        congestion_config.get(
            "enabled",
            True
        )
    )

    congestion_density_threshold = (
        congestion_config.get(
            "density_threshold",
            0.70
        )
    )

    congestion_speed_threshold = (
        congestion_config.get(
            "speed_threshold",
            15
        )
    )

    congestion_minimum_duration = (
        congestion_config.get(
            "minimum_duration_seconds",
            30
        )
    )

    # -------------------------------------------------
    # BOTTLENECK CONFIG
    # -------------------------------------------------

    bottleneck_config = config.get(
        "bottleneck",
        {}
    )

    bottleneck_enabled = (
        bottleneck_config.get(
            "enabled",
            True
        )
    )

    bottleneck_density_threshold = (
        bottleneck_config.get(
            "density_threshold",
            0.60
        )
    )

    bottleneck_speed_threshold = (
        bottleneck_config.get(
            "speed_threshold",
            15
        )
    )

    bottleneck_minimum_duration = (
        bottleneck_config.get(
            "minimum_duration_seconds",
            10
        )
    )

    bottleneck_rows = (
        bottleneck_config.get(
            "rows",
            3
        )
    )

    bottleneck_columns = (
        bottleneck_config.get(
            "columns",
            3
        )
    )

    bottleneck_max_vehicles = (
        bottleneck_config.get(
            "max_vehicles_per_zone",
            8
        )
    )

    # -------------------------------------------------
    # PRINT CONFIG
    # -------------------------------------------------

    print()

    print("CONFIGURATION")

    print("-" * 60)

    print(
        f"Camera       : {camera_source}"
    )

    print(
        f"Resolution   : "
        f"{camera_width}x{camera_height}"
    )

    print(
        f"Model        : {model_path}"
    )

    print(
        f"Confidence   : "
        f"{confidence_threshold}"
    )

    print(
        f"Image Size   : {image_size}"
    )

    print(
        f"Frame Skip   : {frame_skip}"
    )

    print(
        f"Congestion   : "
        f"{congestion_enabled}"
    )

    print(
        f"Bottleneck   : "
        f"{bottleneck_enabled}"
    )

    print()

    # -------------------------------------------------
    # DETECTOR
    # -------------------------------------------------

    print(
        "Loading YOLO model..."
    )

    detector = VehicleDetector(
        model_path=model_path,
        confidence_threshold=(
            confidence_threshold
        ),
        image_size=image_size
    )

    print(
        "YOLO model loaded."
    )

    # -------------------------------------------------
    # ANALYZERS
    # -------------------------------------------------

    density_analyzer = TrafficDensity(
        maximum_vehicles=20
    )

    congestion_analyzer = (
        CongestionAnalyzer(
            density_threshold=(
                congestion_density_threshold
            ),
            speed_threshold=(
                congestion_speed_threshold
            ),
            minimum_duration_seconds=(
                congestion_minimum_duration
            )
        )
    )

    speed_estimator = SpeedEstimator(
        smoothing_factor=0.7
    )

    bottleneck_analyzer = (
        BottleneckAnalyzer(
            density_threshold=(
                bottleneck_density_threshold
            ),
            speed_threshold=(
                bottleneck_speed_threshold
            ),
            minimum_duration_seconds=(
                bottleneck_minimum_duration
            ),
            rows=bottleneck_rows,
            columns=bottleneck_columns,
            max_vehicles_per_zone=(
                bottleneck_max_vehicles
            )
        )
    )

    # -------------------------------------------------
    # PEAK TRAFFIC ANALYZER
    # -------------------------------------------------

    peak_analyzer = PeakTrafficAnalyzer()

    # -------------------------------------------------
    # CAMERA
    # -------------------------------------------------

    print(
        "Opening laptop camera..."
    )

    camera = cv2.VideoCapture(
        camera_source
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        camera_width
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        camera_height
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    print(
        "Camera opened successfully."
    )

    print()

    print(
        "Traffic analytics started."
    )

    print(
        "Press Q to exit."
    )

    print()

    # -------------------------------------------------
    # FPS
    # -------------------------------------------------

    previous_display_time = time.time()

    display_fps = 0.0

    inference_fps = 0.0

    frame_counter = 0

    last_detections = []

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read frame."
            )

            break

        frame_counter += 1

        # -------------------------------------------------
        # AI INFERENCE
        # -------------------------------------------------

        should_infer = (
            frame_counter
            % (frame_skip + 1)
            == 0
        )

        if should_infer:

            inference_start = time.time()

            detections = detector.track(
                frame
            )

            inference_end = time.time()

            inference_time = (
                inference_end
                - inference_start
            )

            # AI FPS
            if inference_time > 0:

                current_ai_fps = (
                    1.0
                    / inference_time
                )

                if inference_fps == 0:

                    inference_fps = (
                        current_ai_fps
                    )

                else:

                    inference_fps = (
                        0.8 * inference_fps
                        +
                        0.2 * current_ai_fps
                    )

            # Speed
            detections = (
                speed_estimator.update(
                    detections
                )
            )

            last_detections = detections

        # -------------------------------------------------
        # CURRENT VEHICLES
        # -------------------------------------------------

        total_vehicles = 0

        for detection in last_detections:

            if detection["class_name"] in {
                "car",
                "motorcycle",
                "bus",
                "truck"
            }:

                total_vehicles += 1

        # -------------------------------------------------
        # PEAK TRAFFIC
        # -------------------------------------------------

        peak_analyzer.update(
            total_vehicles
        )

        peak_result = (
            peak_analyzer.get_result()
        )

        # -------------------------------------------------
        # DENSITY
        # -------------------------------------------------

        density_percentage = (
            density_analyzer.percentage(
                total_vehicles
            )
        )

        traffic_level = (
            density_analyzer.traffic_level(
                total_vehicles
            )
        )

        # -------------------------------------------------
        # SPEED
        # -------------------------------------------------

        average_speed = (
            speed_estimator.average_speed(
                last_detections
            )
        )

        moving_count = (
            speed_estimator.moving_vehicle_count(
                last_detections,
                minimum_speed=5.0
            )
        )

        slow_count = (
            speed_estimator.slow_vehicle_count(
                last_detections,
                maximum_speed=5.0
            )
        )

        # -------------------------------------------------
        # CONGESTION
        # -------------------------------------------------

        if congestion_enabled:

            congestion_result = (
                congestion_analyzer.update(
                    density_percentage,
                    average_speed
                )
            )

        else:

            congestion_result = {
                "status": "DISABLED",
                "is_congested": False,
                "duration_seconds": 0.0
            }

        # -------------------------------------------------
        # BOTTLENECK
        # -------------------------------------------------

        if bottleneck_enabled:

            bottleneck_result = (
                bottleneck_analyzer.analyze(
                    last_detections,
                    frame.shape[1],
                    frame.shape[0]
                )
            )

        else:

            bottleneck_result = {
                "status": "DISABLED",
                "strongest_zone": None
            }

        # -------------------------------------------------
        # DISPLAY FPS
        # -------------------------------------------------

        current_time = time.time()

        elapsed = (
            current_time
            - previous_display_time
        )

        if elapsed > 0:

            current_display_fps = (
                1.0 / elapsed
            )

            if display_fps == 0:

                display_fps = (
                    current_display_fps
                )

            else:

                display_fps = (
                    0.8 * display_fps
                    +
                    0.2 * current_display_fps
                )

        previous_display_time = (
            current_time
        )

        # -------------------------------------------------
        # DRAW
        # -------------------------------------------------

        frame = draw_detections(
            frame,
            last_detections
        )

        frame = draw_statistics(
            frame,
            last_detections,
            density_analyzer,
            congestion_result,
            bottleneck_result,
            peak_result,
            average_speed,
            moving_count,
            slow_count,
            display_fps,
            inference_fps
        )

        # -------------------------------------------------
        # SHOW
        # -------------------------------------------------

        cv2.imshow(
            "AI Urban Intelligence - Traffic Analytics",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break

    # -------------------------------------------------
    # CLEANUP
    # -------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()

    print()

    print("=" * 60)

    print(
        "TRAFFIC ANALYTICS STOPPED"
    )

    print(
        f"Peak Vehicles: "
        f"{peak_analyzer.get_peak_vehicles()}"
    )

    print(
        f"Peak Time: "
        f"{peak_analyzer.get_peak_time()}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()