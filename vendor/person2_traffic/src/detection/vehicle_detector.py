from ultralytics import YOLO


class VehicleDetector:
    """
    Vehicle detector using YOLO + BoT-SORT.

    Vehicle classes from COCO:
        2 = car
        3 = motorcycle
        5 = bus
        7 = truck
    """

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path="yolo11s.pt",
        confidence_threshold=0.15,
        image_size=640
    ):

        print(
            f"Loading vehicle model: {model_path}"
        )

        self.model = YOLO(model_path)

        self.confidence_threshold = (
            confidence_threshold
        )

        self.image_size = image_size

        print("Vehicle model loaded.")

    def track(self, frame):

        results = self.model.track(

            source=frame,

            conf=self.confidence_threshold,

            imgsz=self.image_size,

            # Only vehicle classes
            classes=[
                2, 3, 5, 7
            ],

            # Normal NMS
            iou=0.50,

            tracker="botsort.yaml",

            persist=True,

            verbose=False
        )

        detections = []

        if not results:

            return detections

        result = results[0]

        if result.boxes is None:

            return detections

        boxes = result.boxes

        for box in boxes:

            class_id = int(
                box.cls[0].item()
            )

            if class_id not in self.VEHICLE_CLASSES:

                continue

            confidence = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            track_id = None

            if box.id is not None:

                track_id = int(
                    box.id[0].item()
                )

            class_name = (
                self.VEHICLE_CLASSES[
                    class_id
                ]
            )

            # -------------------------------------------------
            # VEHICLE SIZE
            # -------------------------------------------------

            width = x2 - x1
            height = y2 - y1

            area = width * height

            # -------------------------------------------------
            # CENTER
            # -------------------------------------------------

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            detections.append({

                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ],

                "class_id": class_id,

                "class_name": class_name,

                "confidence": confidence,

                "track_id": track_id,

                "width": width,

                "height": height,

                "area": area,

                "center": (
                    int(center_x),
                    int(center_y)
                )
            })

        return detections