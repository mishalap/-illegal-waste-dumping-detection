import math
import numpy as np
from ultralytics import YOLO
from config import (
    DEVICE, DETECTION_MODEL,
    DETECTION_CONFIDENCE,
    PERSON_CLASS_ID, WASTE_CLASS_IDS, CAR_CLASS_IDS
)


class Detector:
    """Wraps a YOLOv8 object detection model for person + waste detection."""

    def __init__(self):
        print(f"[Detector] Loading model on device: {DEVICE}")
        self.detection_model = YOLO(DETECTION_MODEL)
        print(f"[Detector] Detection model: {DETECTION_MODEL}")

    @staticmethod
    def _safe_track_id(boxes, i):
        """Safely extract track ID, guarding against None tensor and NaN values."""
        if boxes.id is None:
            return -1
        val = boxes.id[i].item()
        if math.isnan(val):
            return -1
        return int(val)

    def detect(self, frame):
        """
        Run object detection with tracking on a frame.

        Returns:
            dict with keys:
                - "persons": list of dicts {box, confidence, track_id}
                - "waste": list of dicts {box, confidence, class_id, class_name, track_id}
                - "raw_results": ultralytics Results object
        """
        results = self.detection_model.track(
            frame,
            conf=DETECTION_CONFIDENCE,
            device=DEVICE,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        persons = []
        waste_objects = []
        cars = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    box = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    track_id = self._safe_track_id(boxes, i)

                    entry = {
                        "box": box,  # [x1, y1, x2, y2]
                        "confidence": conf,
                        "track_id": track_id,
                        "class_id": cls_id,
                        "class_name": self.detection_model.names[cls_id]
                    }

                    if cls_id == PERSON_CLASS_ID:
                        persons.append(entry)
                    elif cls_id in WASTE_CLASS_IDS:
                        waste_objects.append(entry)
                    elif cls_id in CAR_CLASS_IDS:
                        cars.append(entry)

        return {
            "persons": persons,
            "waste": waste_objects,
            "cars": cars,
            "raw_results": results
        }

    def detect_for_video(self, frame):
        """
        Run detection without tracking (for uploaded video analysis).
        Tracking state should not persist across non-sequential frames.

        Returns same format as detect() but without track_ids.
        """
        results = self.detection_model(
            frame,
            conf=DETECTION_CONFIDENCE,
            device=DEVICE,
            verbose=False
        )

        persons = []
        waste_objects = []
        cars = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    box = boxes.xyxy[i].cpu().numpy().astype(int).tolist()

                    entry = {
                        "box": box,
                        "confidence": conf,
                        "track_id": -1,
                        "class_id": cls_id,
                        "class_name": self.detection_model.names[cls_id]
                    }

                    if cls_id == PERSON_CLASS_ID:
                        persons.append(entry)
                    elif cls_id in WASTE_CLASS_IDS:
                        waste_objects.append(entry)
                    elif cls_id in CAR_CLASS_IDS:
                        cars.append(entry)

        return {
            "persons": persons,
            "waste": waste_objects,
            "cars": cars,
            "raw_results": results
        }
