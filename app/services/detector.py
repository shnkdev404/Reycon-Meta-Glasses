import logging
import math
from typing import List
import cv2
import numpy as np
from ultralytics import YOLO
from app.models import Detection, Position

logger = logging.getLogger("DetectionEngine")

# Typical real-world physical heights in meters for object categories
OBJECT_REAL_HEIGHTS = {
    "person": 1.7,
    "bicycle": 1.1,
    "car": 1.5,
    "motorcycle": 1.2,
    "bus": 3.2,
    "truck": 3.0,
    "traffic light": 2.5,
    "fire hydrant": 0.8,
    "stop sign": 2.2,
    "bench": 0.8,
    "dog": 0.6,
    "cat": 0.3,
    "backpack": 0.5,
    "umbrella": 0.9,
    "handbag": 0.4,
    "suitcase": 0.6,
    "bottle": 0.25,
    "cup": 0.15,
    "fork": 0.2,
    "knife": 0.2,
    "spoon": 0.15,
    "bowl": 0.15,
    "banana": 0.18,
    "apple": 0.1,
    "sandwich": 0.1,
    "chair": 0.9,
    "couch": 0.9,
    "potted plant": 0.6,
    "bed": 0.8,
    "dining table": 0.75,
    "toilet": 0.7,
    "tv": 0.5,
    "laptop": 0.3,
    "mouse": 0.05,
    "remote": 0.15,
    "keyboard": 0.1,
    "cell phone": 0.15,
    "microwave": 0.35,
    "oven": 0.8,
    "toaster": 0.25,
    "sink": 0.8,
    "refrigerator": 1.8,
    "book": 0.25,
    "clock": 0.3,
    "vase": 0.3,
    "scissors": 0.2,
    "teddy bear": 0.3,
    "forklift": 2.5
}


def estimate_object_distance(class_name: str, box_height_px: float, frame_height_px: float) -> float:
    """
    Estimate physical distance to detected object using pinhole camera model.
    D = (f_y * H_real) / h_box
    f_y is estimated as 1.1 * frame_height (approx. 50 deg vertical FOV).
    """
    clean_name = class_name.lower().split(' #')[0]
    real_h = OBJECT_REAL_HEIGHTS.get(clean_name, 1.0)
    box_h = max(2.0, float(box_height_px))
    f_y = 1.1 * float(frame_height_px)
    dist = (f_y * real_h) / box_h
    return max(0.5, min(40.0, round(dist, 2)))


class DetectionEngine:
    """YOLO11 Object Detection Engine for processing video frames and parsing spatial directions & bounding boxes."""

    def __init__(self, model_name: str = "yolo11n.pt"):
        logger.info(f"Initializing YOLO model '{model_name}'...")
        try:
            self.model = YOLO(model_name)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{model_name}': {e}. Using fallback mode.")
            self.model = None

    def get_direction(self, x_center: float, frame_width: float) -> str:
        """Calculate relative horizontal direction zone (LEFT, FRONT, RIGHT)."""
        third = frame_width / 3.0
        if x_center < third:
            return "LEFT"
        elif x_center > 2.0 * third:
            return "RIGHT"
        return "FRONT"

    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO11 object detection on an OpenCV frame (BGR NumPy array).
        Returns a list of Detection Pydantic models with calculated directions, estimated 3D position, and bounding boxes.
        """
        detections: List[Detection] = []
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or self.model is None:
            return detections

        frame_height, frame_width = frame.shape[:2]

        # Run inference in quiet mode
        results = self.model(frame, verbose=False)

        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            class_name = self.model.names.get(class_id, f"class_{class_id}")

            # Filter low confidence detections (< 0.25)
            if confidence < 0.25:
                continue

            center_x = (x1 + x2) / 2.0
            direction = self.get_direction(center_x, frame_width)

            # Estimate relative distance based on bounding box height
            box_height = max(1.0, y2 - y1)
            estimated_distance = estimate_object_distance(class_name, box_height, frame_height)

            # Normalized horizontal offset (-1.0 to 1.0)
            norm_x = (center_x - frame_width / 2.0) / (frame_width / 2.0)
            bearing_rad = norm_x * math.radians(30.0)
            rel_x = round(estimated_distance * math.sin(bearing_rad), 2)
            rel_y = round(estimated_distance * math.cos(bearing_rad), 2)

            detection = Detection(
                class_name=class_name,
                label=class_name,
                confidence=round(confidence, 4),
                position=Position(
                    x=rel_x,
                    y=rel_y,
                    z=0.0
                ),
                direction=direction,
                bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                distance=estimated_distance,
                bearing=round(math.degrees(bearing_rad), 1)
            )
            
            detections.append(detection)

        return detections


detector = DetectionEngine()

