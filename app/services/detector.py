import logging
from typing import List
import cv2
import numpy as np
from ultralytics import YOLO
from app.models import Detection, Position

logger = logging.getLogger("DetectionEngine")


class DetectionEngine:
    """YOLO11 Object Detection Engine for processing video frames and parsing spatial directions & bounding boxes."""

    def __init__(self, model_name: str = "yolo11n.pt"):
        logger.info(f"Initializing YOLO model '{model_name}'...")
        self.model = YOLO(model_name)
        logger.info("YOLO model loaded successfully.")

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
        Returns a list of Detection Pydantic models with calculated directions and bounding box reticle coordinates.
        """
        detections: List[Detection] = []
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
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

            # Filter low confidence detections (< 35%)
            if confidence < 0.35:
                continue

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            direction = self.get_direction(center_x, frame_width)

            detection = Detection(
                class_name=class_name,
                confidence=round(confidence, 4),
                position=Position(
                    x=round(center_x, 2),
                    y=round(center_y, 2),
                    z=0.0
                ),
                direction=direction,
                bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            )
            detections.append(detection)

        return detections


detector = DetectionEngine()
