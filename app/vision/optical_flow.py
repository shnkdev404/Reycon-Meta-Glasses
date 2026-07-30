"""
Phase 3: Dense Optical Flow Motion Engine.

Uses OpenCV Farneback dense optical flow to compute pixel-level motion vectors (dx, dy)
and motion magnitudes between consecutive camera frames. Detects fast-approaching dynamic
hazards before bounding box displacement tracking stabilizes.
"""
import logging
import math
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger("OpticalFlow")


class OpticalFlowEngine:
    """
    Dense Farneback Optical Flow motion calculation engine.
    Measures frame-to-frame pixel displacements and evaluates region motion vectors.
    """

    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15, iterations: int = 3):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.gray_prev: Optional[np.ndarray] = None

    def compute_flow(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Computes dense Farneback optical flow for input frame against previous frame.
        Returns: (flow_vectors, motion_magnitude, mean_frame_motion)
        flow_vectors shape: (height, width, 2) [dx, dy]
        motion_magnitude shape: (height, width)
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None, None, 0.0

        try:
            if len(frame.shape) == 3:
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_curr = frame

            if self.gray_prev is None or self.gray_prev.shape != gray_curr.shape:
                self.gray_prev = gray_curr
                h, w = gray_curr.shape[:2]
                empty_flow = np.zeros((h, w, 2), dtype=np.float32)
                empty_mag = np.zeros((h, w), dtype=np.float32)
                return empty_flow, empty_mag, 0.0

            # Farneback dense optical flow calculation
            flow = cv2.calcOpticalFlowFarneback(
                self.gray_prev,
                gray_curr,
                None,
                pyr_scale=self.pyr_scale,
                levels=self.levels,
                winsize=self.winsize,
                iterations=self.iterations,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mean_motion = float(np.mean(magnitude))

            self.gray_prev = gray_curr
            return flow, magnitude, round(mean_motion, 3)
        except Exception as e:
            logger.error(f"Optical flow computation error: {e}")
            return None, None, 0.0

    @staticmethod
    def compute_region_motion(bbox: list, flow: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates mean motion magnitude, mean vx, and mean vy inside a bounding box region.
        Returns: (mean_magnitude, mean_vx, mean_vy)
        """
        if flow is None or not hasattr(flow, "shape") or not bbox:
            return 0.0, 0.0, 0.0

        try:
            h, w = flow.shape[:2]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                return 0.0, 0.0, 0.0

            roi_flow = flow[y1:y2, x1:x2]
            vx = float(np.mean(roi_flow[..., 0]))
            vy = float(np.mean(roi_flow[..., 1]))
            roi_mags = np.sqrt(roi_flow[..., 0] ** 2 + roi_flow[..., 1] ** 2)
            mag = float(np.mean(roi_mags))
            return round(mag, 2), round(vx, 2), round(vy, 2)
        except Exception as e:
            logger.error(f"Region motion error: {e}")
            return 0.0, 0.0, 0.0

    def detect_fast_moving_threats(
        self,
        frame: np.ndarray,
        bboxes: List[list],
        motion_threshold: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Identifies bounding box regions with high optical flow motion magnitudes exceeding motion_threshold.
        Returns list of dynamic threat alerts:
        [
            {
                "bbox_index": int,
                "motion_magnitude": float,
                "vector": (vx, vy),
                "is_fast_threat": True
            }
        ]
        """
        flow, magnitude, mean_motion = self.compute_flow(frame)
        if flow is None:
            return []

        fast_threats = []
        for idx, bbox in enumerate(bboxes):
            mag, vx, vy = self.compute_region_motion(bbox, flow)
            if mag >= motion_threshold:
                fast_threats.append({
                    "bbox_index": idx,
                    "bbox": bbox,
                    "motion_magnitude": mag,
                    "vector": (vx, vy),
                    "is_fast_threat": True
                })

        return fast_threats
