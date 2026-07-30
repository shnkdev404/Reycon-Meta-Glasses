"""
Optical Flow Velocity Estimator Module.
Estimates object 2D motion velocity vectors (vx, vy, magnitude) using Farneback optical flow.
"""
import logging
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpticalFlowVelocity:
    """Estimate object velocity from optical flow."""
    
    def __init__(self, method: str = 'farneback'):
        self.method = method
        self.prev_gray: Optional[np.ndarray] = None
        self.flow: Optional[np.ndarray] = None
    
    def compute_flow(self, frame: np.ndarray) -> np.ndarray:
        """Compute optical flow between current and previous frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return np.zeros((384, 640, 2), dtype=np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
        
        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        else:
            flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
        
        self.prev_gray = gray
        self.flow = flow
        return flow
    
    def get_roi_velocity(self, bbox: Tuple[float, float, float, float], flow: np.ndarray) -> Tuple[float, float, float]:
        """
        Get average velocity magnitude and direction within bbox region.
        
        Returns:
            (velocity_x, velocity_y, magnitude)
        """
        if flow is None or not isinstance(flow, np.ndarray) or flow.size == 0:
            return 0.0, 0.0, 0.0

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(flow.shape[1], x2)
        y2 = min(flow.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, 0.0
        
        roi_flow = flow[y1:y2, x1:x2]
        
        avg_flow_x = np.mean(roi_flow[..., 0])
        avg_flow_y = np.mean(roi_flow[..., 1])
        
        magnitude = np.sqrt(avg_flow_x ** 2 + avg_flow_y ** 2)
        
        return float(avg_flow_x), float(avg_flow_y), float(magnitude)
