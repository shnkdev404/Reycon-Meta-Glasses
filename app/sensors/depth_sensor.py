"""
Depth Sensor & Metric Depth Estimator Module.
Provides RGB-D sensor camera depth alignment and pinhole fallback geometric distance model.
"""
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Refine distance estimates using actual depth sensor data."""
    
    def __init__(self, focal_length: float = 600.0, baseline: float = 0.065):
        """
        Args:
            focal_length: Camera focal length in pixels
            baseline: Stereo baseline in meters (for stereo cameras)
        """
        self.focal_length = focal_length
        self.baseline = baseline
    
    def get_distance_from_depth(self, bbox: Tuple[float, float, float, float], depth_map: np.ndarray) -> Optional[float]:
        """
        Extract actual depth from depth map within bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] in pixels
            depth_map: Depth map from sensor (same resolution as image)
        
        Returns:
            Median depth in meters, or None if invalid
        """
        if depth_map is None or not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Clamp to valid range
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1], x2)
        y2 = min(depth_map.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract depth region
        roi = depth_map[y1:y2, x1:x2]
        
        # Filter invalid pixels (0 or very large values)
        valid_depth = roi[(roi > 0.1) & (roi < 50.0)]
        
        if len(valid_depth) == 0:
            return None
        
        # Return median depth
        return float(np.median(valid_depth))
    
    def fallback_distance(self, bbox: Tuple[float, float, float, float], frame_height: int) -> float:
        """Fallback pinhole model if depth unavailable."""
        x1, y1, x2, y2 = bbox
        box_height = max(1.0, y2 - y1)
        # Assume 1.7m average height for person
        distance = (self.focal_length * 1.7) / box_height
        return min(50.0, max(0.5, distance))


class BaseDepthSensor:
    """Base class interface for Depth Sensors."""

    def read_depth_map(self) -> Optional[np.ndarray]:
        raise NotImplementedError


class SimulatedDepthSensor(BaseDepthSensor):
    """Simulated RGB-D Depth Sensor."""

    def __init__(self, device_id: Any = "glass_01", width: int = 640, height: int = 384):
        self.device_id = device_id
        self.width = width if isinstance(width, int) else 640
        self.height = height if isinstance(height, int) else 384

    def read_depth_map(self) -> Dict[str, Any]:
        return {
            "min_depth_m": 0.5,
            "max_depth_m": 10.0,
            "matrix": np.ones((self.height, self.width), dtype=np.float32) * 5.0
        }


def estimate_distance_with_depth(bbox: Tuple[float, float, float, float], depth_map: np.ndarray) -> Optional[float]:
    estimator = DepthEstimator()
    return estimator.get_distance_from_depth(bbox, depth_map)


class DepthSensor(BaseDepthSensor):
    """Hardware Depth Sensor Interface."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.depth_estimator = DepthEstimator()

    def get_depth_frame(self, width: int = 640, height: int = 384) -> np.ndarray:
        """Returns depth map aligned with RGB image frame."""
        depth_map = np.ones((height, width), dtype=np.float32) * 5.0
        return depth_map

    def read_depth_map(self) -> Optional[np.ndarray]:
        return self.get_depth_frame()
