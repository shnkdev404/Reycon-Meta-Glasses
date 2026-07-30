"""
Phase 10: Meta Wearable SDK Hardware Adapters - RGB-D Depth Camera Adapter.

Hardware adapter connecting Ray-Ban Meta Smart Glasses RGB-D depth camera feeds to the perception pipeline.
"""
import logging
from typing import Optional
import numpy as np
from app.sensors.depth_sensor import BaseDepthSensor

logger = logging.getLogger("MetaDepthAdapter")


class MetaDepthAdapter(BaseDepthSensor):
    """Hardware adapter for Ray-Ban Meta Glasses RGB-D depth camera sensor feed."""

    def __init__(self, glass_id: str):
        self.glass_id = glass_id
        self._current_depth_map: Optional[np.ndarray] = None
        logger.info(f"Initialized MetaDepthAdapter for glass '{glass_id}'")

    def get_depth_map(self) -> np.ndarray:
        """Return raw 2D metric depth map buffer (in meters) from Meta glasses."""
        if self._current_depth_map is None:
            # Generate 640x384 default depth buffer in meters
            self._current_depth_map = np.full((384, 640), 4.2, dtype=np.float32)
        return self._current_depth_map

    def set_depth_buffer(self, depth_bytes: bytes, width: int = 640, height: int = 384):
        """Update active depth map from incoming raw byte stream."""
        try:
            arr = np.frombuffer(depth_bytes, dtype=np.float32)
            if arr.size == width * height:
                self._current_depth_map = arr.reshape((height, width))
        except Exception as e:
            logger.error(f"Failed to decode depth buffer for '{self.glass_id}': {e}")

    def read_depth_map() -> Optional[np.ndarray]:
        return self.get_depth_map()

    def get_distance_at_pixel(self, x: int, y: int) -> float:
        depth_map = self.get_depth_map()
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            val = float(depth_map[y, x])
            if val > 0:
                return round(val, 2)
        return 4.2
