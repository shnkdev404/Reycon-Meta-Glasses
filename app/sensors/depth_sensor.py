"""
Phase 2: Sensor Interface Layer - Depth Sensor Interface.

Independent contract and simulated depth sensor interface providing 3D depth maps
or metric distance readings.
"""
from abc import ABC, abstractmethod
from typing import Optional, Any


class BaseDepthSensor(ABC):
    """Abstract contract for Depth sensors (Time-of-Flight / Stereo / Neural Depth)."""

    @abstractmethod
    def read_depth_map(self) -> Optional[Any]:
        """Fetch latest raw metric depth map buffer."""
        pass

    @abstractmethod
    def get_distance_at_pixel(self, x: int, y: int) -> float:
        """Fetch metric distance (meters) at specific pixel coordinate."""
        pass


class SimulatedDepthSensor(BaseDepthSensor):
    """Synthetic Depth sensor generating simulated metric distances for testing."""

    def __init__(self, glass_id: str):
        self.glass_id = glass_id

    def read_depth_map(self) -> Optional[dict]:
        return {
            "glass_id": self.glass_id,
            "min_depth_m": 0.5,
            "max_depth_m": 20.0,
            "unit": "meters"
        }

    def get_distance_at_pixel(self, x: int, y: int) -> float:
        return 5.0  # Default simulated metric depth distance (meters)
