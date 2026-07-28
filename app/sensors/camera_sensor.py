"""
Phase 2: Sensor Interface Layer - Camera Sensor Interface.

Independent contract and simulated camera stream interface.
Can be replaced by native Meta Wearable SDK camera feed or OpenCV video capture.
"""
from abc import ABC, abstractmethod
from typing import Optional, Any
from datetime import datetime


class BaseCameraSensor(ABC):
    """Abstract contract for Smart Glasses camera video feed."""

    @abstractmethod
    def start() -> bool:
        """Initialize and start video capture."""
        pass

    @abstractmethod
    def read_frame(self) -> Optional[Any]:
        """Fetch the latest raw camera frame buffer."""
        pass

    @abstractmethod
    def stop(self):
        """Stop video capture and release hardware resources."""
        pass


class SimulatedCameraSensor(BaseCameraSensor):
    """Synthetic camera sensor producing simulated frames for hackathon testing."""

    def __init__(self, glass_id: str, resolution: tuple = (1920, 1080), fps: int = 30):
        self.glass_id = glass_id
        self.resolution = resolution
        self.fps = fps
        self._is_active = False

    def start(self) -> bool:
        self._is_active = True
        return True

    def read_frame(self) -> Optional[dict]:
        if not self._is_active:
            return None
        return {
            "glass_id": self.glass_id,
            "timestamp": datetime.utcnow().timestamp(),
            "width": self.resolution[0],
            "height": self.resolution[1],
            "format": "RGB888",
            "frame_bytes": b"\x00" * 1024  # Simulated raw pixel frame buffer
        }

    def stop(self):
        self._is_active = False
