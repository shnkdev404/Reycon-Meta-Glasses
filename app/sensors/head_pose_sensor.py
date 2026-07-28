"""
Phase 2: Sensor Interface Layer - Head Pose Sensor Interface.

Independent contract and simulated 6DoF head pose orientation sensor interface
reading compass heading, pitch, roll, and spatial origin offsets.
"""
from abc import ABC, abstractmethod
from app.models.glass import GlassPose


class BaseHeadPoseSensor(ABC):
    """Abstract contract for Head Pose & Spatial Orientation sensors."""

    @abstractmethod
    def read_head_pose(self) -> GlassPose:
        """Fetch latest 6DoF head pose orientation and position."""
        pass


class SimulatedHeadPoseSensor(BaseHeadPoseSensor):
    """Synthetic Head Pose sensor generating simulated spatial orientation telemetry."""

    def __init__(self, glass_id: str, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        self.glass_id = glass_id
        self.x = x
        self.y = y
        self.heading = heading

    def read_head_pose(self) -> GlassPose:
        return GlassPose(
            x=self.x,
            y=self.y,
            z=1.65,  # Typical user standing eye-level height in meters
            heading=self.heading,
            pitch=0.0,
            roll=0.0
        )
