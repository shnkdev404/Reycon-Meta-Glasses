"""
Pydantic data models representing Meta Smart Glass pose, sensors, health, and state.
"""
import time
from typing import Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, Field


class GlassPose(BaseModel):
    """3D spatial position & orientation of the Meta Smart Glass."""
    x: float = Field(default=0.0, description="Global X position (meters)")
    y: float = Field(default=0.0, description="Global Y position (meters)")
    z: float = Field(default=0.0, description="Global Z position (meters)")
    heading: float = Field(default=0.0, description="Compass orientation (0-360 degrees)")
    pitch: float = Field(default=0.0, description="Tilt angle up/down (-90 to 90 degrees)")
    roll: float = Field(default=0.0, description="Roll angle left/right (-180 to 180 degrees)")


class GlassSensors(BaseModel):
    """Raw IMU & environment sensor readings from Meta Wearable SDK."""
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class GlassHealth(BaseModel):
    """Device connection health & battery status."""
    battery_level: float = Field(default=100.0, ge=0.0, le=100.0)
    signal_strength_dbm: float = Field(default=-50.0)
    latency_ms: float = Field(default=10.0)
    is_active: bool = True


class GlassState(BaseModel):
    """Full synchronized state of a single connected smart glass unit."""
    glass_id: str = Field(..., min_length=1, description="Unique glass device identifier")
    pose_obj: Optional[GlassPose] = Field(default=None)
    position: Optional[Any] = None
    gps: Optional[Any] = None
    heading: float = 0.0
    sensors: Optional[GlassSensors] = Field(default_factory=GlassSensors)
    health: Optional[GlassHealth] = Field(default_factory=GlassHealth)
    detections: List[Any] = Field(default_factory=list)
    tracked_objects: List[Any] = Field(default_factory=list)
    local_map: Optional[Any] = None
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    last_update: datetime = Field(default_factory=datetime.utcnow)
    timestamp: float = Field(default_factory=time.time)

    @property
    def pose(self) -> GlassPose:
        if self.pose_obj is not None:
            return self.pose_obj
        pos_x = getattr(self.position, 'x', 0.0) if self.position else 0.0
        pos_y = getattr(self.position, 'y', 0.0) if self.position else 0.0
        pos_z = getattr(self.position, 'z', 0.0) if self.position else 0.0
        return GlassPose(x=pos_x, y=pos_y, z=pos_z, heading=self.heading)

    @pose.setter
    def pose(self, val: GlassPose):
        self.pose_obj = val
