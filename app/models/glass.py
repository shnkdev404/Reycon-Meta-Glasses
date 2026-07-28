"""
Pydantic data models representing Meta Smart Glass pose, sensors, health, and state.
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class GlassPose(BaseModel):
    """3D spatial position & orientation of the Meta Smart Glass."""
    x: float = Field(..., description="Global X position (meters)")
    y: float = Field(..., description="Global Y position (meters)")
    z: float = Field(default=0.0, description="Global Z position (meters)")
    heading: float = Field(..., description="Compass orientation (0-360 degrees)")
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
    pose: GlassPose
    sensors: Optional[GlassSensors] = Field(default_factory=GlassSensors)
    health: Optional[GlassHealth] = Field(default_factory=GlassHealth)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    last_update: datetime = Field(default_factory=datetime.utcnow)
