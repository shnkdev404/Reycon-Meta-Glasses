"""
Pydantic data models for 2D visual detections and 3D fused World Objects.
"""
from typing import Optional, List, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class BoundingBox2D(BaseModel):
    """2D Bounding Box in image pixel coordinates (x_min, y_min, x_max, y_max)."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class Detection2D(BaseModel):
    """Visual perception detection from camera frame."""
    model_config = ConfigDict(populate_by_name=True)

    label: str = Field(..., alias="class", description="Object category (vehicle, person, forklift, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[BoundingBox2D] = None
    distance: float = Field(..., description="Estimated distance in meters")
    bearing: float = Field(..., description="Relative bearing angle in degrees")


class WorldObject(BaseModel):
    """Fused 3D spatial object in global World coordinates."""
    object_id: str = Field(..., description="Unique global object identifier")
    label: str = Field(..., description="Primary object category")
    confidence: float = Field(..., ge=0.0, le=1.0)
    position_x: float = Field(..., description="World X coordinate (meters)")
    position_y: float = Field(..., description="World Y coordinate (meters)")
    position_z: float = Field(default=0.0, description="World Z coordinate (meters)")
    velocity_x: float = Field(default=0.0, description="Estimated X velocity (m/s)")
    velocity_y: float = Field(default=0.0, description="Estimated Y velocity (m/s)")
    velocity_z: float = Field(default=0.0, description="Estimated Z velocity (m/s)")
    source_glasses: List[str] = Field(default_factory=list, description="Glass IDs observing this object")
    detection_count: int = Field(default=1, description="Number of glasses confirming detection")
    last_seen: datetime = Field(default_factory=datetime.utcnow)
