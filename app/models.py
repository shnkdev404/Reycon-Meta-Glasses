from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class Position(BaseModel):
    """2D/3D spatial coordinates of the smart glasses."""
    x: float = Field(..., description="X coordinate (meters)")
    y: float = Field(..., description="Y coordinate (meters)")
    z: Optional[float] = Field(default=0.0, description="Z / height coordinate (meters)")


class Detection(BaseModel):
    """Visual perception detection object from smart glasses AI camera."""
    model_config = ConfigDict(populate_by_name=True)

    label: str = Field(..., alias="class", description="Detected object class (e.g. vehicle, person)")
    distance: float = Field(..., description="Estimated distance in meters")
    bearing: float = Field(..., description="Relative bearing angle in degrees (-180 to 180)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence score (0 to 1)")


class GlassState(BaseModel):
    """Complete perception state packet sent by a single smart glass unit."""
    glass_id: str = Field(..., min_length=1, description="Unique device identifier (e.g. glass_A)")
    position: Position = Field(..., description="Current spatial position")
    heading: float = Field(..., description="Compass orientation angle in degrees (0 to 360)")
    detections: List[Detection] = Field(default_factory=list, description="List of active detections")
