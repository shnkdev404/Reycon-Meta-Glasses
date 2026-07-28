import time
from typing import List, Optional
from pydantic import BaseModel, Field


class Position(BaseModel):
    """3D spatial position coordinates (relative meters or pixels)."""
    x: float = Field(default=0.0, description="X spatial coordinate")
    y: float = Field(default=0.0, description="Y spatial coordinate")
    z: float = Field(default=0.0, description="Z spatial coordinate / elevation")


class GPSLocation(BaseModel):
    """Exact GPS geographic location telemetry from mobile hardware sensors."""
    latitude: float = Field(..., description="Geographic latitude in degrees (-90 to 90)")
    longitude: float = Field(..., description="Geographic longitude in degrees (-180 to 180)")
    altitude: Optional[float] = Field(default=0.0, description="Altitude above sea level in meters")
    accuracy: Optional[float] = Field(default=0.0, description="Horizontal GPS accuracy radius in meters")


class Detection(BaseModel):
    """Computer vision detected object details with bounding box reticle coordinates."""
    class_name: str = Field(..., description="Name of detected object class (e.g., person, truck)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score (0.0 to 1.0)")
    position: Position = Field(default_factory=Position, description="Spatial position of detected object")
    direction: str = Field(default="FRONT", description="Horizontal position relative to camera view (LEFT, FRONT, RIGHT)")
    bbox: List[float] = Field(default_factory=list, description="Bounding box pixel coordinates [x1, y1, x2, y2]")


class GlassState(BaseModel):
    """Current state of a connected device (smartphone or smart glass)."""
    glass_id: str = Field(..., description="Unique hardware identifier for client device")
    position: Position = Field(default_factory=Position, description="Current relative position")
    gps: Optional[GPSLocation] = Field(default=None, description="Exact GPS geographic telemetry")
    heading: float = Field(default=0.0, description="Compass orientation heading in degrees (0 to 360)")
    detections: List[Detection] = Field(default_factory=list, description="Active detections from current frame")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of state update in epoch seconds")
