"""
Configuration settings for the Shared Perception Platform.
"""
import os
from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = "Shared Perception Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Networking & Connection
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    HEARTBEAT_INTERVAL_SEC: float = 5.0
    AUTH_SECRET_KEY: str = "kaya_hackathon_meta_secret_key"

    # Perception & Spatial Fusion
    FUSION_DISTANCE_THRESHOLD: float = 2.0  # Objects within 2 meters can be fused
    IOU_MATCH_THRESHOLD: float = 0.4        # 2D Bounding Box IoU threshold for fusion

    # Threat Engine
    TTC_WARNING_THRESHOLD_SEC: float = 4.0  # Warn if collision predicted within 4 seconds
    DANGER_RADIUS_METERS: float = 10.0      # Proximity alert distance threshold


settings = Settings()
