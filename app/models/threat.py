"""
Pydantic data models for Threat Assessment and Directed Alerts.
"""
from enum import Enum
from datetime import datetime, timezone
from typing import Dict
from pydantic import BaseModel, Field


class ThreatLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ThreatType(str, Enum):
    VEHICLE_APPROACH = "VEHICLE_APPROACH"
    FORKLIFT_APPROACH = "FORKLIFT_APPROACH"
    PERSON_RUNNING = "PERSON_RUNNING"
    FALLING_OBJECT = "FALLING_OBJECT"
    COLLISION_RISK = "COLLISION_RISK"
    BLIND_SPOT_OBSTACLE = "BLIND_SPOT_OBSTACLE"
    BEHAVIOR_ANOMALY = "BEHAVIOR_ANOMALY"
    PERSON_FROZEN = "PERSON_FROZEN"


class ThreatAlert(BaseModel):
    """Targeted spatial warning dispatched exclusively to an affected smart glass unit."""
    alert_id: str = Field(..., description="Unique alert identifier")
    target_glass_id: str = Field(..., description="Target glass ID that MUST receive warning")
    trigger_object_id: str = Field(..., description="Global object ID causing the threat")
    threat_type: ThreatType
    threat_level: ThreatLevel
    time_to_collision: float = Field(..., description="Estimated TTC in seconds")
    distance: float = Field(..., description="Current distance to threat in meters")
    bearing: float = Field(..., description="Relative direction angle to threat in degrees")
    warning_message: str = Field(..., description="Human-readable warning text")
    threat_score: float = Field(default=0.0, description="Multi-factor weighted threat score (0.0 to 1.0)")
    score_components: Dict[str, float] = Field(default_factory=dict, description="Multi-factor risk score component breakdown")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
