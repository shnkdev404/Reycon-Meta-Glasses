"""
Multi-Factor Threat Assessment Engine.
Calculates 6-factor threat scores (confidence, proximity, velocity, size, pose, anomaly)
and classifies threat levels (LOW, MEDIUM, HIGH, CRITICAL).
"""
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThreatAssessment:
    """Multi-factor threat score."""
    base_score: float        # 0-1, from model confidence
    proximity_score: float   # Weight by distance (closer = more threat)
    velocity_score: float    # Weight by movement speed toward camera
    size_score: float        # Weight by object size relative to frame
    pose_score: float        # Weight by body pose (standing > sitting)
    anomaly_score: float     # Unusual behavior detected
    
    @property
    def total_score(self) -> float:
        """Weighted combination of all factors."""
        return (
            0.30 * self.base_score +       # Confidence
            0.25 * self.proximity_score +  # Proximity (1/distance)
            0.20 * self.velocity_score +   # Speed toward camera
            0.15 * self.size_score +       # Relative size
            0.05 * self.pose_score +       # Body pose
            0.05 * self.anomaly_score      # Anomalous behavior
        )
    
    @property
    def threat_level(self) -> str:
        """Classify threat level."""
        score = self.total_score
        if score < 0.3:
            return "LOW"
        elif score < 0.6:
            return "MEDIUM"
        elif score < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"


class ThreatScorer:
    """Compute threat scores for detected objects."""
    
    def __init__(self):
        self.distance_weights = {
            "person": {"close": (0, 2), "medium": (2, 8), "far": (8, 30)},
            "vehicle": {"close": (0, 5), "medium": (5, 20), "far": (20, 50)},
        }
    
    def compute_threat(
        self,
        class_name: str,
        confidence: float,
        distance: float,
        velocity: float,
        bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int],
        pose_data: Optional[Dict[str, Any]] = None
    ) -> ThreatAssessment:
        """
        Compute comprehensive threat score.
        """
        # Base confidence score
        base_score = min(1.0, confidence)
        
        # Proximity score: closer = higher threat
        proximity_score = 1.0 / (1.0 + distance / 2.0)
        proximity_score = min(1.0, proximity_score)
        
        # Velocity score: faster toward camera = higher threat
        velocity_score = min(1.0, velocity / 10.0)
        
        # Size score: larger object = more threatening (vehicle vs toy)
        x1, y1, x2, y2 = bbox
        bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
        frame_area = max(1.0, float(frame_shape[0] * frame_shape[1]))
        size_ratio = bbox_area / frame_area
        size_score = min(1.0, size_ratio * 50)
        
        # Pose score: standing person > sitting person
        pose_score = 0.5
        if pose_data and "action" in pose_data:
            action_threats = {"running": 0.9, "attacking": 1.0, "standing": 0.5, "sitting": 0.2}
            pose_score = action_threats.get(pose_data["action"], 0.5)
        
        anomaly_score = 0.0
        
        return ThreatAssessment(
            base_score=base_score,
            proximity_score=proximity_score,
            velocity_score=velocity_score,
            size_score=size_score,
            pose_score=pose_score,
            anomaly_score=anomaly_score
        )
