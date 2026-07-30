"""
Phase 13: Multi-Factor Threat Scoring System.

Calculates multi-factor risk weighted threat scores for objects and threats:
  threat_score = (
      0.4 * confidence +           # Detection confidence
      0.3 * (1 / distance) +       # Proximity weight
      0.2 * person_size_ratio +    # Relative size (aggressiveness)
      0.1 * velocity_magnitude     # Speed toward camera/glass
  )
"""
import logging
from typing import Dict, Any, Tuple
from app.models.threat import ThreatLevel

logger = logging.getLogger("ThreatScorer")


class MultiFactorThreatScorer:
    """
    Multi-factor risk weighted threat scoring engine.
    Replaces simplistic binary threat checks with weighted continuous risk modeling.
    """

    def __init__(
        self,
        weight_confidence: float = 0.4,
        weight_proximity: float = 0.3,
        weight_size_ratio: float = 0.2,
        weight_velocity: float = 0.1
    ):
        self.w_conf = weight_confidence
        self.w_prox = weight_proximity
        self.w_size = weight_size_ratio
        self.w_vel = weight_velocity

    def compute_threat_score(
        self,
        confidence: float,
        distance: float,
        person_size_ratio: float = 0.1,
        velocity_magnitude: float = 0.0,
        extra_risk_modifier: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates multi-factor weighted threat score in range [0.0, 1.0] and component breakdowns.
        
        Formula:
          conf_weight = min(1.0, max(0.0, confidence))
          prox_weight = min(1.0, 1.0 / max(0.5, distance))
          size_weight = min(1.0, max(0.0, person_size_ratio))
          vel_weight  = min(1.0, max(0.0, velocity_magnitude / 5.0)) # Normalized up to 5 m/s
        """
        conf_w = min(1.0, max(0.0, float(confidence)))
        
        # Proximity weight: inversely proportional to distance (max weight at <= 0.5m)
        safe_dist = max(0.5, float(distance))
        prox_w = min(1.0, round(1.0 / safe_dist, 4))

        # Size ratio (bbox area vs frame area, default scaled if not provided)
        size_w = min(1.0, max(0.0, float(person_size_ratio)))

        # Speed / Velocity magnitude toward glass (normalized to 5.0 m/s max)
        vel_w = min(1.0, max(0.0, float(velocity_magnitude) / 5.0))

        raw_score = (
            self.w_conf * conf_w +
            self.w_prox * prox_w +
            self.w_size * size_w +
            self.w_vel * vel_w
        )

        # Apply optional extra risk modifier (e.g., trajectory anomaly, blind spot, or weapon detection)
        final_score = min(1.0, max(0.0, raw_score + float(extra_risk_modifier)))
        final_score = round(final_score, 4)

        components = {
            "confidence_component": round(self.w_conf * conf_w, 4),
            "proximity_component": round(self.w_prox * prox_w, 4),
            "size_ratio_component": round(self.w_size * size_w, 4),
            "velocity_component": round(self.w_vel * vel_w, 4),
            "extra_risk_modifier": round(extra_risk_modifier, 4),
            "raw_score": round(raw_score, 4)
        }

        return final_score, components

    def score_to_threat_level(self, threat_score: float) -> ThreatLevel:
        """Maps continuous numerical threat score to 4-tier ThreatLevel enum."""
        if threat_score >= 0.75:
            return ThreatLevel.CRITICAL
        elif threat_score >= 0.50:
            return ThreatLevel.HIGH
        elif threat_score >= 0.25:
            return ThreatLevel.MEDIUM
        return ThreatLevel.LOW


threat_scorer = MultiFactorThreatScorer()
