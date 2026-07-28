"""
Phase 10: Meta Wearable SDK Directed Alert Adapter.

Adapter executing directed HUD, spatial audio, and haptic feedback alerts
on Ray-Ban Meta Smart Glasses.
TODO: Call Meta Wearable SDK Haptic & Audio APIs.
"""
from app.models.threat import ThreatAlert
from app.utils.logger import get_logger

logger = get_logger("MetaAlertAdapter")


from typing import Dict, Any
from app.models.threat import ThreatAlert, ThreatLevel
from app.utils.logger import get_logger

logger = get_logger("MetaAlertAdapter")


class MetaAlertAdapter:
    """Delivering targeted spatial warnings directly to Ray-Ban Meta Smart Glasses."""

    # Haptic feedback vibration profiles mapped to threat severity
    HAPTIC_PROFILES = {
        ThreatLevel.CRITICAL: {"intensity": 1.0, "duration_ms": 600, "pattern": "PULSE_RAPID"},
        ThreatLevel.HIGH: {"intensity": 0.75, "duration_ms": 400, "pattern": "PULSE_DOUBLE"},
        ThreatLevel.MEDIUM: {"intensity": 0.50, "duration_ms": 250, "pattern": "PULSE_SINGLE"},
        ThreatLevel.LOW: {"intensity": 0.25, "duration_ms": 150, "pattern": "SUBTLE"},
    }

    def __init__(self, glass_id: str):
        self.glass_id = glass_id

    async def send_alert(self, alert: ThreatAlert) -> bool:
        """
        Deliver directed spatial warning to the glasses HUD, spatial audio, and haptic engine.
        """
        # Step 1: Calculate Spatial Audio Azimuth Panning (-180° Left to +180° Right)
        azimuth_deg = (alert.bearing + 180.0) % 360.0 - 180.0
        
        # Step 2: Query Haptic Profile
        haptic = self.HAPTIC_PROFILES.get(alert.threat_level, self.HAPTIC_PROFILES[ThreatLevel.MEDIUM])

        # Step 3: Trigger Meta Wearable SDK HUD & Feedback Adapters
        logger.warning(
            f"🚨 [Meta Glasses HUD - {self.glass_id}] {alert.warning_message} "
            f"(TTC: {alert.time_to_collision:.1f}s, Audio Azimuth: {azimuth_deg:.1f}°, "
            f"Haptics: {haptic['intensity']*100:.0f}% intensity)"
        )
        return True

    def calculate_audio_azimuth(self, bearing_deg: float) -> float:
        """Compute relative spatial audio azimuth panning angle in degrees (-180° to +180°)."""
        return (bearing_deg + 180.0) % 360.0 - 180.0

