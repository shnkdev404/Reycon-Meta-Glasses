"""
Phase 8: Threat Prediction Engine.

Extrapolates 3D dynamic trajectories of vehicles, forklifts, and obstacles.
Calculates Time-To-Collision (TTC), relative distance, and relative bearing
to evaluate collision risk for each connected Meta Smart Glass unit.
"""
import uuid
from typing import List, Dict
from datetime import datetime
from app.models.glass import GlassState
from app.models.object import WorldObject
from app.models.threat import ThreatAlert, ThreatLevel, ThreatType
from app.utils.math import euclidean_distance_2d, calculate_bearing, calculate_ttc
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("PredictionEngine")


class ThreatPredictionEngine:
    """Predicts dynamic collisions and generates targeted spatial threat vectors."""

    def __init__(self, cooldown_seconds: float = 1.5):
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_times: Dict[Tuple[str, str], float] = {}

    def evaluate_threats(
        self,
        glasses: Dict[str, GlassState],
        world_objects: Dict[str, WorldObject]
    ) -> List[ThreatAlert]:
        """
        Evaluate all active world objects against connected glasses.
        Returns a list of targeted ThreatAlert instances.
        """
        active_alerts: List[ThreatAlert] = []

        for obj_id, obj in world_objects.items():
            label = obj.label.lower()
            clean_label = label.split(" #")[0].strip()
            
            # Target hazardous dynamic objects and obstacles
            if not any(k in clean_label for k in ["vehicle", "car", "forklift", "truck", "machine", "person", "obstacle", "hazard"]):
                continue

            for glass_id, glass in glasses.items():
                alert = self._check_glass_threat(glass, obj)
                if alert:
                    active_alerts.append(alert)

        return active_alerts

    def _check_glass_threat(self, glass: GlassState, obj: WorldObject) -> ThreatAlert | None:
        """Check if a specific world object poses a threat to a specific smart glass."""
        now_ts = datetime.utcnow().timestamp()
        cooldown_key = (glass.glass_id, obj.object_id)

        # Check cooldown suppression to avoid alert spam
        last_time = self._last_alert_times.get(cooldown_key, 0.0)
        if (now_ts - last_time) < self.cooldown_seconds:
            return None

        glass_pos = (glass.pose.x, glass.pose.y, glass.pose.z)
        glass_vel = (glass.velocity_x, glass.velocity_y, glass.velocity_z)

        obj_pos = (obj.position_x, obj.position_y, obj.position_z)
        obj_vel = (obj.velocity_x, obj.velocity_y, obj.velocity_z)

        dist = euclidean_distance_2d((glass.pose.x, glass.pose.y), (obj.position_x, obj.position_y))

        # Ignore if object is outside danger radius
        if dist > settings.DANGER_RADIUS_METERS:
            return None

        # Calculate exact or estimated Time-To-Collision (TTC)
        ttc = calculate_ttc(obj_pos, obj_vel, glass_pos, glass_vel)
        
        # Fallback for hazards detected within danger radius
        if ttc is None and dist <= settings.DANGER_RADIUS_METERS:
            ttc = dist / 2.5  # Estimated TTC assuming 2.5 m/s motion

        if ttc is None or ttc > settings.TTC_WARNING_THRESHOLD_SEC:
            return None

        # Determine relative bearing from target glass's orientation
        abs_bearing = calculate_bearing((glass.pose.x, glass.pose.y), (obj.position_x, obj.position_y))
        rel_bearing = (abs_bearing - glass.pose.heading + 360.0) % 360.0

        # Classify threat type and 4-tier severity level
        threat_level = self._classify_threat_level(ttc, dist)
        is_blind_spot = 135.0 <= rel_bearing < 225.0
        threat_type = self._classify_threat_type(obj.label, is_blind_spot)

        # Construct direction text
        direction_text = self._bearing_to_direction_text(rel_bearing)
        clean_name = obj.label.split(" #")[0].capitalize()
        warning_msg = f"{threat_level.value}: {clean_name} detected at {direction_text} ({dist:.1f}m away!)"

        alert = ThreatAlert(
            alert_id=f"alt_{uuid.uuid4().hex[:6]}",
            target_glass_id=glass.glass_id,
            trigger_object_id=obj.object_id,
            threat_type=threat_type,
            threat_level=threat_level,
            time_to_collision=round(ttc, 2),
            distance=round(dist, 2),
            bearing=round(rel_bearing, 1),
            warning_message=warning_msg,
            timestamp=datetime.utcnow()
        )

        self._last_alert_times[cooldown_key] = now_ts

        logger.warning(
            f"🎯 Threat Warning Generated for Target Glass [{glass.glass_id}]: {warning_msg} (TTC: {ttc:.1f}s)"
        )
        return alert

    @staticmethod
    def _classify_threat_level(ttc: float, dist: float) -> ThreatLevel:
        """4-Tier Threat Level Classification: CRITICAL, HIGH, MEDIUM, LOW."""
        if ttc < 2.0 or dist < 2.0:
            return ThreatLevel.CRITICAL
        elif ttc < 4.0 or dist <= 5.0:
            return ThreatLevel.HIGH
        elif ttc <= 7.0 or dist <= 8.0:
            return ThreatLevel.MEDIUM
        return ThreatLevel.LOW

    @staticmethod
    def _classify_threat_type(label: str, is_blind_spot: bool) -> ThreatType:
        l = label.lower()
        if "forklift" in l:
            return ThreatType.FORKLIFT_APPROACH
        elif "vehicle" in l or "car" in l or "truck" in l:
            return ThreatType.VEHICLE_APPROACH
        elif "person" in l:
            return ThreatType.PERSON_RUNNING
        elif "falling" in l:
            return ThreatType.FALLING_OBJECT
        elif is_blind_spot:
            return ThreatType.BLIND_SPOT_OBSTACLE
        return ThreatType.COLLISION_RISK

    @staticmethod
    def _bearing_to_direction_text(rel_bearing: float) -> str:
        if rel_bearing >= 315 or rel_bearing < 45:
            return "Front"
        elif 45 <= rel_bearing < 135:
            return "Right"
        elif 135 <= rel_bearing < 225:
            return "Behind (Blind Spot)"
        else:
            return "Left"


prediction_engine = ThreatPredictionEngine()

