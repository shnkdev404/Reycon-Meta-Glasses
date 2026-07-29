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
            
            # Exclude non-hazard objects (office/personal items & people) from hazard threat classification
            if any(k in clean_label for k in ["person", "human", "laptop", "phone", "chair", "bottle", "cup", "backpack", "keyboard", "mouse"]):
                continue

            # Target hazardous dynamic objects, machinery, and obstacles
            if not any(k in clean_label for k in ["vehicle", "car", "forklift", "truck", "excavator", "machine", "obstacle", "hazard"]):
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


    def predict_trajectory(
        self,
        obj: WorldObject,
        time_horizon_sec: float = 5.0,
        step_sec: float = 0.5
    ) -> List[Dict[str, float]]:
        """
        Extrapolate 3D trajectory path for an object into the future across a time horizon.
        P_future(t) = P_current + V * t
        """
        trajectory_points = []
        t = 0.0
        while t <= time_horizon_sec:
            px = obj.position_x + obj.velocity_x * t
            py = obj.position_y + obj.velocity_y * t
            pz = obj.position_z + obj.velocity_z * t
            trajectory_points.append({
                "time_offset_sec": round(t, 2),
                "x": round(px, 2),
                "y": round(py, 2),
                "z": round(pz, 2)
            })
            t += step_sec
        return trajectory_points

    def estimate_collision_probability(
        self,
        obj: WorldObject,
        glass: GlassState,
        miss_threshold_meters: float = 2.5
    ) -> Tuple[float, Optional[float]]:
        """
        Predict future trajectories of object and worker to estimate collision probability (0.0 to 1.0)
        and Time-To-Collision (TTC in seconds).
        """
        trajectory = self.predict_trajectory(obj, time_horizon_sec=6.0, step_sec=0.2)
        min_dist = float("inf")
        best_ttc = None

        for pt in trajectory:
            t = pt["time_offset_sec"]
            # Predict worker position assuming worker velocity or stationary
            g_px = glass.pose.x + glass.velocity_x * t
            g_py = glass.pose.y + glass.velocity_y * t

            dist = euclidean_distance_2d((g_px, g_py), (pt["x"], pt["y"]))
            if dist < min_dist:
                min_dist = dist
                best_ttc = t

        # Compute collision probability inversely proportional to miss distance
        if min_dist <= miss_threshold_meters:
            prob = max(0.0, min(1.0, 1.0 - (min_dist / miss_threshold_meters)))
        else:
            prob = 0.0

        return round(prob, 2), best_ttc


prediction_engine = ThreatPredictionEngine()

