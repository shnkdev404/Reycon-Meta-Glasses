import math
import logging
import time
from typing import Dict, List, Any, Optional
from app.models import GlassState, Detection, GPSLocation

logger = logging.getLogger("WorldManager")

HAZARD_CLASSES = {
    "person", "human", "laptop", "cell phone", "phone", "chair", "bottle", "cup",
    "car", "truck", "bus", "motorcycle", "bicycle", "forklift", "excavator",
    "stop sign", "hazard", "obstacle", "backpack", "dog", "cat", "tv", "remote"
}


def calculate_gps_distance_and_bearing(lat1: float, lon1: float, lat2: float, lon2: float):
    """
    Calculate relative distance (in meters), bearing angle (in degrees),
    and local cartesian offsets (dx, dy in meters) between two GPS points.
    """
    R = 6371000.0  # Earth radius in meters
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat_rad = math.radians(lat2 - lat1)
    delta_lon_rad = math.radians(lon2 - lon1)

    # Local cartesian projection offset
    x = delta_lon_rad * math.cos((lat1_rad + lat2_rad) / 2.0)
    y = delta_lat_rad

    dx_m = x * R  # East-West offset in meters
    dy_m = y * R  # North-South offset in meters
    distance_m = math.sqrt(dx_m * dx_m + dy_m * dy_m)

    # Initial bearing angle calculation (0 = North, 90 = East)
    y_b = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    x_b = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)
    bearing_deg = (math.degrees(math.atan2(y_b, x_b)) + 360.0) % 360.0

    return round(distance_m, 2), round(bearing_deg, 1), round(dx_m, 2), round(dy_m, 2)


class WorldManager:
    """Maintains synchronized state of all connected mobile devices, GPS telemetry, and safety threats."""

    def __init__(self):
        self.active_glasses: Dict[str, GlassState] = {}
        self.active_threats: List[Dict[str, Any]] = []

    def update_glass(self, state: GlassState) -> Dict[str, Any]:
        """
        Update connected device state, calculate spatial radar blips and multi-level safety threats.
        Returns detailed spatial payload.
        """
        self.active_glasses[state.glass_id] = state
        self.active_threats = self.evaluate_threats()
        radar_blips = self.generate_radar_blips(origin_glass_id=state.glass_id)

        return {
            "threats": self.active_threats,
            "radar_blips": radar_blips,
            "all_devices_gps": self.get_all_devices_gps()
        }

    def get_glass(self, glass_id: str) -> Optional[GlassState]:
        """Retrieve connected device state by ID."""
        return self.active_glasses.get(glass_id)

    def get_glass_trajectory(self, glass_id: str) -> List[Any]:
        """Return motion trajectory history for specified device."""
        glass = self.active_glasses.get(glass_id)
        if glass:
            return [glass.position]
        return []

    def get_world_objects(self) -> Dict[str, Any]:
        """Return active world objects dictionary."""
        return self.get_world_state()["world_objects"]

    def reset_world_state(self):
        """Clear all active devices and threat alerts."""
        self.active_glasses.clear()
        self.active_threats.clear()

    def prune_stale_world_objects(self, max_age_seconds: float = 5.0):
        """Prune devices that have not transmitted telemetry recently."""
        now = time.time()
        stale_ids = [
            gid for gid, state in self.active_glasses.items()
            if (now - state.timestamp) > max_age_seconds
        ]
        for gid in stale_ids:
            self.remove_glass(gid)

    async def update_glass_telemetry(self, glass_state: GlassState, detections: Any = None):
        """Process glass telemetry update and evaluate threats."""
        if detections and isinstance(detections, list):
            glass_state.detections = detections
        return self.update_glass(glass_state)["threats"]

    async def get_full_world_state(self) -> Dict[str, Any]:
        """Return full world model summary for API routers and test runners."""
        return self.get_world_state()

    def remove_glass(self, glass_id: str):
        """Remove device from active world state."""
        if glass_id in self.active_glasses:
            del self.active_glasses[glass_id]
            logger.info(f"Removed device '{glass_id}' from WorldManager.")
            self.active_threats = self.evaluate_threats()

    def evaluate_threats(self) -> List[Dict[str, Any]]:
        """Scan detections across all connected devices and generate active hazard threat objects with multi-tier severity levels."""
        threats: List[Dict[str, Any]] = []

        for glass_id, state in self.active_glasses.items():
            gps_info = state.gps.model_dump() if state.gps else None
            for detection in state.detections:
                obj_name = getattr(detection, 'class_name', None) or getattr(detection, 'label', '')
                direction = getattr(detection, 'direction', 'FRONT')
                confidence = getattr(detection, 'confidence', 0.9)
                bbox = getattr(detection, 'bbox', [])
                if isinstance(bbox, list):
                    pass
                else:
                    bbox = []

                if obj_name.split(' #')[0].lower() in HAZARD_CLASSES:
                    # Classify threat severity (CRITICAL vs WARNING)
                    is_critical = (direction == "FRONT") or (confidence > 0.85)
                    severity = "CRITICAL" if is_critical else "WARNING"

                    threat = {
                        "glass_id": glass_id,
                        "severity": severity,
                        "class_name": obj_name,
                        "confidence": confidence,
                        "direction": direction,
                        "bbox": bbox,
                        "heading": state.heading,
                        "gps": gps_info,
                        "timestamp": time.time(),
                        "warning_message": (
                            f"{severity}: {obj_name.upper()} detected on {direction} zone!"
                        )
                    }
                    threats.append(threat)

        return threats

    def generate_radar_blips(self, origin_glass_id: str) -> List[Dict[str, Any]]:
        """
        Compute relative spatial radar blip coordinates for all connected devices and detections
        relative to the target origin device.
        """
        origin_state = self.active_glasses.get(origin_glass_id)
        blips: List[Dict[str, Any]] = []

        if not origin_state:
            return blips

        origin_gps = origin_state.gps

        for gid, state in self.active_glasses.items():
            if gid == origin_glass_id:
                # Origin self-device at center (0, 0)
                blips.append({
                    "id": gid,
                    "type": "SELF",
                    "label": f"{gid} (You)",
                    "dx_m": 0.0,
                    "dy_m": 0.0,
                    "distance_m": 0.0,
                    "bearing_deg": origin_state.heading,
                    "heading": origin_state.heading,
                    "gps": origin_gps.model_dump() if origin_gps else None
                })
            else:
                dx_m, dy_m, distance_m, bearing_deg = 0.0, 0.0, 0.0, 0.0
                if origin_gps and state.gps:
                    distance_m, bearing_deg, dx_m, dy_m = calculate_gps_distance_and_bearing(
                        origin_gps.latitude, origin_gps.longitude,
                        state.gps.latitude, state.gps.longitude
                    )

                blips.append({
                    "id": gid,
                    "type": "DEVICE",
                    "label": gid,
                    "dx_m": dx_m,
                    "dy_m": dy_m,
                    "distance_m": distance_m,
                    "bearing_deg": bearing_deg,
                    "heading": state.heading,
                    "gps": state.gps.model_dump() if state.gps else None
                })

            # Add vision detections as relative hazard blips
            for idx, det in enumerate(state.detections):
                direction = getattr(det, 'direction', 'FRONT')
                obj_name = getattr(det, 'class_name', None) or getattr(det, 'label', 'object')
                confidence = getattr(det, 'confidence', 0.9)

                angle_offset = 0.0
                if direction == "LEFT":
                    angle_offset = -30.0
                elif direction == "RIGHT":
                    angle_offset = 30.0

                det_bearing = (state.heading + angle_offset) % 360.0
                est_dist = 5.0
                rad = math.radians(det_bearing)
                det_dx = round(est_dist * math.sin(rad), 2)
                det_dy = round(est_dist * math.cos(rad), 2)

                is_hazard = obj_name.split(' #')[0].lower() in HAZARD_CLASSES
                blips.append({
                    "id": f"{gid}_det_{idx}",
                    "type": "HAZARD" if is_hazard else "OBJECT",
                    "label": f"{obj_name.upper()} ({int(confidence * 100)}%)",
                    "dx_m": det_dx,
                    "dy_m": det_dy,
                    "distance_m": est_dist,
                    "bearing_deg": det_bearing,
                    "heading": state.heading,
                    "gps": None
                })

        return blips

    def get_all_devices_gps(self) -> List[Dict[str, Any]]:
        """Return list of all connected devices with their current GPS locations."""
        devices_gps = []
        for gid, state in self.active_glasses.items():
            devices_gps.append({
                "glass_id": gid,
                "heading": state.heading,
                "gps": state.gps.model_dump() if state.gps else None,
                "last_seen": state.timestamp
            })
        return devices_gps

    def get_world_state(self) -> Dict[str, Any]:
        """Return full current snapshot of active devices, GPS locations, and threats."""
        world_objs = {}
        for gid, g in self.active_glasses.items():
            for idx, det in enumerate(g.detections):
                obj_name = getattr(det, 'class_name', None) or getattr(det, 'label', 'obj')
                world_objs[f"{gid}_obj_{idx}"] = {
                    "object_id": f"{gid}_obj_{idx}",
                    "label": obj_name,
                    "confidence": getattr(det, 'confidence', 0.9)
                }

        return {
            "active_devices_count": len(self.active_glasses),
            "active_glasses_count": len(self.active_glasses),
            "glasses": {gid: g.model_dump() for gid, g in self.active_glasses.items()},
            "world_objects": world_objs,
            "active_threats": self.active_threats,
            "timestamp": time.time()
        }


world_manager = WorldManager()
