"""
Shared World Model - The Intelligence Center
Manages all glasses, detections, threats, and the global map.

Key Feature: If Glass B detects threat → Glass A gets notified
            even if Glass A can't see it!
"""
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class Position3D:
    """3D world position"""
    x: float
    y: float
    z: float = 0.0
    
    def to_dict(self):
        return {'x': self.x, 'y': self.y, 'z': self.z}
    
    def distance_to(self, other: 'Position3D') -> float:
        """Euclidean distance"""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

@dataclass
class ThreatObject:
    """Detected threat object"""
    threat_id: str
    object_type: str  # 'person', 'truck', 'forklift', etc.
    position: Position3D
    velocity: Tuple[float, float, float] = (0, 0, 0)
    confidence: float = 0.8
    detected_by_glass_id: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    last_seen: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def is_fresh(self, timeout: float = 5.0) -> bool:
        """Check if threat is still active"""
        age = datetime.now().timestamp() - self.last_seen
        return age < timeout

@dataclass
class MapPoint:
    """3D landmark/map point"""
    point_id: str
    position: Position3D
    descriptor: Optional[np.ndarray] = None
    observations: Dict[str, int] = field(default_factory=dict)  # glass_id -> frame_count
    color: Tuple[int, int, int] = (255, 0, 0)
    
    def to_dict(self):
        return {
            'point_id': self.point_id,
            'position': self.position.to_dict(),
            'observations': self.observations
        }

@dataclass
class Keyframe:
    """SLAM keyframe from each glass"""
    keyframe_id: str
    glass_id: str
    pose: np.ndarray  # 4x4 transformation matrix
    timestamp: float
    map_points: List[str] = field(default_factory=list)  # point_ids

def extract_heading_deg(data: dict) -> float:
    """
    Extract 0-360 compass heading / yaw angle in degrees from diverse telemetry formats:
    - Direct scalar: 'heading', 'compass', 'yaw', 'bearing'
    - Nested dict: 'pose': {'heading': ...} or {'yaw': ...}
    - 4x4 Transformation matrix array: 'pose': [[r00, r01, ...], ...] -> yaw = atan2(R[1, 0], R[0, 0])
    - Quaternion: 'orientation': {'x':..., 'y':..., 'z':..., 'w':...} -> yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    - Direction vector: 'direction': {'x':..., 'y':...} or [dx, dy] -> yaw = atan2(dy, dx)
    """
    if not isinstance(data, dict):
        return 0.0

    # 1. Direct scalar field
    for k in ["heading", "compass", "yaw", "bearing", "orientation_deg"]:
        val = data.get(k)
        if val is not None and isinstance(val, (int, float)):
            return float(val) % 360.0

    # 2. Check pose field
    raw_pose = data.get("pose")
    if isinstance(raw_pose, dict):
        for k in ["heading", "yaw", "compass"]:
            if k in raw_pose and isinstance(raw_pose[k], (int, float)):
                return float(raw_pose[k]) % 360.0
    elif isinstance(raw_pose, (list, np.ndarray)):
        arr = np.array(raw_pose, dtype=float)
        if arr.shape == (4, 4) or arr.shape == (3, 3):
            # Rotation matrix: yaw angle in XY plane
            r00, r01 = arr[0, 0], arr[0, 1]
            r10, r11 = arr[1, 0], arr[1, 1]
            yaw_rad = math.atan2(r10, r00)
            return math.degrees(yaw_rad) % 360.0

    # 3. Check orientation field (quaternion or euler)
    orient = data.get("orientation")
    if isinstance(orient, dict):
        if "yaw" in orient and isinstance(orient["yaw"], (int, float)):
            return float(orient["yaw"]) % 360.0
        if all(k in orient for k in ["x", "y", "z", "w"]):
            x, y, z, w = float(orient["x"]), float(orient["y"]), float(orient["z"]), float(orient["w"])
            yaw_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            return math.degrees(yaw_rad) % 360.0
    elif isinstance(orient, (list, tuple)) and len(orient) == 4:
        x, y, z, w = float(orient[0]), float(orient[1]), float(orient[2]), float(orient[3])
        yaw_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return math.degrees(yaw_rad) % 360.0

    # 4. Check direction vector
    direction = data.get("direction")
    if isinstance(direction, (list, tuple)) and len(direction) >= 2:
        dx, dy = float(direction[0]), float(direction[1])
        if abs(dx) > 1e-4 or abs(dy) > 1e-4:
            return math.degrees(math.atan2(dy, dx)) % 360.0
    elif isinstance(direction, dict) and "x" in direction and "y" in direction:
        dx, dy = float(direction["x"]), float(direction["y"])
        if abs(dx) > 1e-4 or abs(dy) > 1e-4:
            return math.degrees(math.atan2(dy, dx)) % 360.0

    return 0.0


class SharedWorldManager:
    """
    Central server that maintains:
    1. All glasses' positions and orientations
    2. All detected threats
    3. Shared 3D map (landmarks)
    4. Threat history and patterns
    5. Cross-glass collaboration
    """
    
    def __init__(self):
        self.glasses: Dict[str, dict] = {}  # glass_id -> {position, pose, etc}
        self.threats: Dict[str, ThreatObject] = {}  # threat_id -> ThreatObject
        self.map_points: Dict[str, MapPoint] = {}  # point_id -> MapPoint
        self.keyframes: Dict[str, Keyframe] = {}  # keyframe_id -> Keyframe
        
        # Threat history for persistence
        self.threat_history: List[dict] = []  # Log of all threats
        self.threat_timeout = 10.0  # Seconds
        
        # Map persistence
        self.map_file = "data/shared_map.json"
        self.keyframe_file = "data/keyframes.npz"
        
        # GPS Reference Anchor for relative room coordinate projection
        self.ref_gps: Optional[Tuple[float, float]] = None
        
        # Thread lock for concurrent access
        self.lock = threading.RLock()
        
        # Load saved map and sync persistent objects
        self.load_map()
        self.sync_persistent_objects()
        
        logger.info("✅ SharedWorldManager initialized")

    def resolve_glass_position(self, position_in: Any, gps_info: Optional[Any] = None) -> Position3D:
        """
        Resolve 3D Cartesian position from position dictionary/object or GPS geographic coordinates.
        If position is zero/empty and valid GPS coordinates are supplied, project GPS location relative
        to the reference anchor GPS location.
        """
        x, y, z = 0.0, 0.0, 0.0
        if hasattr(position_in, "x"):
            x = float(getattr(position_in, "x", 0.0))
            y = float(getattr(position_in, "y", 0.0))
            z = float(getattr(position_in, "z", 0.0))
        elif isinstance(position_in, dict):
            x = float(position_in.get("x", 0.0))
            y = float(position_in.get("y", 0.0))
            z = float(position_in.get("z", 0.0))

        # Convert GPS to relative Cartesian meters if x and y are 0.0
        if abs(x) < 1e-4 and abs(y) < 1e-4 and gps_info:
            lat, lon = None, None
            if hasattr(gps_info, "latitude"):
                lat = float(getattr(gps_info, "latitude", 0.0))
                lon = float(getattr(gps_info, "longitude", 0.0))
            elif isinstance(gps_info, dict):
                lat = float(gps_info.get("latitude", 0.0))
                lon = float(gps_info.get("longitude", 0.0))

            if lat is not None and lon is not None and abs(lat) > 1.0 and abs(lon) > 1.0:
                if self.ref_gps is None:
                    self.ref_gps = (lat, lon)
                    x, y = 0.0, 0.0
                else:
                    ref_lat, ref_lon = self.ref_gps
                    dlat_rad = math.radians(lat - ref_lat)
                    dlon_rad = math.radians(lon - ref_lon)
                    avg_lat_rad = math.radians((ref_lat + lat) / 2.0)
                    R_earth = 6371000.0
                    x = dlon_rad * math.cos(avg_lat_rad) * R_earth
                    y = dlat_rad * R_earth

        return Position3D(x=round(x, 2), y=round(y, 2), z=round(z, 2))

    
    # ============ GLASS MANAGEMENT ============
    
    def register_glass(self, glass_id: str, position: Position3D, heading: float = 0.0):
        """Register a new glass device"""
        with self.lock:
            self.glasses[glass_id] = {
                'id': glass_id,
                'position': position,
                'heading': float(heading),
                'pose': np.eye(4),  # Identity transform
                'timestamp': datetime.now().timestamp(),
                'connected': True,
                'map_points': [],  # Points visible from this glass
                'threat_list': [],  # Threats it can see
            }
            logger.info(f"✅ Glass registered: {glass_id} at {position} heading={heading}°")
    
    def update_glass_pose(self, glass_id: str, pose: np.ndarray, position: Position3D, heading: float = 0.0, gps_info: Optional[Any] = None):
        """Update glass position and orientation"""
        with self.lock:
            resolved_pos = self.resolve_glass_position(position, gps_info)
            if glass_id not in self.glasses:
                self.register_glass(glass_id, resolved_pos, heading)
            else:
                self.glasses[glass_id]['pose'] = pose
                self.glasses[glass_id]['position'] = resolved_pos
                self.glasses[glass_id]['heading'] = float(heading)
                self.glasses[glass_id]['timestamp'] = datetime.now().timestamp()
                self.glasses[glass_id]['connected'] = True

    
    def prune_stale_glasses(self, max_age_seconds: float = 15.0):
        """Remove inactive glass devices that have not sent telemetry recently."""
        with self.lock:
            now = datetime.now().timestamp()
            stale_ids = [
                gid for gid, ginfo in self.glasses.items()
                if (now - ginfo.get('timestamp', 0)) > max_age_seconds
            ]
            for gid in stale_ids:
                del self.glasses[gid]
                logger.info(f"🧹 Pruned stale glass device '{gid}' due to inactivity.")

    def get_glass_position(self, glass_id: str) -> Optional[Position3D]:
        """Get glass position"""
        with self.lock:
            self.prune_stale_glasses()
            if glass_id in self.glasses:
                return self.glasses[glass_id]['position']
        return None
    
    # ============ SHARED THREAT DETECTION ============
    
    def add_threat(
        self,
        threat_id: str,
        object_type: str,
        position: Position3D,
        detected_by_glass_id: str,
        confidence: float = 0.8,
        velocity: Tuple[float, float, float] = (0, 0, 0)
    ) -> Optional[ThreatObject]:
        """
        Glass detects a threat and reports it to the server.
        Server makes it visible to ALL nearby glasses.
        Deduplicates nearby existing threats detected by the same glass.
        """
        clean_type = object_type.lower().split(" #")[0].strip()

        with self.lock:
            # Check if matching threat exists nearby from same glass
            existing_match_id = None
            for tid, existing in self.threats.items():
                if existing.detected_by_glass_id == detected_by_glass_id and existing.object_type == object_type:
                    if existing.position.distance_to(position) <= 2.0:
                        existing_match_id = tid
                        break
            
            target_id = existing_match_id or threat_id

            threat = ThreatObject(
                threat_id=target_id,
                object_type=object_type,
                position=position,
                velocity=velocity,
                confidence=confidence,
                detected_by_glass_id=detected_by_glass_id
            )
            
            self.threats[target_id] = threat
            
            # Add to history
            self.threat_history.append({
                'threat_id': target_id,
                'type': object_type,
                'position': position.to_dict(),
                'detected_by': detected_by_glass_id,
                'timestamp': datetime.now().isoformat()
            })

            # Save to persistent memory store
            try:
                from app.services.memory_manager import memory_manager
                memory_manager.add_or_update_object(
                    object_id=target_id,
                    label=object_type,
                    position=position.to_dict(),
                    detected_by=detected_by_glass_id,
                    confidence=confidence
                )
            except Exception as e:
                logger.error(f"Error persisting object to memory: {e}")
            
            logger.info(f"🚨 Threat updated: {object_type} [{target_id}] at {position}")
            return threat

    def correct_object_label(self, object_id: str, new_label: str) -> Optional[dict]:
        """Correct object label in active threats, map memory, and persistent disk store."""
        with self.lock:
            from app.services.memory_manager import memory_manager
            updated_obj = memory_manager.correct_object_label(object_id, new_label)

            # Update active threat if present
            if object_id in self.threats:
                self.threats[object_id].object_type = new_label
            else:
                for tid, threat in self.threats.items():
                    if object_id in tid or tid in object_id:
                        threat.object_type = new_label

            return updated_obj

    
    def get_threats_for_glass(self, glass_id: str, max_distance: float = 20.0) -> List[ThreatObject]:
        """
        Get all threats visible from this glass.
        
        KEY FEATURE: Returns threats detected by OTHER glasses too!
        (Not just what this glass sees, but what nearby glasses see)
        """
        with self.lock:
            glass_pos = self.glasses.get(glass_id, {}).get('position')
            if not glass_pos:
                return []
            
            # Get threats within range (from ANY glass!)
            visible_threats = []
            for threat in self.threats.values():
                if not threat.is_fresh(self.threat_timeout):
                    continue
                
                distance = glass_pos.distance_to(threat.position)
                
                # If threat is within 50m, this glass should know about it
                if distance < max_distance:
                    visible_threats.append(threat)
                else:
                    # EVEN if beyond camera range, notify if nearby glass detected it
                    if threat.detected_by_glass_id != glass_id:
                        # Other glass detected threat - notify anyway!
                        visible_threats.append(threat)
            
            return visible_threats
    
    def update_threat(self, threat_id: str, position: Position3D, velocity: Tuple = None):
        """Update threat position (continuous tracking)"""
        with self.lock:
            if threat_id in self.threats:
                self.threats[threat_id].position = position
                self.threats[threat_id].last_seen = datetime.now().timestamp()
                if velocity:
                    self.threats[threat_id].velocity = velocity
    
    def get_all_threats(self) -> List[dict]:
        """Get all active threats (for dashboard)"""
        with self.lock:
            return [
                {
                    'threat_id': t.threat_id,
                    'type': t.object_type,
                    'position': t.position.to_dict(),
                    'velocity': t.velocity,
                    'confidence': t.confidence,
                    'detected_by': t.detected_by_glass_id,
                    'fresh': t.is_fresh()
                }
                for t in self.threats.values()
            ]
    
    # ============ SHARED MAP MANAGEMENT ============
    
    def add_map_point(
        self,
        point_id: str,
        position: Position3D,
        glass_id: str,
        descriptor: Optional[np.ndarray] = None
    ) -> MapPoint:
        """
        Add 3D landmark to shared map.
        All glasses see the same landmarks!
        """
        with self.lock:
            if point_id not in self.map_points:
                point = MapPoint(
                    point_id=point_id,
                    position=position,
                    descriptor=descriptor
                )
                self.map_points[point_id] = point
            else:
                point = self.map_points[point_id]
            
            # Track which glasses can see this point
            if glass_id not in point.observations:
                point.observations[glass_id] = 1
            else:
                point.observations[glass_id] += 1
            
            logger.debug(f"📍 Map point added: {point_id}")
            return point
    
    def get_map_points(self, region: Optional[Tuple[float, float, float, float]] = None) -> List[MapPoint]:
        """
        Get all map points in region (or all if no region).
        
        region = (min_x, min_y, max_x, max_y)
        """
        with self.lock:
            if region is None:
                return list(self.map_points.values())
            
            min_x, min_y, max_x, max_y = region
            return [
                point for point in self.map_points.values()
                if min_x <= point.position.x <= max_x and
                   min_y <= point.position.y <= max_y
            ]
    
    def add_keyframe(
        self,
        keyframe_id: str,
        glass_id: str,
        pose: np.ndarray,
        map_point_ids: List[str]
    ) -> Keyframe:
        """Store SLAM keyframe from glass"""
        with self.lock:
            kf = Keyframe(
                keyframe_id=keyframe_id,
                glass_id=glass_id,
                pose=pose,
                timestamp=datetime.now().timestamp(),
                map_points=map_point_ids
            )
            self.keyframes[keyframe_id] = kf
            logger.debug(f"🎬 Keyframe added: {keyframe_id} from {glass_id}")
            return kf
    
    def get_recent_keyframes(self, glass_id: Optional[str] = None, limit: int = 100) -> List[Keyframe]:
        """Get recent keyframes (optionally filtered by glass)"""
        with self.lock:
            kfs = self.keyframes.values()
            if glass_id:
                kfs = [kf for kf in kfs if kf.glass_id == glass_id]
            
            # Sort by timestamp, return most recent
            return sorted(kfs, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    # ============ CROSS-GLASS COLLABORATION ============
    
    def find_collaborating_glasses(self, glass_id: str, max_distance: float = 50.0) -> List[str]:
        """
        Find other glasses within collaboration distance.
        These glasses should share their observations!
        """
        with self.lock:
            glass_pos = self.glasses.get(glass_id, {}).get('position')
            if not glass_pos:
                return []
            
            collaborators = []
            for other_id, other_glass in self.glasses.items():
                if other_id == glass_id:
                    continue
                
                other_pos = other_glass.get('position')
                if other_pos:
                    dist = glass_pos.distance_to(other_pos)
                    if dist < max_distance:
                        collaborators.append(other_id)
            
            return collaborators
    
    def get_alerts_for_glass(self, glass_id: str) -> List[dict]:
        """
        Get all alerts this glass should receive.
        
        Includes:
        1. Threats it detected itself
        2. Threats other glasses detected nearby
        3. Threats in its path
        """
        with self.lock:
            glass_pos = self.get_glass_position(glass_id)
            if not glass_pos:
                return []
            
            alerts = []
            
            for threat in self.threats.values():
                if not threat.is_fresh():
                    continue
                
                distance = glass_pos.distance_to(threat.position)
                
                # Alert if:
                # 1. This glass detected it
                # 2. Distance < 20m (nearby)
                # 3. Distance < 50m AND detected by collaborating glass
                
                if threat.detected_by_glass_id == glass_id:
                    priority = "DIRECT"  # You detected it
                elif distance < 20.0:
                    priority = "CLOSE"   # It's close to you
                elif distance < 50.0 and threat.detected_by_glass_id != glass_id:
                    priority = "SHARED"  # Other glass detected it
                else:
                    continue
                
                alerts.append({
                    'threat_id': threat.threat_id,
                    'type': threat.object_type,
                    'position': threat.position.to_dict(),
                    'distance': distance,
                    'priority': priority,
                    'detected_by': threat.detected_by_glass_id,
                    'confidence': threat.confidence,
                    'velocity': threat.velocity
                })
            
            return sorted(alerts, key=lambda x: x['distance'])
    
    # ============ MAP PERSISTENCE ============
    
    def save_map(self):
        """Save map to disk (persistent memory)"""
        with self.lock:
            try:
                # Save map points
                map_data = {
                    point_id: {
                        'position': point.position.to_dict(),
                        'observations': point.observations
                    }
                    for point_id, point in self.map_points.items()
                }
                
                with open(self.map_file, 'w') as f:
                    json.dump(map_data, f)
                
                logger.info(f"💾 Map saved: {len(self.map_points)} points")
            except Exception as e:
                logger.error(f"Error saving map: {e}")
    
    def load_map(self):
        """Load map from disk"""
        try:
            with open(self.map_file, 'r') as f:
                map_data = json.load(f)
            
            for point_id, data in map_data.items():
                pos = data['position']
                point = MapPoint(
                    point_id=point_id,
                    position=Position3D(pos['x'], pos['y'], pos['z']),
                    observations=data.get('observations', {})
                )
                self.map_points[point_id] = point
            
            logger.info(f"📂 Map loaded: {len(self.map_points)} points")
        except FileNotFoundError:
            logger.info("No saved map found")
        except Exception as e:
            logger.error(f"Error loading map: {e}")
    
    def get_map_statistics(self) -> dict:
        """Get map statistics"""
        with self.lock:
            return {
                'total_map_points': len(self.map_points),
                'total_keyframes': len(self.keyframes),
                'total_threats_detected': len(self.threat_history),
                'active_threats': sum(1 for t in self.threats.values() if t.is_fresh()),
                'connected_glasses': sum(1 for g in self.glasses.values() if g.get('connected'))
            }
    
    def sync_persistent_objects(self):
        """Load persistent remembered objects from PersistentMemoryManager into initial threats."""
        try:
            from app.services.memory_manager import memory_manager
            objs = memory_manager.get_all_persistent_objects()
            for obj in objs:
                tid = obj.get("object_id", "")
                lbl = obj.get("label", "object")
                pos_dict = obj.get("position", {})
                pos = Position3D(
                    x=float(pos_dict.get("x", 0.0)) if isinstance(pos_dict, dict) else 0.0,
                    y=float(pos_dict.get("y", 0.0)) if isinstance(pos_dict, dict) else 0.0,
                    z=float(pos_dict.get("z", 0.0)) if isinstance(pos_dict, dict) else 0.0
                )
                by_id = obj.get("detected_by", "system")
                conf = float(obj.get("confidence", 0.85))

                threat = ThreatObject(
                    threat_id=tid,
                    object_type=lbl,
                    position=pos,
                    confidence=conf,
                    detected_by_glass_id=by_id
                )
                self.threats[tid] = threat
            logger.info(f"📂 Synced {len(objs)} persistent objects into SharedWorldManager.")
        except Exception as e:
            logger.error(f"Error syncing persistent objects into SharedWorldManager: {e}")

    def reset(self, clear_persistent_memory: bool = True):
        """Reset all data (for new session)"""
        with self.lock:
            self.glasses.clear()
            self.threats.clear()
            self.map_points.clear()
            self.keyframes.clear()
            self.threat_history.clear()
            self.ref_gps = None
            if clear_persistent_memory:
                try:
                    from app.services.memory_manager import memory_manager
                    memory_manager.clear_persistent_objects()
                except Exception as e:
                    logger.error(f"Error clearing persistent memory on reset: {e}")
            logger.info("🔄 World manager reset")


# Global instance
world_manager = SharedWorldManager()
