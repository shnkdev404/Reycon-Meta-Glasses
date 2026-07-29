"""
Shared World Model - The Intelligence Center
Manages all glasses, detections, threats, and the global map.

Key Feature: If Glass B detects threat → Glass A gets notified
            even if Glass A can't see it!
"""
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
        
        # Thread lock for concurrent access
        self.lock = threading.RLock()
        
        # Load saved map
        self.load_map()
        
        logger.info("✅ SharedWorldManager initialized")
    
    # ============ GLASS MANAGEMENT ============
    
    def register_glass(self, glass_id: str, position: Position3D):
        """Register a new glass device"""
        with self.lock:
            self.glasses[glass_id] = {
                'id': glass_id,
                'position': position,
                'pose': np.eye(4),  # Identity transform
                'timestamp': datetime.now().timestamp(),
                'connected': True,
                'map_points': [],  # Points visible from this glass
                'threat_list': [],  # Threats it can see
            }
            logger.info(f"✅ Glass registered: {glass_id}")
    
    def update_glass_pose(self, glass_id: str, pose: np.ndarray, position: Position3D):
        """Update glass position and orientation"""
        with self.lock:
            if glass_id not in self.glasses:
                self.register_glass(glass_id, position)
            
            self.glasses[glass_id]['pose'] = pose
            self.glasses[glass_id]['position'] = position
            self.glasses[glass_id]['timestamp'] = datetime.now().timestamp()
    
    def get_glass_position(self, glass_id: str) -> Optional[Position3D]:
        """Get glass position"""
        with self.lock:
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
    ) -> ThreatObject:
        """
        Glass detects a threat and reports it to the server.
        Server makes it visible to ALL nearby glasses.
        
        KEY FEATURE: Glass B detects threat → All glasses within range get alerted!
        """
        with self.lock:
            threat = ThreatObject(
                threat_id=threat_id,
                object_type=object_type,
                position=position,
                velocity=velocity,
                confidence=confidence,
                detected_by_glass_id=detected_by_glass_id
            )
            
            self.threats[threat_id] = threat
            
            # Add to history (for memory/persistence)
            self.threat_history.append({
                'threat_id': threat_id,
                'type': object_type,
                'position': position.to_dict(),
                'detected_by': detected_by_glass_id,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"🚨 Threat detected: {object_type} at {position}")
            
            return threat
    
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
    
    def reset(self):
        """Reset all data (for new session)"""
        with self.lock:
            self.glasses.clear()
            self.threats.clear()
            self.map_points.clear()
            self.keyframes.clear()
            self.threat_history.clear()
            logger.info("🔄 World manager reset")

# Global instance
world_manager = SharedWorldManager()