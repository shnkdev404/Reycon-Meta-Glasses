"""
Phase 6: Shared World Model & Perception Orchestrator.

Maintains synchronized state of all connected Meta Smart Glasses, fused 3D World Objects,
historical trajectories, and coordinates spatial fusion, threat prediction, and directed alerts.
"""
import asyncio
from typing import Dict, List
from datetime import datetime
from app.models.glass import GlassState
from app.models.object import Detection2D, WorldObject
from app.models.threat import ThreatAlert
from app.services.coordinate_transform import coordinate_transformer
from app.services.fusion_engine import fusion_engine
from app.services.prediction_engine import prediction_engine
from app.services.alert_engine import alert_engine
from app.utils.logger import get_logger

logger = get_logger("WorldManager")


class WorldManager:
    """
    Centralized in-memory World Model for the Shared Perception Stack.
    """

    def __init__(self):
        self._glasses: Dict[str, GlassState] = {}
        self._world_objects: Dict[str, WorldObject] = {}
        self._trajectories: Dict[str, List[Dict]] = {}
        self._active_threats: List[ThreatAlert] = []
        self._lock = asyncio.Lock()

    async def update_glass_telemetry(
        self,
        glass_state: GlassState,
        detections: List[Detection2D]
    ) -> List[ThreatAlert]:
        """
        Process a new telemetry and detection update from a smart glass unit.
        Transforms coordinates, fuses objects, evaluates threats, and dispatches directed alerts.
        """
        async with self._lock:
            # 1. Update Glass State & Trajectory
            self._glasses[glass_state.glass_id] = glass_state
            self._record_glass_trajectory(glass_state)

            # 2. Transform 2D Detections -> 3D World Coordinates
            new_world_objects = [
                coordinate_transformer.transform_detection_to_world(det, glass_state)
                for det in detections
            ]

            # 3. Perception Fusion (Cluster multi-glass observations & update velocities)
            self._world_objects = fusion_engine.fuse_objects(
                new_objects=new_world_objects,
                existing_world_objects=self._world_objects
            )

            # 4. Prune stale unobserved world objects
            self.prune_stale_world_objects()

            # 5. Threat Prediction & Trajectory Evaluation
            self._active_threats = prediction_engine.evaluate_threats(
                glasses=self._glasses,
                world_objects=self._world_objects
            )

            # 6. Dispatch Targeted Non-Broadcast Alerts (Sent ONLY to threatened glasses)
            if self._active_threats:
                await alert_engine.dispatch_alerts(self._active_threats)

            # 7. Output synchronized World Model
            self._print_world_summary()

            return self._active_threats

    def prune_stale_world_objects(self, max_age_seconds: float = 5.0):
        """Expire fused world objects that have not been observed recently."""
        now = datetime.utcnow()
        stale_ids = [
            obj_id for obj_id, obj in self._world_objects.items()
            if (now - obj.last_seen).total_seconds() > max_age_seconds
        ]
        for obj_id in stale_ids:
            del self._world_objects[obj_id]
            logger.info(f"🗑️ Pruned stale world object '{obj_id}'.")

    async def remove_glass(self, glass_id: str):
        """Remove glass from active world state upon disconnect."""
        async with self._lock:
            if glass_id in self._glasses:
                del self._glasses[glass_id]
                logger.info(f"📌 Glass '{glass_id}' removed from World Model.")
                self._print_world_summary()

    def get_glass(self, glass_id: str) -> Optional[GlassState]:
        """Retrieve state of a specific connected smart glass unit."""
        return self._glasses.get(glass_id)

    def get_glass_trajectory(self, glass_id: str) -> List[Dict]:
        """Retrieve 6DoF path trajectory history for a smart glass unit."""
        return self._trajectories.get(glass_id, [])

    def get_world_objects(self) -> Dict[str, WorldObject]:
        """Retrieve all currently active 3D fused WorldObjects."""
        return self._world_objects.copy()

    def get_active_threats(self) -> List[ThreatAlert]:
        """Retrieve currently active threat alerts."""
        return self._active_threats.copy()

    def reset_world_state(self):
        """Reset all active glasses, world objects, trajectories, and threat alerts."""
        self._glasses.clear()
        self._world_objects.clear()
        self._trajectories.clear()
        self._active_threats.clear()

    async def get_full_world_state(self) -> Dict:
        """Return serialized state of glasses, fused objects, and threats."""
        async with self._lock:
            return {
                "active_glasses_count": len(self._glasses),
                "glasses": {gid: g.model_dump(mode="json") for gid, g in self._glasses.items()},
                "world_objects": {oid: obj.model_dump(mode="json") for oid, obj in self._world_objects.items()},
                "active_threats": [t.model_dump(mode="json") for t in self._active_threats],
                "timestamp": datetime.utcnow().isoformat()
            }

    def _record_glass_trajectory(self, glass: GlassState):
        """Log glass position history for trajectory tracking."""
        if glass.glass_id not in self._trajectories:
            self._trajectories[glass.glass_id] = []
        
        self._trajectories[glass.glass_id].append({
            "x": glass.pose.x,
            "y": glass.pose.y,
            "heading": glass.pose.heading,
            "time": datetime.utcnow().isoformat()
        })
        # Keep last 50 trajectory points per glass
        if len(self._trajectories[glass.glass_id]) > 50:
            self._trajectories[glass.glass_id].pop(0)

    def _print_world_summary(self):
        """Print synchronized World Model summary to console."""
        print("\n========== SHARED PERCEPTION WORLD ==========")
        if not self._glasses:
            print("No active glasses connected.")
        else:
            for gid, glass in self._glasses.items():
                px = int(glass.pose.x) if glass.pose.x.is_integer() else glass.pose.x
                py = int(glass.pose.y) if glass.pose.y.is_integer() else glass.pose.y
                heading_val = int(glass.pose.heading) if glass.pose.heading.is_integer() else glass.pose.heading
                print(f"Glass: {gid}")
                print(f"Position: ({px},{py})")
                print(f"Heading: {heading_val}°")
                
                # Filter objects observed by this glass or nearby
                glass_objs = [o for o in self._world_objects.values() if gid in o.source_glasses]
                print("Detections:")
                if not glass_objs:
                    print("None")
                else:
                    for obj in glass_objs:
                        print(f"- {obj.label.capitalize()} (ID: {obj.object_id}, Pos: ({obj.position_x},{obj.position_y}))")
                print()

        if self._active_threats:
            print("[!] ACTIVE THREAT ALERTS:")
            for alert in self._active_threats:
                print(f"  └─ Target: {alert.target_glass_id} | {alert.warning_message}")

        print("=============================================\n")


world_manager = WorldManager()

