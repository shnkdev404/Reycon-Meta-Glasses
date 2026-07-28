"""
Phase 3/5: World Object Tracking Manager.

Maintains temporal history of fused 3D WorldObjects to calculate smoothed trajectories.
"""
from typing import Dict, List
from app.models.object import WorldObject


from datetime import datetime
from typing import Dict, List, Optional
from app.models.object import WorldObject


class TrackingManager:
    """
    Manages persistent trajectories for fused 3D WorldObjects.
    Calculates 3D velocity vectors, performs trajectory smoothing, and prunes stale tracks.
    """

    def __init__(self, max_history: int = 30):
        self.max_history = max_history
        self._history: Dict[str, List[WorldObject]] = {}

    def update_tracks(self, current_objects: Dict[str, WorldObject]) -> Dict[str, WorldObject]:
        """
        Append latest object states to history, calculate 3D velocities, and apply spatial smoothing.
        Returns dictionary of updated WorldObjects populated with computed velocities.
        """
        updated_objects: Dict[str, WorldObject] = {}

        for obj_id, obj in current_objects.items():
            if obj_id not in self._history:
                self._history[obj_id] = []

            history = self._history[obj_id]
            updated_obj = obj.model_copy(deep=True)

            # Compute velocity if previous history exists
            if history:
                prev_obj = history[-1]
                dt = (updated_obj.last_seen - prev_obj.last_seen).total_seconds()
                
                if dt > 0.001:
                    vx = (updated_obj.position_x - prev_obj.position_x) / dt
                    vy = (updated_obj.position_y - prev_obj.position_y) / dt
                    vz = (updated_obj.position_z - prev_obj.position_z) / dt

                    # Exponential Moving Average (EMA) smoothing for velocity (alpha=0.6)
                    alpha = 0.6
                    updated_obj.velocity_x = round(alpha * vx + (1 - alpha) * prev_obj.velocity_x, 2)
                    updated_obj.velocity_y = round(alpha * vy + (1 - alpha) * prev_obj.velocity_y, 2)
                    updated_obj.velocity_z = round(alpha * vz + (1 - alpha) * prev_obj.velocity_z, 2)

            history.append(updated_obj)
            if len(history) > self.max_history:
                history.pop(0)

            updated_objects[obj_id] = updated_obj

        self.prune_stale_tracks()
        return updated_objects

    def prune_stale_tracks(self, max_age_seconds: float = 5.0):
        """Prune track history for objects not observed within max_age_seconds."""
        now = datetime.utcnow()
        stale_ids = []
        for obj_id, history in self._history.items():
            if history:
                last_seen = history[-1].last_seen
                if (now - last_seen).total_seconds() > max_age_seconds:
                    stale_ids.append(obj_id)

        for obj_id in stale_ids:
            del self._history[obj_id]

    def get_trajectory(self, object_id: str) -> List[WorldObject]:
        """Return the temporal trajectory history for a specified object ID."""
        return self._history.get(object_id, [])


tracking_manager = TrackingManager()

