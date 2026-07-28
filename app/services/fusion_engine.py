"""
Phase 7: Perception Fusion Engine.

Fuses perception telemetry from multiple smart glasses observing the same 3D environment.
Avoids duplicates, clusters spatial observations, boosts confidence for multi-observed objects,
and computes estimated 3D velocity vectors.
"""
from typing import List, Dict
from datetime import datetime
from app.models.object import WorldObject
from app.utils.math import euclidean_distance_3d
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("FusionEngine")


class PerceptionFusionEngine:
    """
    Merges overlapping spatial detections from multiple Meta Smart Glasses.
    De-duplicates objects, boosts confidence for multi-observed targets,
    and updates 3D velocity vectors.
    """

    def fuse_objects(
        self,
        new_objects: List[WorldObject],
        existing_world_objects: Dict[str, WorldObject]
    ) -> Dict[str, WorldObject]:
        """
        Fuse newly transformed world objects with the existing world model objects.
        """
        fused_world: Dict[str, WorldObject] = dict(existing_world_objects)

        for new_obj in new_objects:
            matched_id = self._find_matching_object(new_obj, fused_world)

            if matched_id:
                # Merge observation into existing global object
                existing = fused_world[matched_id]
                
                # Combine observing glasses list without duplicates
                combined_glasses = list(set(existing.source_glasses + new_obj.source_glasses))
                num_glasses = len(combined_glasses)

                # Compute elapsed time for 3D velocity estimation
                dt = (new_obj.last_seen - existing.last_seen).total_seconds()
                vel_x = existing.velocity_x
                vel_y = existing.velocity_y
                vel_z = existing.velocity_z

                if dt > 0.01:
                    inst_vx = (new_obj.position_x - existing.position_x) / dt
                    inst_vy = (new_obj.position_y - existing.position_y) / dt
                    inst_vz = (new_obj.position_z - existing.position_z) / dt

                    # Exponential Moving Average (EMA) smoothing for velocity (alpha = 0.6)
                    alpha = 0.6
                    vel_x = alpha * inst_vx + (1 - alpha) * existing.velocity_x
                    vel_y = alpha * inst_vy + (1 - alpha) * existing.velocity_y
                    vel_z = alpha * inst_vz + (1 - alpha) * existing.velocity_z

                # Confidence-weighted spatial centroid update
                w_existing = max(0.1, existing.confidence)
                w_new = max(0.1, new_obj.confidence)
                total_w = w_existing + w_new

                fused_x = (w_existing * existing.position_x + w_new * new_obj.position_x) / total_w
                fused_y = (w_existing * existing.position_y + w_new * new_obj.position_y) / total_w
                fused_z = (w_existing * existing.position_z + w_new * new_obj.position_z) / total_w

                # Boost confidence for multi-glass confirmations (+0.10 per extra observing glass, max 1.0)
                max_base_conf = max(existing.confidence, new_obj.confidence)
                boosted_conf = min(1.0, max_base_conf + 0.10 * (num_glasses - 1))

                fused_world[matched_id] = WorldObject(
                    object_id=matched_id,
                    label=existing.label,
                    confidence=round(boosted_conf, 2),
                    position_x=round(fused_x, 2),
                    position_y=round(fused_y, 2),
                    position_z=round(fused_z, 2),
                    velocity_x=round(vel_x, 2),
                    velocity_y=round(vel_y, 2),
                    velocity_z=round(vel_z, 2),
                    source_glasses=combined_glasses,
                    detection_count=num_glasses,
                    last_seen=datetime.utcnow()
                )
                logger.info(f"🔗 Fused multi-glass detection from '{new_obj.source_glasses}' into Global Object [{matched_id}] (Observers: {num_glasses}, Conf: {boosted_conf:.2f})")
            else:
                # Insert as new Global Object
                fused_world[new_obj.object_id] = new_obj

        return fused_world

    def _find_matching_object(self, new_obj: WorldObject, world: Dict[str, WorldObject]) -> str | None:
        """Find matching object ID using spatial proximity and normalized class label."""
        new_clean = self._clean_label(new_obj.label)
        best_id = None
        min_dist = float("inf")

        for obj_id, existing in world.items():
            existing_clean = self._clean_label(existing.label)
            if existing_clean == new_clean:
                dist = euclidean_distance_3d(
                    (existing.position_x, existing.position_y, existing.position_z),
                    (new_obj.position_x, new_obj.position_y, new_obj.position_z)
                )
                if dist <= settings.FUSION_DISTANCE_THRESHOLD and dist < min_dist:
                    min_dist = dist
                    best_id = obj_id

        return best_id

    @staticmethod
    def _clean_label(label_str: str) -> str:
        """Strip track ID suffixes e.g. 'vehicle #1' -> 'vehicle'."""
        return label_str.split(" #")[0].strip().lower()


fusion_engine = PerceptionFusionEngine()

