"""
Phase 7: Perception Fusion Engine.

Fuses perception telemetry from multiple smart glasses observing the same 3D environment.
Avoids duplicates, clusters spatial observations, boosts confidence for multi-observed objects,
and computes estimated 3D velocity vectors.
"""
from typing import List, Dict
from datetime import datetime, timezone
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
                    last_seen=datetime.now(timezone.utc)
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


    def fuse_local_maps(
        self,
        local_map_dict: Dict,
        global_map_points: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Merge individual local SLAM maps into a unified Global Shared Map graph.
        De-duplicates overlapping 3D landmark points and accumulates observation counts.
        """
        fused_landmarks = dict(global_map_points)
        local_points = local_map_dict.get("map_points", {})

        for pt_id, pt in local_points.items():
            px, py, pz = float(pt.get("x", 0.0)), float(pt.get("y", 0.0)), float(pt.get("z", 0.0))
            matched_id = None
            min_dist = float("inf")

            for g_id, g_pt in fused_landmarks.items():
                gx, gy, gz = float(g_pt["x"]), float(g_pt["y"]), float(g_pt["z"])
                dist = euclidean_distance_3d((px, py, pz), (gx, gy, gz))
                if dist < 0.8 and dist < min_dist:
                    min_dist = dist
                    matched_id = g_id

            if matched_id:
                # Merge observation count
                fused_landmarks[matched_id]["observed_count"] += int(pt.get("observed_count", 1))
            else:
                fused_landmarks[pt_id] = {
                    "point_id": pt_id,
                    "x": px,
                    "y": py,
                    "z": pz,
                    "observed_count": int(pt.get("observed_count", 1))
                }

        return fused_landmarks


    def associate_detected_person_with_glass(
        self,
        person_x: float,
        person_y: float,
        person_z: float,
        active_glasses: Dict,
        max_matching_distance: float = 1.5
    ) -> str | None:
        """
        3D Spatial Identity Association Algorithm.
        Matches a detected 'person' 3D bounding box against active connected smart glasses poses.
        If 3D Euclidean distance <= 1.5m, returns the matched target glass_id.
        """
        best_glass_id = None
        min_dist = float("inf")

        for glass_id, glass in active_glasses.items():
            g_x = glass.pose.x if hasattr(glass, "pose") and glass.pose else getattr(glass.position, "x", 0.0)
            g_y = glass.pose.y if hasattr(glass, "pose") and glass.pose else getattr(glass.position, "y", 0.0)
            g_z = glass.pose.z if hasattr(glass, "pose") and glass.pose else getattr(glass.position, "z", 0.0)

            dist = euclidean_distance_3d((person_x, person_y, person_z), (g_x, g_y, g_z))
            if dist <= max_matching_distance and dist < min_dist:
                min_dist = dist
                best_glass_id = glass_id

        if best_glass_id:
            logger.info(
                f"🎯 3D Spatial Match: Detected person at ({person_x:.1f}m, {person_y:.1f}m) "
                f"matched to active glass unit '{best_glass_id}' (Dist: {min_dist:.2f}m)"
            )

        return best_glass_id


fusion_engine = PerceptionFusionEngine()

