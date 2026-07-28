"""
Phase 5: Coordinate Transformation Pipeline.

Converts Pixel Coordinates -> Camera Coordinates -> World Coordinates.
Transforms local 2D camera detections into global 3D WorldObject instances.
"""
import uuid
from datetime import datetime
from app.models.object import Detection2D, WorldObject
from app.models.glass import GlassState
from app.services.geometry import polar_to_cartesian_relative, camera_to_world_2d


import uuid
from typing import List, Dict, Tuple
from datetime import datetime
from app.models.object import Detection2D, WorldObject
from app.models.glass import GlassState, GlassPose
from app.services.geometry import (
    polar_to_cartesian_relative,
    camera_to_world_3d,
    world_to_relative_polar,
    world_to_camera_3d
)


class CoordinateTransformer:
    """Transforms local camera frame detections into Global World Frame objects and vice-versa."""

    def transform_detection_to_world(
        self,
        detection: Detection2D,
        glass_state: GlassState
    ) -> WorldObject:
        """
        Convert a 2D camera detection into a 3D spatial WorldObject using 6DoF glass pose.
        """
        # Step 1: Polar -> Local Cartesian (rel_x, rel_y)
        rel_x, rel_y = polar_to_cartesian_relative(detection.distance, detection.bearing)
        rel_z = 0.0

        # Step 2: 3D Camera -> World Coordinates Transformation
        world_x, world_y, world_z = camera_to_world_3d(
            rel_x=rel_x,
            rel_y=rel_y,
            rel_z=rel_z,
            glass_pose=glass_state.pose
        )

        # Generate unique or track-based object ID
        clean_label = detection.label.split(" #")[0].strip().lower()
        if "#" in detection.label:
            track_num = detection.label.split("#")[1].strip()
            obj_id = f"obj_{clean_label}_{track_num}"
        else:
            obj_id = f"obj_{clean_label}_{uuid.uuid4().hex[:6]}"

        return WorldObject(
            object_id=obj_id,
            label=detection.label,
            confidence=detection.confidence,
            position_x=round(world_x, 2),
            position_y=round(world_y, 2),
            position_z=round(world_z, 2),
            velocity_x=0.0,
            velocity_y=0.0,
            velocity_z=0.0,
            source_glasses=[glass_state.glass_id],
            detection_count=1,
            last_seen=datetime.utcnow()
        )

    def transform_world_to_glass_relative(
        self,
        world_obj: WorldObject,
        glass_pose: GlassPose
    ) -> Dict[str, float]:
        """
        Inverse transformation: Compute distance, relative bearing (-180° to +180°),
        and local (rel_x, rel_y, rel_z) relative to a glass user.
        """
        dist, rel_bearing = world_to_relative_polar(
            world_x=world_obj.position_x,
            world_y=world_obj.position_y,
            glass_pose=glass_pose
        )
        rel_x, rel_y, rel_z = world_to_camera_3d(
            world_x=world_obj.position_x,
            world_y=world_obj.position_y,
            world_z=world_obj.position_z,
            glass_pose=glass_pose
        )

        return {
            "distance": dist,
            "bearing": rel_bearing,
            "rel_x": rel_x,
            "rel_y": rel_y,
            "rel_z": rel_z
        }

    def batch_transform_detections(
        self,
        detections: List[Detection2D],
        glass_state: GlassState
    ) -> List[WorldObject]:
        """Transform a list of 2D detections into 3D WorldObject instances."""
        return [self.transform_detection_to_world(d, glass_state) for d in detections]


coordinate_transformer = CoordinateTransformer()
