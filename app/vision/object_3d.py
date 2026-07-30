"""
Phase 3: 3D Object Detection & 2D-to-3D Back-Projection Lifting.

Lifts 2D bounding boxes into full 3D oriented bounding cuboids with (X, Y, Z) centroids,
3D physical dimensions (Width, Height, Depth), yaw orientation angles, and 8 cuboid corner vertices.
"""
import logging
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("Object3D")


class BoundingBox3D:
    """
    3D Bounding Box Representation (Oriented Cuboid).
    """

    def __init__(
        self,
        center_3d: Tuple[float, float, float],
        size_3d: Tuple[float, float, float],
        yaw_deg: float = 0.0,
        corners_3d: Optional[List[Tuple[float, float, float]]] = None,
        label: str = "object",
        confidence: float = 0.85
    ):
        self.center_3d = center_3d  # (x, y, z) in meters
        self.size_3d = size_3d      # (width, height, depth) in meters
        self.yaw_deg = yaw_deg      # Orientation yaw angle in degrees
        self.label = label
        self.confidence = confidence
        self.corners_3d = corners_3d or self._compute_corners()

    def _compute_corners(self) -> List[Tuple[float, float, float]]:
        """Computes the 8 vertices of the 3D bounding cuboid in metric 3D space."""
        cx, cy, cz = self.center_3d
        w, h, d = self.size_3d
        rad = math.radians(self.yaw_deg)
        cos_y, sin_y = math.cos(rad), math.sin(rad)

        # 8 local offsets
        half_w, half_h, half_d = w / 2.0, h / 2.0, d / 2.0
        local_offsets = [
            (-half_w, -half_h, -half_d),
            (half_w, -half_h, -half_d),
            (half_w, half_h, -half_d),
            (-half_w, half_h, -half_d),
            (-half_w, -half_h, half_d),
            (half_w, -half_h, half_d),
            (half_w, half_h, half_d),
            (-half_w, half_h, half_d),
        ]

        corners = []
        for dx, dy, dz in local_offsets:
            # Rotate around Y-axis (yaw)
            rx = dx * cos_y + dz * sin_y
            ry = dy
            rz = -dx * sin_y + dz * cos_y

            corners.append((
                round(cx + rx, 2),
                round(cy + ry, 2),
                round(cz + rz, 2)
            ))
        return corners

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "center_3d": {
                "x": round(self.center_3d[0], 2),
                "y": round(self.center_3d[1], 2),
                "z": round(self.center_3d[2], 2)
            },
            "size_3d": {
                "width": round(self.size_3d[0], 2),
                "height": round(self.size_3d[1], 2),
                "depth": round(self.size_3d[2], 2)
            },
            "yaw_deg": round(self.yaw_deg, 1),
            "corners_3d": self.corners_3d
        }


def lift_2d_to_3d(
    bbox_2d: list,
    depth_map: Optional[np.ndarray] = None,
    camera_matrix: Optional[np.ndarray] = None,
    label: str = "object",
    confidence: float = 0.85
) -> BoundingBox3D:
    """
    Back-projects a 2D bounding box [x1, y1, x2, y2] into a 3D metric bounding cuboid
    using pinhole camera intrinsics K and RGB-D depth map sampling.
    """
    x1, y1, x2, y2 = float(bbox_2d[0]), float(bbox_2d[1]), float(bbox_2d[2]), float(bbox_2d[3])
    u_center = (x1 + x2) / 2.0
    v_center = (y1 + y2) / 2.0
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    # Intrinsic camera parameters (Ray-Ban Meta Glasses default / calibrated)
    if camera_matrix is not None and camera_matrix.shape == (3, 3):
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
    else:
        fx = 525.0
        fy = 525.0
        cx = 320.0
        cy = 192.0

    # Determine metric Z depth
    z_depth = 4.0  # Default metric depth fallback
    if depth_map is not None and hasattr(depth_map, "shape"):
        try:
            h_img, w_img = depth_map.shape[:2]
            ix1, ix2 = max(0, min(w_img - 1, int(x1))), max(0, min(w_img - 1, int(x2)))
            iy1, iy2 = max(0, min(h_img - 1, int(y1))), max(0, min(h_img - 1, int(y2)))

            if ix2 > ix1 and iy2 > iy1:
                roi = depth_map[iy1:iy2, ix1:ix2]
                valid = roi[roi > 0.1]
                if valid.size > 0:
                    z_depth = float(np.median(valid))
        except Exception as e:
            logger.error(f"Depth sampling error in lift_2d_to_3d: {e}")

    # Pinhole ray back-projection for centroid
    X_center = (u_center - cx) * z_depth / fx
    Y_center = (v_center - cy) * z_depth / fy
    Z_center = z_depth

    # Estimate 3D dimensions (width, height, depth length)
    width_3d = (box_w * z_depth) / fx
    height_3d = (box_h * z_depth) / fy
    depth_3d = max(0.5, min(width_3d, height_3d) * 0.8)  # Metric depth thickness estimation

    # Yaw estimation based on offset from optical center
    yaw_deg = math.degrees(math.atan2(X_center, Z_center))

    return BoundingBox3D(
        center_3d=(X_center, Y_center, Z_center),
        size_3d=(width_3d, height_3d, depth_3d),
        yaw_deg=yaw_deg,
        label=label,
        confidence=confidence
    )


class Object3DDetector:
    """
    3D Object Detection Engine using 2D-to-3D Back-Projection Lifting.
    """

    def __init__(self, camera_matrix: Optional[np.ndarray] = None):
        self.camera_matrix = camera_matrix

    def detect_3d_objects(
        self,
        detections: List[Dict[str, Any]],
        depth_map: Optional[np.ndarray] = None
    ) -> List[BoundingBox3D]:
        """
        Lifts a list of 2D object detections to 3D BoundingBox3D cuboids.
        """
        boxes_3d = []
        for det in detections:
            bbox = det.get("bbox") or det.get("bbox_2d")
            if bbox:
                label = det.get("label", det.get("class_name", "object"))
                conf = float(det.get("confidence", 0.85))
                box3d = lift_2d_to_3d(
                    bbox_2d=bbox,
                    depth_map=depth_map,
                    camera_matrix=self.camera_matrix,
                    label=label,
                    confidence=conf
                )
                boxes_3d.append(box3d)

        return boxes_3d
