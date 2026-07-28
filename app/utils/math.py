"""
3D Spatial & Vector Math Utilities.
Provides functions for Euclidean distances, bearing, Time-To-Collision (TTC),
and vector projections.
"""
import math
from typing import Tuple, Optional


def euclidean_distance_3d(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance between two points."""
    return math.sqrt(
        (pos1[0] - pos2[0]) ** 2 +
        (pos1[1] - pos2[1]) ** 2 +
        (pos1[2] - pos2[2]) ** 2
    )


def euclidean_distance_2d(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate 2D Euclidean distance on the ground plane."""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def calculate_bearing(origin: Tuple[float, float], target: Tuple[float, float]) -> float:
    """
    Calculate relative compass bearing in degrees (0 to 360) from origin to target.
    """
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)
    return (angle_deg + 360.0) % 360.0


def calculate_ttc(
    pos_obj: Tuple[float, float, float],
    vel_obj: Tuple[float, float, float],
    pos_target: Tuple[float, float, float],
    vel_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Optional[float]:
    """
    Compute Time-To-Collision (TTC) in seconds between an approaching object and a target point/glass.
    Returns None if object is moving away or parallel.
    """
    r_x = pos_target[0] - pos_obj[0]
    r_y = pos_target[1] - pos_obj[1]
    r_z = pos_target[2] - pos_obj[2]

    v_x = vel_obj[0] - vel_target[0]
    v_y = vel_obj[1] - vel_target[1]
    v_z = vel_obj[2] - vel_target[2]

    # Relative speed along line-of-sight (dot product r . v)
    r_dot_v = r_x * v_x + r_y * v_y + r_z * v_z
    v_squared = v_x ** 2 + v_y ** 2 + v_z ** 2

    # If object is not moving or moving away
    if v_squared < 1e-4 or r_dot_v <= 0:
        return None

    ttc = (r_x ** 2 + r_y ** 2 + r_z ** 2) / r_dot_v
    return max(0.0, ttc)
