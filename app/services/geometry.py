import math
from typing import Tuple, Dict, Any, Optional
from app.models.glass import GlassPose


def heading_to_vector_2d(heading_deg: float) -> Tuple[float, float]:
    """
    Convert compass heading angle in degrees (0 = North/Y+, 90 = East/X+)
    to a normalized 2D direction unit vector (dx, dy).
    """
    rad = math.radians(heading_deg)
    dx = math.sin(rad)
    dy = math.cos(rad)
    return dx, dy


def polar_to_cartesian_relative(distance: float, bearing_deg: float) -> Tuple[float, float]:
    """
    Convert relative distance and relative bearing (in degrees) to local camera (x, y) offset.
    bearing = 0 -> Straight ahead (+Y)
    bearing = 90 -> Right (+X)
    bearing = -90 -> Left (-X)
    """
    rad = math.radians(bearing_deg)
    rel_x = distance * math.sin(rad)
    rel_y = distance * math.cos(rad)
    return rel_x, rel_y


def camera_to_world_2d(
    rel_x: float,
    rel_y: float,
    glass_x: float,
    glass_y: float,
    glass_heading_deg: float
) -> Tuple[float, float]:
    """
    Transform relative camera offset (rel_x, rel_y) to Global World Coordinates (world_x, world_y)
    using the smart glass origin position and compass heading orientation.
    """
    rad = math.radians(glass_heading_deg)
    cos_h = math.cos(rad)
    sin_h = math.sin(rad)

    # 2D Rotation by heading + translation
    world_x = glass_x + (rel_x * cos_h + rel_y * sin_h)
    world_y = glass_y + (-rel_x * sin_h + rel_y * cos_h)

    return world_x, world_y


def camera_to_world_3d(
    rel_x: float,
    rel_y: float,
    rel_z: float,
    glass_pose: GlassPose
) -> Tuple[float, float, float]:
    """
    Transform local camera 3D relative offset (rel_x, rel_y, rel_z) to Global World 3D Coordinates
    incorporating 6DoF heading, pitch, and roll.
    """
    rad_h = math.radians(glass_pose.heading)
    rad_p = math.radians(glass_pose.pitch)
    rad_r = math.radians(glass_pose.roll)

    # Rotation matrix components for Yaw (Heading), Pitch, and Roll
    cos_h, sin_h = math.cos(rad_h), math.sin(rad_h)
    cos_p, sin_p = math.cos(rad_p), math.sin(rad_p)

    # Apply 3D orientation rotation and translation
    rot_x = rel_x * cos_h + rel_y * sin_h
    rot_y = (-rel_x * sin_h + rel_y * cos_h) * cos_p - rel_z * sin_p
    rot_z = (-rel_x * sin_h + rel_y * cos_h) * sin_p + rel_z * cos_p

    world_x = glass_pose.x + rot_x
    world_y = glass_pose.y + rot_y
    world_z = glass_pose.z + rot_z

    return round(world_x, 3), round(world_y, 3), round(world_z, 3)


def world_to_camera_3d(
    world_x: float,
    world_y: float,
    world_z: float,
    glass_pose: GlassPose
) -> Tuple[float, float, float]:
    """
    Inverse transformation: Convert Global World 3D coordinates into camera-relative 3D coordinates.
    """
    dx = world_x - glass_pose.x
    dy = world_y - glass_pose.y
    dz = world_z - glass_pose.z

    rad_h = math.radians(-glass_pose.heading)
    cos_h, sin_h = math.cos(rad_h), math.sin(rad_h)

    rel_x = dx * cos_h - dy * sin_h
    rel_y = dx * sin_h + dy * cos_h
    rel_z = dz

    return round(rel_x, 3), round(rel_y, 3), round(rel_z, 3)


def world_to_relative_polar(
    world_x: float,
    world_y: float,
    glass_pose: GlassPose
) -> Tuple[float, float]:
    """
    Calculate Euclidean distance and relative bearing (-180° to +180°) from glass user to world target.
    Relative bearing: 0° = Dead ahead, +90° = Right, -90° = Left, +/-180° = Directly behind.
    """
    dx = world_x - glass_pose.x
    dy = world_y - glass_pose.y
    distance = math.sqrt(dx ** 2 + dy ** 2)

    # World compass bearing
    target_heading_deg = math.degrees(math.atan2(dx, dy))
    
    # Relative bearing angle wrt glass compass heading
    relative_bearing = target_heading_deg - glass_pose.heading
    # Wrap to [-180, 180]
    relative_bearing = (relative_bearing + 180.0) % 360.0 - 180.0

    return round(distance, 2), round(relative_bearing, 1)


def gps_to_enu(
    lat: float,
    lon: float,
    alt: float,
    ref_lat: float = 37.7749,
    ref_lon: float = -122.4194,
    ref_alt: float = 0.0
) -> Tuple[float, float, float]:
    """
    Convert WGS84 Geodetic GPS coordinates (Latitude, Longitude, Altitude) to
    Local East-North-Up (ENU) metric Cartesian coordinates relative to a reference origin.
    """
    R_EARTH = 6378137.0  # WGS84 Earth equatorial radius in meters
    rad_ref_lat = math.radians(ref_lat)

    d_lat = math.radians(lat - ref_lat)
    d_lon = math.radians(lon - ref_lon)

    north = d_lat * R_EARTH
    east = d_lon * R_EARTH * math.cos(rad_ref_lat)
    up = alt - ref_alt

    return round(east, 2), round(north, 2), round(up, 2)


def pixel_to_camera_ray(
    u: float,
    v: float,
    image_w: float = 1920.0,
    image_h: float = 1080.0,
    hfov_deg: float = 90.0
) -> Tuple[float, float, float]:
    """
    Compute normalized 3D direction vector (rx, ry, rz) in camera frame for pixel coordinate (u, v).
    """
    aspect_ratio = image_w / image_h
    hfov_rad = math.radians(hfov_deg)
    focal_length = (image_w / 2.0) / math.tan(hfov_rad / 2.0)

    # Normalized camera plane coordinates
    x_c = (u - image_w / 2.0) / focal_length
    y_c = (v - image_h / 2.0) / focal_length
    z_c = 1.0

    norm = math.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2)
    return round(x_c / norm, 4), round(z_c / norm, 4), round(-y_c / norm, 4)

