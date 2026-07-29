"""
Automated unit and integration tests for Phase 5: Spatial Geometry & Coordinate Transformation Pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.glass import GlassState, GlassPose
from app.models.object import Detection2D, BoundingBox2D, WorldObject
from app.services.geometry import (
    polar_to_cartesian_relative,
    camera_to_world_2d,
    camera_to_world_3d,
    world_to_camera_3d,
    world_to_relative_polar,
    gps_to_enu,
    pixel_to_camera_ray
)
from app.services.coordinate_transform import coordinate_transformer


def test_2d_and_3d_transformations():
    print("--- 1. Testing 2D and 3D Forward Transformations ---")
    glass_pose = GlassPose(x=10.0, y=20.0, z=1.65, heading=90.0, pitch=0.0, roll=0.0) # Facing East (+X)
    
    # Target 5 meters straight ahead in camera (+Y_cam)
    rel_x, rel_y = polar_to_cartesian_relative(distance=5.0, bearing_deg=0.0)
    assert rel_x == 0.0
    assert rel_y == 5.0
    
    # Transform to world coordinates (Heading = 90° East) -> Forward should be +X in world
    wx, wy = camera_to_world_2d(rel_x, rel_y, glass_pose.x, glass_pose.y, glass_pose.heading)
    assert round(wx, 1) == 15.0, f"Expected wx=15.0, got {wx}"
    assert round(wy, 1) == 20.0, f"Expected wy=20.0, got {wy}"
    
    # 3D Transform
    wx3, wy3, wz3 = camera_to_world_3d(rel_x, rel_y, 0.0, glass_pose)
    assert abs(wx3 - 15.0) < 0.01
    assert abs(wy3 - 20.0) < 0.01
    assert abs(wz3 - 1.65) < 0.01
    print(f"✅ Forward 2D & 3D Transformations passed! World Point: ({wx3}, {wy3}, {wz3})")



def test_inverse_world_to_glass_projection():
    print("\n--- 2. Testing Inverse World-to-Glass Projection ---")
    glass_pose = GlassPose(x=10.0, y=20.0, z=1.65, heading=0.0) # Facing North (+Y)
    
    # World target at (15.0, 20.0) -> 5 meters to the right (+X) of user facing North
    dist, bearing = world_to_relative_polar(world_x=15.0, world_y=20.0, glass_pose=glass_pose)
    assert round(dist, 1) == 5.0, f"Expected distance=5.0, got {dist}"
    assert round(bearing, 1) == 90.0, f"Expected bearing=90.0 (Right), got {bearing}"
    
    # Inverse 3D
    rx, ry, rz = world_to_camera_3d(world_x=15.0, world_y=20.0, world_z=1.65, glass_pose=glass_pose)
    assert round(rx, 1) == 5.0
    assert round(ry, 1) == 0.0
    print(f"✅ Inverse World-to-Glass Projection passed! Relative bearing: {bearing}°, distance: {dist}m")


def test_gps_to_enu():
    print("\n--- 3. Testing WGS84 GPS to Local ENU Conversion ---")
    ref_lat, ref_lon = 37.7749, -122.4194
    # Point ~111 meters North
    target_lat = ref_lat + 0.001
    target_lon = ref_lon
    
    east, north, up = gps_to_enu(target_lat, target_lon, alt=10.0, ref_lat=ref_lat, ref_lon=ref_lon, ref_alt=0.0)
    assert abs(east) < 1.0, f"Expected east near 0, got {east}"
    assert 100.0 <= north <= 120.0, f"Expected north ~111m, got {north}"
    assert up == 10.0
    print(f"✅ GPS to ENU conversion passed! Local ENU: (East={east}m, North={north}m, Up={up}m)")


def test_coordinate_transformer_service():
    print("\n--- 4. Testing CoordinateTransformer Service Pipeline ---")
    glass_state = GlassState(
        glass_id="glass_alpha",
        pose=GlassPose(x=0.0, y=0.0, z=1.65, heading=0.0)
    )
    det = Detection2D(
        label="forklift #2",
        confidence=0.95,
        bbox=BoundingBox2D(xmin=100, ymin=100, xmax=200, ymax=200),
        distance=10.0,
        bearing=45.0 # 45 degrees to the right
    )
    
    world_obj = coordinate_transformer.transform_detection_to_world(det, glass_state)
    assert world_obj.object_id == "obj_forklift_2"
    assert world_obj.position_x > 0.0
    assert world_obj.position_y > 0.0
    
    # Test inverse transformation
    rel_info = coordinate_transformer.transform_world_to_glass_relative(world_obj, glass_state.pose)
    assert round(rel_info["distance"], 1) == 10.0
    assert round(rel_info["bearing"], 1) == 45.0
    print(f"✅ CoordinateTransformer pipeline passed! Fused Object: {world_obj.object_id} at ({world_obj.position_x}, {world_obj.position_y})")


if __name__ == "__main__":
    test_2d_and_3d_transformations()
    test_inverse_world_to_glass_projection()
    test_gps_to_enu()
    test_coordinate_transformer_service()
    print("\n🎉 ALL PHASE 5 TESTS PASSED SUCCESSFULLY!")
