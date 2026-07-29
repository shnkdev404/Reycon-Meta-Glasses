import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision.slam import SLAMManager
from app.models.glass import GlassPose, GlassSensors


def test_slam_initialization():
    print("--- 1. Testing SLAMManager Initialization ---")
    slam = SLAMManager()
    pose = slam.get_pose()
    assert pose.x == 0.0, f"Expected x=0.0, got {pose.x}"
    assert pose.y == 0.0, f"Expected y=0.0, got {pose.y}"
    assert pose.z == 1.65, f"Expected z=1.65, got {pose.z}"
    print(f"✅ SLAMManager initialized cleanly with default pose: {pose}")


def test_imu_dead_reckoning():
    print("\n--- 2. Testing IMU Kinematic Dead-Reckoning ---")
    slam = SLAMManager()
    
    # Simulate IMU forward acceleration (accel_x = 2.0 m/s^2, gyro_z = 0.1 rad/s)
    imu_data = GlassSensors(
        accel_x=2.0, accel_y=0.0, accel_z=9.81,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.1
    )
    
    # Step forward over 10 ticks (0.1s each = 1s total)
    for _ in range(10):
        slam.track_pose(imu_data=imu_data, dt=0.1)
        
    updated_pose = slam.get_pose()
    assert updated_pose.x > 0.0 or updated_pose.y > 0.0, "IMU dead reckoning failed to advance position"
    assert updated_pose.heading > 0.0, "Gyro integration failed to update heading"
    print(f"✅ IMU Kinematic Dead Reckoning passed! Updated Pose: {updated_pose}")


def test_visual_odometry_fusion():
    print("\n--- 3. Testing Visual Odometry Fusion ---")
    slam = SLAMManager()
    
    # Inject visual odometry delta frame
    frame = {
        "visual_odometry": {
            "dx": 2.5,
            "dy": 1.0,
            "dheading": 15.0
        }
    }
    updated_pose = slam.track_pose(frame=frame, dt=0.1)
    
    assert updated_pose.x >= 1.0, f"Visual odometry failed to update X position: {updated_pose.x}"
    assert updated_pose.y >= 0.5, f"Visual odometry failed to update Y position: {updated_pose.y}"
    print(f"✅ Visual Odometry Fusion passed! Pose after EKF fusion: {updated_pose}")


def test_origin_reset():
    print("\n--- 4. Testing Spatial Origin Reset ---")
    slam = SLAMManager()
    slam.track_pose(frame={"visual_odometry": {"dx": 10.0, "dy": 5.0}}, dt=0.1)
    
    # Reset origin anchor to (100, 200, 1.7) heading=90
    slam.reset_origin(x=100.0, y=200.0, z=1.7, heading=90.0)
    reset_pose = slam.get_pose()
    
    assert reset_pose.x == 100.0, f"Origin reset X failed: {reset_pose.x}"
    assert reset_pose.y == 200.0, f"Origin reset Y failed: {reset_pose.y}"
    assert reset_pose.heading == 90.0, f"Origin reset heading failed: {reset_pose.heading}"
    print(f"✅ Spatial Origin Reset passed! Reset pose: {reset_pose}")


if __name__ == "__main__":
    test_slam_initialization()
    test_imu_dead_reckoning()
    test_visual_odometry_fusion()
    test_origin_reset()
    print("\n🎉 ALL PHASE 4 TESTS PASSED SUCCESSFULLY!")
