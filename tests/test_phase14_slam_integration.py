"""
Phase 14: Real-Time SLAM Integration Unit & Integration Tests.

Verifies:
1. ORBSLAMWrapper("ORBvoc.txt", "camera.yaml") initialization.
2. slam.track_mono(gray_frame, timestamp) processing monocular frames & updating global coordinate frame.
3. poses = slam.get_all_poses() retrieving 6DoF trajectory history.
4. Loop Closure Detection and Relocalization.
5. SLAMManager integration.
"""
import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.glass import GlassPose, GlassSensors
from app.slam.orbslam3_wrapper import ORBSLAMWrapper, ORBSLAM3Wrapper
from app.vision.slam import SLAMManager


def test_orbslam_wrapper_pattern_matching_prompt():
    print("--- 1. Testing ORBSLAMWrapper Pattern Matching Prompt Specifications ---")
    
    # Initialize matching prompt: slam = ORBSLAMWrapper("ORBvoc.txt", "camera.yaml")
    slam = ORBSLAMWrapper("ORBvoc.txt", "camera.yaml")
    assert slam is not None
    assert slam.vocab_path == "ORBvoc.txt"
    assert slam.config_path == "camera.yaml"
    print("✅ ORBSLAMWrapper initialized with 'ORBvoc.txt' and 'camera.yaml'.")

    # Generate synthetic monocular gray frame
    gray_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    cv2.circle(gray_frame, (320, 240), 40, 255, -1)
    
    # Process frames matching prompt: slam.track_mono(gray_frame, timestamp)
    start_time = time.time()
    for i in range(15):
        ts = start_time + i * 0.033
        pose = slam.track_mono(gray_frame, timestamp=ts)
        assert isinstance(pose, GlassPose)

    # Retrieve all poses matching prompt: poses = slam.get_all_poses()
    poses = slam.get_all_poses()
    assert len(poses) >= 15
    assert isinstance(poses[0], GlassPose)
    
    print(f"✅ Real-Time SLAM track_mono & get_all_poses passed! Total poses recorded in global coordinate frame: {len(poses)}")


def test_loop_closure_and_relocalization():
    print("\n--- 2. Testing Loop Closure Detection & Relocalization ---")
    slam = ORBSLAMWrapper("ORBvoc.txt", "camera.yaml")
    
    frame_a = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame_a, (100, 100), (300, 300), (255, 255, 255), -1)

    # Track multiple frames to create KeyFrames & MapPoints
    for i in range(25):
        slam.track_mono(frame_a, timestamp=time.time() + i * 0.05)

    local_map = slam.get_local_map()
    assert len(local_map.keyframes) > 0
    assert len(local_map.map_points) > 0
    assert slam.loop_closure_count > 0

    # Perform Relocalization
    reloc_res = slam.relocalize(frame_a)
    assert reloc_res["success"] is True
    assert reloc_res["relocalization_count"] == 1

    print(f"✅ Loop Closure & Relocalization passed! KeyFrames: {len(local_map.keyframes)}, Loop Closures: {slam.loop_closure_count}, Relocalization Success: {reloc_res['success']}")


def test_slam_manager_integration():
    print("\n--- 3. Testing SLAMManager Backend Integration ---")
    manager = SLAMManager(backend="ORBSLAM3", vocab_path="ORBvoc.txt", config_path="camera.yaml")
    assert manager._slam_engine is not None

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    sensors = GlassSensors(accel_x=0.1, accel_y=0.0, accel_z=9.81, gyro_z=0.02)

    pose = manager.track_pose(frame=frame, imu_data=sensors, dt=0.033)
    assert isinstance(pose, GlassPose)

    all_poses = manager.get_all_poses()
    assert len(all_poses) > 0

    reloc = manager.relocalize(frame)
    assert "success" in reloc

    print(f"✅ SLAMManager Integration passed! Pose: ({pose.x:.2f}, {pose.y:.2f}, heading={pose.heading:.1f}°)")


if __name__ == "__main__":
    test_orbslam_wrapper_pattern_matching_prompt()
    test_loop_closure_and_relocalization()
    test_slam_manager_integration()
    print("\n🎉 ALL PHASE 14 REAL-TIME SLAM INTEGRATION TESTS PASSED SUCCESSFULLY!")
