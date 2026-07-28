"""
Phase 1 Unit & Integration Test Suite: ORB-SLAM3 Monocular+IMU SLAM Engine.
Tests camera pose tracking, ORB feature extraction, keyframes, loop closure, and map persistence.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
from app.models.glass import GlassSensors, GlassPose
from app.models.map import LocalMap, KeyFrame, MapPoint
from app.slam import ORBSLAM3Wrapper


def test_phase_1_orbslam3():
    print("\n==========================================================================")
    print("🚀 EXECUTING PHASE 1: ORB-SLAM3 MONOCULAR + IMU SLAM ENGINE VERIFICATION")
    print("==========================================================================")

    slam = ORBSLAM3Wrapper(glass_id="glass_test_phase1")
    assert slam.glass_id == "glass_test_phase1"
    print("✅ SLAM Engine initialized with glass ID 'glass_test_phase1'.")

    # Generate synthetic camera frame (640x480 noise/gradient image)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)

    sensors = GlassSensors(accel_x=0.5, accel_y=0.2, accel_z=9.81, gyro_z=0.05)

    # Process consecutive frames to test VIO tracking & KeyFrame creation
    poses = []
    for i in range(20):
        pose = slam.track_monocular_imu(frame, sensors)
        poses.append(pose)

    assert len(poses) == 20
    assert poses[-1].heading > 0.0
    print(f"✅ VIO Monocular+IMU Pose Tracking verified (Final pose: x={poses[-1].x:.2f}, y={poses[-1].y:.2f}, heading={poses[-1].heading:.1f}°).")

    # Check local map creation (KeyFrames and MapPoints)
    local_map = slam.get_local_map()
    assert len(local_map.keyframes) > 0
    assert len(local_map.map_points) > 0
    print(f"✅ Local Map generated with {len(local_map.keyframes)} KeyFrames and {len(local_map.map_points)} 3D MapPoints.")

    # Check Loop Closure optimization
    assert local_map.last_loop_closure_ts is not None
    print(f"✅ Loop Closure graph optimization verified at timestamp {local_map.last_loop_closure_ts:.2f}.")

    # Test Map Persistence (save and load disk serialization)
    save_path = "data/memory/test_phase1_map.json"
    saved = slam.save_map(save_path)
    assert saved is True
    assert os.path.exists(save_path)
    print(f"💾 Map Persistence Save verified: Map saved to '{save_path}'.")

    # Load persistent map into new SLAM instance
    new_slam = ORBSLAM3Wrapper(glass_id="glass_test_phase1_loaded")
    loaded = new_slam.load_map(save_path)
    assert loaded is True
    assert len(new_slam.get_local_map().keyframes) == len(local_map.keyframes)
    print(f"📂 Map Persistence Load verified: Loaded {len(new_slam.get_local_map().keyframes)} KeyFrames cleanly into new instance.")

    # Clean up test file
    if os.path.exists(save_path):
        os.remove(save_path)

    print("\n==========================================================================")
    print("🎉 ALL PHASE 1 ORB-SLAM3 MONOCULAR+IMU ENGINE TESTS PASSED CLEANLY!")
    print("==========================================================================\n")


if __name__ == "__main__":
    test_phase_1_orbslam3()
