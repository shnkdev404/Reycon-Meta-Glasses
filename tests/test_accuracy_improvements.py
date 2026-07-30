"""
Unit tests for Vision Accuracy & Distance Estimation Improvements:
1. Issue 2.1: 0.5+ Confidence & WBF Refinement
2. Issue 2.2: RGB-D Depth Map Sampling & Calibrated Focal Length
3. Issue 2.3: 3D Kalman Filtering & Temporal Position Smoothing
4. Issue 2.4: Weighted Box Fusion (WBF)
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models import Detection, Position, BoundingBox2D
from app.services.detector import (
    DetectionEngine, estimate_object_distance,
    apply_weighted_box_fusion, KalmanFilter3D, TemporalTracker
)
from app.vision.depth import DepthEstimatorWrapper


def test_rgbd_depth_sampling_and_calibration():
    print("--- 1. Testing RGB-D Depth Sampling & Calibrated Geometry ---")
    # Synthetic depth map (640x480) with 3.5m distance in region (100, 100) to (200, 200)
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[100:200, 100:200] = 3.52

    bbox = [100, 100, 200, 200]
    sampled_dist = estimate_object_distance("car", box_height_px=100, frame_height_px=480, depth_map=depth_map, bbox=bbox)
    assert abs(sampled_dist - 3.52) < 0.05, f"Expected ~3.52m from RGB-D depth map, got {sampled_dist}m"

    # Calibrated focal length test
    calibrated_dist = estimate_object_distance("person", box_height_px=200, frame_height_px=480, focal_length_px=800.0)
    # Expected: (800 * 1.7) / 200 = 6.8m
    assert abs(calibrated_dist - 6.8) < 0.1, f"Expected 6.8m with f_px=800.0, got {calibrated_dist}m"
    print(f"✅ RGB-D depth map sampling & camera calibration passed! Sampled depth: {sampled_dist}m, Calibrated depth: {calibrated_dist}m")


def test_3d_kalman_filtering_temporal_consistency():
    print("\n--- 2. Testing 3D Kalman Filter & Temporal Consistency ---")
    kf = KalmanFilter3D(x=0.0, y=10.0, z=0.0)
    
    # Observe sequence with measurement noise (10.2m, 10.4m, 10.1m, 10.3m)
    measurements = [(0.1, 10.2, 0.0), (0.2, 10.4, 0.0), (0.15, 10.1, 0.0), (0.25, 10.3, 0.0)]
    for m in measurements:
        kf.predict(dt=0.1)
        kf.update(m)

    smoothed_pos = kf.position
    assert 9.8 <= smoothed_pos[1] <= 10.5, f"Expected smoothed position ~10.2m, got {smoothed_pos}"
    
    # Test TemporalTracker integration
    tracker = TemporalTracker()
    det1 = Detection(class_name="truck", label="truck", confidence=0.9, position=Position(x=1.0, y=12.0, z=0.0), direction="FRONT", bbox=[10, 10, 50, 50])
    smoothed_dets = tracker.update_and_smooth([det1])
    assert len(smoothed_dets) == 1
    print(f"✅ 3D Kalman Filter temporal consistency passed! Smoothed position: {smoothed_pos}")


def test_weighted_box_fusion():
    print("\n--- 3. Testing Weighted Box Fusion (WBF) ---")
    dets = [
        Detection(class_name="forklift", label="forklift", confidence=0.90, position=Position(x=2.0, y=5.0, z=0.0), direction="FRONT", bbox=[100, 100, 200, 200]),
        Detection(class_name="forklift", label="forklift", confidence=0.80, position=Position(x=2.2, y=5.2, z=0.0), direction="FRONT", bbox=[105, 105, 205, 205])
    ]

    fused = apply_weighted_box_fusion(dets, iou_threshold=0.5, confidence_threshold=0.5)
    assert len(fused) == 1, f"Expected 1 fused detection, got {len(fused)}"
    assert fused[0].confidence > 0.85
    print(f"✅ Weighted Box Fusion (WBF) passed! Fused detection position: ({fused[0].position.x}, {fused[0].position.y})")


if __name__ == "__main__":
    test_rgbd_depth_sampling_and_calibration()
    test_3d_kalman_filtering_temporal_consistency()
    test_weighted_box_fusion()
    print("\n🎉 ALL ACCURACY IMPROVEMENT TESTS PASSED SUCCESSFULLY!")
