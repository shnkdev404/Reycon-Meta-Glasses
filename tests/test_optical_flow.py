"""
Unit tests for Dense Optical Flow Motion Engine:
1. OpticalFlowEngine.compute_flow() Farneback motion calculation.
2. Bounding box region motion extraction (compute_region_motion).
3. Fast-moving threat detection (detect_fast_moving_threats).
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision import OpticalFlowEngine


def test_farneback_optical_flow_calculation():
    print("--- 1. Testing Farneback Optical Flow Calculation ---")
    flow_engine = OpticalFlowEngine()

    # Frame 1: Solid background with white square at (100, 100)
    frame1 = np.zeros((384, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (100, 100), (200, 200), (255, 255, 255), -1)

    # Frame 2: White square shifted to (120, 100) - horizontal motion rightwards
    frame2 = np.zeros((384, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (120, 100), (220, 200), (255, 255, 255), -1)

    # Frame 1 pass (initializes previous gray frame)
    flow1, mag1, mean_m1 = flow_engine.compute_flow(frame1)
    assert mean_m1 == 0.0, "First frame should return zero motion"

    # Frame 2 pass (computes optical flow against Frame 1)
    flow2, mag2, mean_m2 = flow_engine.compute_flow(frame2)
    assert flow2 is not None, "Optical flow vectors must not be None!"
    assert mag2 is not None, "Motion magnitude array must not be None!"
    assert mean_m2 > 0.0, f"Expected non-zero motion magnitude between moving frames, got {mean_m2}"

    print(f"✅ Farneback dense optical flow passed! Mean motion magnitude between frames: {mean_m2:.3f} px/frame")


def test_region_motion_and_fast_threat_detection():
    print("\n--- 2. Testing Region Motion & Fast Threat Detection ---")
    flow_engine = OpticalFlowEngine()

    frame1 = np.zeros((384, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (200, 150), (350, 300), (255, 255, 255), -1)

    frame2 = np.zeros((384, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (250, 150), (400, 300), (255, 255, 255), -1)

    # Prime frame 1
    flow_engine.compute_flow(frame1)

    # Evaluate fast threat detection on frame 2
    bboxes = [[200, 150, 350, 300]]
    threats = flow_engine.detect_fast_moving_threats(frame2, bboxes, motion_threshold=1.5)

    assert len(threats) >= 1, "Expected fast-moving threat alert for dynamic bounding box region!"
    first_threat = threats[0]
    assert first_threat["is_fast_threat"] is True
    assert first_threat["motion_magnitude"] >= 1.5

    print(f"✅ Region motion & fast threat detection passed! Detected threat motion: {first_threat['motion_magnitude']:.2f} px/frame")


if __name__ == "__main__":
    test_farneback_optical_flow_calculation()
    test_region_motion_and_fast_threat_detection()
    print("\n🎉 ALL OPTICAL FLOW TESTS PASSED SUCCESSFULLY!")
