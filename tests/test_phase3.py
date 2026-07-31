"""
Automated unit and integration tests for Phase 3: Vision & Multi-Object Tracking.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from datetime import datetime, timedelta, timezone
from app.vision.detector import YOLOWrapper
from app.vision.tracker import TrackManager
from app.vision.depth import DepthEstimatorWrapper
from app.services.tracking_manager import TrackingManager
from app.models.object import BoundingBox2D, Detection2D, WorldObject


def test_yolo_detector():
    print("--- 1. Testing YOLOWrapper ---")
    detector = YOLOWrapper()
    
    # Test frame dict input
    frame_sample = {
        "detections": [
            {"class": "forklift", "confidence": 0.95, "distance": 4.5, "bearing": 5.0}
        ]
    }
    dets = detector.detect(frame_sample)
    assert len(dets) >= 1, "Detector failed to parse frame detections"
    assert dets[0].label == "forklift", f"Unexpected label: {dets[0].label}"
    print(f"✅ YOLOWrapper passed! Detections count: {len(dets)}, First label: {dets[0].label}")


def test_track_manager():
    print("\n--- 2. Testing TrackManager MOT Association ---")
    tracker = TrackManager()
    
    # Frame 1
    dets_f1 = [
        Detection2D(label="vehicle", confidence=0.9, bbox=BoundingBox2D(xmin=10, ymin=10, xmax=50, ymax=50), distance=10.0, bearing=0.0)
    ]
    tracked_f1 = tracker.update(dets_f1)
    assert len(tracked_f1) == 1
    track_label_1 = tracked_f1[0].label
    assert "#1" in track_label_1, f"Expected #1 in track label, got {track_label_1}"
    
    # Frame 2 (slightly moved bounding box)
    dets_f2 = [
        Detection2D(label="vehicle", confidence=0.92, bbox=BoundingBox2D(xmin=12, ymin=12, xmax=52, ymax=52), distance=9.8, bearing=0.1)
    ]
    tracked_f2 = tracker.update(dets_f2)
    assert len(tracked_f2) == 1
    track_label_2 = tracked_f2[0].label
    assert "#1" in track_label_2, f"Track ID changed unexpectedly! Got {track_label_2}"
    print(f"✅ TrackManager passed! Track persistent ID: {track_label_2}")


def test_depth_estimator():
    print("\n--- 3. Testing DepthEstimatorWrapper ---")
    estimator = DepthEstimatorWrapper(focal_length_px=800.0)
    
    bbox = BoundingBox2D(xmin=100, ymin=100, xmax=200, ymax=300) # height = 200px
    estimated_d = estimator.estimate_depth(frame=None, bbox=bbox, label="person")
    
    # Expected depth = (800 * 1.7) / 200 = 6.8 meters
    assert 6.0 <= estimated_d <= 7.5, f"Unexpected estimated depth: {estimated_d}"
    print(f"✅ DepthEstimatorWrapper passed! Estimated depth for 200px box: {estimated_d}m")


def test_tracking_manager_velocity():
    print("\n--- 4. Testing TrackingManager 3D Velocity Engine ---")
    tm = TrackingManager()
    
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(seconds=1.0)
    
    obj_t0 = {
        "veh_1": WorldObject(
            object_id="veh_1", label="vehicle", confidence=0.9,
            position_x=0.0, position_y=0.0, position_z=0.0, last_seen=t0
        )
    }
    tm.update_tracks(obj_t0)
    
    obj_t1 = {
        "veh_1": WorldObject(
            object_id="veh_1", label="vehicle", confidence=0.9,
            position_x=3.0, position_y=4.0, position_z=0.0, last_seen=t1
        )
    }
    updated = tm.update_tracks(obj_t1)
    
    veh = updated["veh_1"]
    assert veh.velocity_x > 1.5, f"Velocity X calculation incorrect: {veh.velocity_x}"
    assert veh.velocity_y > 2.0, f"Velocity Y calculation incorrect: {veh.velocity_y}"
    print(f"✅ TrackingManager velocity passed! Velocity: (vx={veh.velocity_x}, vy={veh.velocity_y}, vz={veh.velocity_z}) m/s")


if __name__ == "__main__":
    test_yolo_detector()
    test_track_manager()
    test_depth_estimator()
    test_tracking_manager_velocity()
    print("\n🎉 ALL PHASE 3 TESTS PASSED SUCCESSFULLY!")
