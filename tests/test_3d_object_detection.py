"""
Unit tests for 3D Object Detection & 2D-to-3D Back-Projection Lifting:
1. lift_2d_to_3d back-projection ray reconstruction.
2. BoundingBox3D 8-corner 3D cuboid vertex generation.
3. Object3DDetector 2D-to-3D batch lifting.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision import BoundingBox3D, Object3DDetector, lift_2d_to_3d


def test_lift_2d_to_3d_backprojection():
    print("--- 1. Testing lift_2d_to_3d Back-Projection ---")
    # Synthetic depth map (384x640) with 3.5m depth in center region
    depth_map = np.full((384, 640), 3.5, dtype=np.float32)

    bbox_2d = [270.0, 142.0, 370.0, 242.0]  # Center ~ (320, 192)
    box3d = lift_2d_to_3d(bbox_2d, depth_map=depth_map, label="car", confidence=0.92)

    assert isinstance(box3d, BoundingBox3D)
    assert box3d.label == "car"
    assert abs(box3d.center_3d[2] - 3.5) < 0.01, f"Expected 3.5m depth Z, got {box3d.center_3d[2]}"
    assert len(box3d.corners_3d) == 8, f"Expected 8 cuboid corner vertices, got {len(box3d.corners_3d)}"

    print(f"✅ 2D-to-3D lifting passed! 3D Center: {box3d.center_3d}, 3D Dimensions: {box3d.size_3d}")


def test_bounding_box_3d_cuboid_corners():
    print("\n--- 2. Testing BoundingBox3D 8-Corner Cuboid Generation ---")
    center = (1.5, 0.5, 4.0)
    size = (2.0, 1.5, 4.5)  # (w, h, d)
    yaw = 15.0

    box3d = BoundingBox3D(center_3d=center, size_3d=size, yaw_deg=yaw, label="forklift")
    corners = box3d.corners_3d

    assert len(corners) == 8, "Must generate exactly 8 3D vertices!"
    for idx, c in enumerate(corners):
        assert len(c) == 3, f"Corner {idx} must be a 3D tuple (x, y, z)"

    d = box3d.to_dict()
    assert "center_3d" in d
    assert "size_3d" in d
    assert "yaw_deg" in d
    assert "corners_3d" in d
    print(f"✅ BoundingBox3D cuboid generation passed! 8 vertices computed around yaw {yaw}°.")


def test_object_3d_detector_batch_lifting():
    print("\n--- 3. Testing Object3DDetector Batch Lifting ---")
    detector = Object3DDetector()
    depth_map = np.full((384, 640), 5.0, dtype=np.float32)

    dets_2d = [
        {"label": "pedestrian", "confidence": 0.88, "bbox": [100, 100, 200, 300]},
        {"label": "truck", "confidence": 0.95, "bbox": [400, 150, 600, 350]},
    ]

    boxes_3d = detector.detect_3d_objects(dets_2d, depth_map=depth_map)
    assert len(boxes_3d) == 2, f"Expected 2 3D boxes, got {len(boxes_3d)}"
    assert boxes_3d[0].label == "pedestrian"
    assert boxes_3d[1].label == "truck"
    print(f"✅ Object3DDetector batch lifting passed! Successfully lifted {len(boxes_3d)} 2D detections to 3D.")


if __name__ == "__main__":
    test_lift_2d_to_3d_backprojection()
    test_bounding_box_3d_cuboid_corners()
    test_object_3d_detector_batch_lifting()
    print("\n🎉 ALL 3D OBJECT DETECTION TESTS PASSED SUCCESSFULLY!")
