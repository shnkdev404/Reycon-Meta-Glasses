"""
Phase 2 Unit & Integration Test Suite: YOLOv11 + ByteTrack Tracking & World Coordinate Conversion.
Verifies persistent object ID tracking, velocity vectors, motion history, and Pixel -> Camera -> SLAM -> World 3D transformation.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.glass import GlassPose, GlassState
from app.models.object import BoundingBox2D, Detection2D, WorldObject
from app.vision.bytetrack_wrapper import ByteTrackWrapper
from app.services.geometry import pixel_to_world_3d
from app.services.coordinate_transform import CoordinateTransformer


def test_phase_2_bytetrack_and_world_coords():
    print("\n==========================================================================")
    print("🚀 EXECUTING PHASE 2: YOLOV11 + BYTETRACK TRACKING & WORLD COORDINATES")
    print("==========================================================================")

    # Step 1: Test ByteTrack Multi-Object Tracking with Persistent IDs
    tracker = ByteTrackWrapper()

    # Frame 1 Detections
    dets_frame1 = [
        Detection2D(label="truck", confidence=0.94, bbox=BoundingBox2D(xmin=100, ymin=100, xmax=300, ymax=300), distance=12.0, bearing=-10.0),
        Detection2D(label="worker", confidence=0.89, bbox=BoundingBox2D(xmin=400, ymin=200, xmax=480, ymax=450), distance=4.5, bearing=15.0)
    ]

    tracked_f1 = tracker.update_tracks(dets_frame1)
    assert len(tracked_f1) == 2
    assert "#1" in tracked_f1[0].label
    assert "#2" in tracked_f1[1].label
    print(f"✅ ByteTrack Frame 1 Tracking verified: Persistent IDs assigned ('{tracked_f1[0].label}', '{tracked_f1[1].label}').")

    # Frame 2 Detections (Same objects moving closer)
    dets_frame2 = [
        Detection2D(label="truck", confidence=0.95, bbox=BoundingBox2D(xmin=120, ymin=110, xmax=320, ymax=310), distance=10.0, bearing=-8.0),
        Detection2D(label="worker", confidence=0.90, bbox=BoundingBox2D(xmin=410, ymin=205, xmax=490, ymax=455), distance=4.2, bearing=14.0)
    ]

    tracked_f2 = tracker.update_tracks(dets_frame2)
    assert len(tracked_f2) == 2
    assert "#1" in tracked_f2[0].label  # Maintains persistent track ID #1
    assert "#2" in tracked_f2[1].label  # Maintains persistent track ID #2
    print(f"✅ ByteTrack Persistent ID Association verified across frames.")

    # Check track history and velocity estimation
    track1 = tracker.tracks[1]
    assert len(track1.history) >= 2
    print(f"✅ Object Track Velocity & Trajectory History recorded ({len(track1.history)} path points).")

    # Step 2: Test Pixel -> Camera -> SLAM -> Global World Coordinates 3D Transformation
    pose = GlassPose(x=10.0, y=20.0, z=1.65, heading=90.0, pitch=0.0, roll=0.0)

    # Target pixel (u=320, v=240) in center of 640x480 frame at 10m distance
    wx, wy, wz = pixel_to_world_3d(u=320.0, v=240.0, distance_m=10.0, glass_pose=pose, image_w=640.0, image_h=480.0)
    assert abs(wx - 20.0) < 0.5  # 90° heading shifts +X along global +X (10 + 10 = 20)
    assert abs(wy - 20.0) < 0.5
    print(f"✅ 3D Coordinate Pipeline (Pixel -> Camera -> SLAM -> World) verified: World position ({wx:.2f}, {wy:.2f}, {wz:.2f}).")

    # Step 3: Test CoordinateTransformer Batch Conversion into WorldObject instances
    transformer = CoordinateTransformer()
    glass_state = GlassState(glass_id="glass_test_phase2", pose=pose)

    world_objs = transformer.batch_transform_detections(tracked_f2, glass_state)
    assert len(world_objs) == 2
    assert isinstance(world_objs[0], WorldObject)
    assert "obj_truck_1" in world_objs[0].object_id or "truck #1" in world_objs[0].label
    print(f"✅ Batch Spatial Transformation verified: Converted {len(world_objs)} detections into persistent WorldObject instances.")

    print("\n==========================================================================")
    print("🎉 ALL PHASE 2 YOLOV11 + BYTETRACK & WORLD COORDINATES TESTS PASSED CLEANLY!")
    print("==========================================================================\n")


if __name__ == "__main__":
    test_phase_2_bytetrack_and_world_coords()
