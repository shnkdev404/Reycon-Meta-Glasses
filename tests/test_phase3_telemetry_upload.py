"""
Phase 3 Unit & Integration Test Suite: Pose, Local Map, and Tracked Objects Upload Pipeline.
Verifies client telemetry packing and WebSocket central server reception for 6DoF pose, SLAM maps, and persistent tracked objects.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import asyncio
from app.models.glass import GlassPose, GlassState
from app.models.map import LocalMap, KeyFrame, MapPoint
from app.models.object import WorldObject
from app.services.world_manager import world_manager


def test_phase_3_telemetry_upload():
    print("\n==========================================================================")
    print("🚀 EXECUTING PHASE 3: TELEMETRY UPLOAD (POSE, MAP, TRACKED OBJECTS) TESTS")
    print("==========================================================================")

    # Reset world state for clean test setup
    world_manager.reset_world_state()

    # Construct test 6DoF camera pose
    pose = GlassPose(x=12.5, y=34.0, z=1.65, heading=180.0, pitch=2.0, roll=0.0)

    # Construct test Local SLAM Map with KeyFrames and MapPoints
    map_point = MapPoint(point_id="mp_upload_01", x=14.0, y=36.0, z=1.65, observed_count=3)
    keyframe = KeyFrame(keyframe_id="kf_upload_01", glass_id="glass_uploader_01", pose=pose, map_point_ids=["mp_upload_01"])
    local_map = LocalMap(
        glass_id="glass_uploader_01",
        map_id="map_upload_test",
        keyframes={"kf_upload_01": keyframe},
        map_points={"mp_upload_01": map_point}
    )

    # Construct Tracked Objects list with persistent IDs
    tracked_objs = [
        {
            "object_id": "obj_truck_1",
            "label": "truck #1",
            "confidence": 0.96,
            "position": {"x": 14.5, "y": 38.0, "z": 0.0},
            "velocity": {"vx": 1.2, "vy": 0.5, "vz": 0.0}
        },
        {
            "object_id": "obj_worker_2",
            "label": "worker #2",
            "confidence": 0.91,
            "position": {"x": 10.0, "y": 32.0, "z": 0.0},
            "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0}
        }
    ]

    # Package client GlassState telemetry payload
    glass_state = GlassState(
        glass_id="glass_uploader_01",
        pose_obj=pose,
        heading=180.0,
        local_map=local_map.model_dump(),
        tracked_objects=tracked_objs,
        timestamp=time.time()
    )

    # Update world manager state via telemetry processing pipeline
    update_result = world_manager.update_glass(glass_state)

    # Verify server world model reception and storage
    assert "glass_uploader_01" in world_manager.active_glasses
    stored_state = world_manager.active_glasses["glass_uploader_01"]

    assert stored_state.pose.x == 12.5
    assert stored_state.pose.y == 34.0
    assert stored_state.pose.heading == 180.0
    print(f"✅ Camera 6DoF Pose Upload verified: Received pose ({stored_state.pose.x}, {stored_state.pose.y}, heading={stored_state.pose.heading}°).")

    assert stored_state.local_map is not None
    assert "keyframes" in stored_state.local_map
    print(f"✅ Local SLAM Map Upload verified: Map uploaded with {len(stored_state.local_map['keyframes'])} keyframes.")

    assert len(stored_state.tracked_objects) == 2
    assert stored_state.tracked_objects[0]["label"] == "truck #1"
    print(f"✅ Persistent Tracked Objects Upload verified: Received {len(stored_state.tracked_objects)} tracked objects.")

    print("\n==========================================================================")
    print("🎉 ALL PHASE 3 TELEMETRY UPLOAD TESTS PASSED CLEANLY!")
    print("==========================================================================\n")


if __name__ == "__main__":
    test_phase_3_telemetry_upload()
