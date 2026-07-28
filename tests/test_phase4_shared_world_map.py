"""
Phase 4 Unit & Integration Test Suite: Shared World Map & Multi-Glass Observation Merging.
Verifies multi-glass perception fusion, landmark graph merging, confidence boosting, and central shared world building.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timedelta
from app.models.object import WorldObject
from app.models.glass import GlassPose, GlassState
from app.models.map import LocalMap, KeyFrame, MapPoint
from app.services.fusion_engine import fusion_engine
from app.services.world_manager import world_manager


def test_phase_4_shared_world_map():
    print("\n==========================================================================")
    print("🚀 EXECUTING PHASE 4: SHARED WORLD MAP & MULTI-GLASS OBSERVATION FUSION TESTS")
    print("==========================================================================")

    # Step 1: Simulate Multi-Glass Observations of the Same Object
    t0 = datetime.utcnow()
    t1 = t0 + timedelta(seconds=0.5)

    # Glass A sees an excavator at (15.0, 20.0, 0.0)
    obj_glass_a = WorldObject(
        object_id="obj_excavator_1",
        label="excavator #1",
        confidence=0.88,
        position_x=15.0,
        position_y=20.0,
        position_z=0.0,
        source_glasses=["glass_A"],
        last_seen=t0
    )

    # Glass B sees the SAME excavator slightly offset at (15.2, 20.1, 0.0)
    obj_glass_b = WorldObject(
        object_id="obj_excavator_2",
        label="excavator #1",
        confidence=0.92,
        position_x=15.2,
        position_y=20.1,
        position_z=0.0,
        source_glasses=["glass_B"],
        last_seen=t1
    )

    # Fuse observations into initial empty world
    world_step1 = fusion_engine.fuse_objects([obj_glass_a], {})
    assert len(world_step1) == 1
    assert world_step1["obj_excavator_1"].confidence == 0.88
    print("✅ Step 1: Initial observation from Glass A added to Shared World Map.")

    # Fuse second observation from Glass B
    world_step2 = fusion_engine.fuse_objects([obj_glass_b], world_step1)
    assert len(world_step2) == 1  # De-duplicated into 1 global object!
    fused_obj = world_step2["obj_excavator_1"]

    assert len(fused_obj.source_glasses) == 2
    assert "glass_A" in fused_obj.source_glasses and "glass_B" in fused_obj.source_glasses
    assert fused_obj.confidence == 1.0  # Boosted confidence!
    print(f"✅ Step 2: Multi-Glass Observation Merged! Observers: {fused_obj.source_glasses}, Boosted Confidence: {fused_obj.confidence:.2f}, Centroid Position: ({fused_obj.position_x}, {fused_obj.position_y}).")

    # Step 2: Test SLAM Landmark Map Graph Merging
    local_map_a = {
        "map_points": {
            "mp_1": {"x": 5.0, "y": 10.0, "z": 0.0, "observed_count": 1},
            "mp_2": {"x": 20.0, "y": 30.0, "z": 0.0, "observed_count": 2}
        }
    }
    local_map_b = {
        "map_points": {
            "mp_1_dup": {"x": 5.1, "y": 10.1, "z": 0.0, "observed_count": 1},  # Overlaps with mp_1
            "mp_3": {"x": 40.0, "y": 50.0, "z": 0.0, "observed_count": 1}
        }
    }

    fused_landmarks = fusion_engine.fuse_local_maps(local_map_a, {})
    fused_landmarks = fusion_engine.fuse_local_maps(local_map_b, fused_landmarks)

    assert len(fused_landmarks) == 3  # mp_1 and mp_1_dup merged!
    assert fused_landmarks["mp_1"]["observed_count"] == 2
    print(f"✅ Step 3: Local SLAM Map Graph Fusion verified ({len(fused_landmarks)} global landmark points merged).")

    print("\n==========================================================================")
    print("🎉 ALL PHASE 4 SHARED WORLD MAP & OBSERVATION FUSION TESTS PASSED CLEANLY!")
    print("==========================================================================\n")


if __name__ == "__main__":
    test_phase_4_shared_world_map()
