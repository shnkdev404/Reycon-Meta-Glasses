"""
Automated unit and integration tests for Phase 7: Perception Fusion Engine.
"""
from datetime import datetime, timedelta
from app.models.object import WorldObject
from app.services.fusion_engine import fusion_engine


def test_single_glass_object_insertion():
    print("--- 1. Testing Single Glass Object Insertion ---")
    obj_a = WorldObject(
        object_id="obj_car_1",
        label="vehicle #1",
        confidence=0.85,
        position_x=5.0, position_y=10.0, position_z=0.0,
        source_glasses=["glass_A"],
        detection_count=1
    )
    
    fused_world = fusion_engine.fuse_objects([obj_a], {})
    assert "obj_car_1" in fused_world
    assert fused_world["obj_car_1"].confidence == 0.85
    assert fused_world["obj_car_1"].detection_count == 1
    print("✅ Single Glass Object Insertion passed!")


def test_multi_glass_fusion_and_confidence_boost():
    print("\n--- 2. Testing Multi-Glass Spatial Clustering & Confidence Boosting ---")
    t0 = datetime.utcnow()
    t1 = t0 + timedelta(seconds=0.5)
    
    # Glass A detection
    obj_a = WorldObject(
        object_id="obj_car_1",
        label="vehicle #1",
        confidence=0.85,
        position_x=5.0, position_y=10.0, position_z=0.0,
        source_glasses=["glass_A"],
        detection_count=1,
        last_seen=t0
    )
    initial_world = fusion_engine.fuse_objects([obj_a], {})
    
    # Glass B detection of the SAME physical vehicle at position (5.2, 10.2)
    obj_b = WorldObject(
        object_id="obj_car_2",
        label="vehicle #2", # Different track label, same normalized class 'vehicle'
        confidence=0.90,
        position_x=5.2, position_y=10.2, position_z=0.0,
        source_glasses=["glass_B"],
        detection_count=1,
        last_seen=t1
    )
    
    fused_world = fusion_engine.fuse_objects([obj_b], initial_world)
    
    # Both detections should fuse into obj_car_1 (de-duplicated)
    assert len(fused_world) == 1, f"Expected 1 fused object, got {len(fused_world)}"
    fused_obj = fused_world["obj_car_1"]
    
    # Confidence should boost from 0.90 to 1.0 (0.90 + 0.10)
    assert fused_obj.confidence == 1.0, f"Expected boosted conf 1.0, got {fused_obj.confidence}"
    
    # Source glasses should contain both glass_A and glass_B
    assert "glass_A" in fused_obj.source_glasses
    assert "glass_B" in fused_obj.source_glasses
    assert fused_obj.detection_count == 2
    
    # Spatial centroid check: weighted avg of (5.0, 10.0) with conf 0.85 and (5.2, 10.2) with conf 0.90
    assert 5.05 <= fused_obj.position_x <= 5.15
    assert 10.05 <= fused_obj.position_y <= 10.15
    
    print(f"✅ Multi-Glass Fusion passed! De-duplicated Object: {fused_obj.object_id}, Conf: {fused_obj.confidence}, Observers: {fused_obj.source_glasses}")


def test_velocity_estimation_during_fusion():
    print("\n--- 3. Testing 3D Velocity Estimation During Fusion ---")
    t0 = datetime.utcnow()
    t1 = t0 + timedelta(seconds=1.0)
    
    obj_t0 = WorldObject(
        object_id="obj_forklift_1",
        label="forklift",
        confidence=0.9,
        position_x=0.0, position_y=0.0, position_z=0.0,
        source_glasses=["glass_A"],
        last_seen=t0
    )
    world_t0 = fusion_engine.fuse_objects([obj_t0], {})
    
    # Same forklift moved +2.0m on X over 1 second
    obj_t1 = WorldObject(
        object_id="obj_forklift_2",
        label="forklift",
        confidence=0.9,
        position_x=2.0, position_y=0.0, position_z=0.0,
        source_glasses=["glass_A"],
        last_seen=t1
    )
    world_t1 = fusion_engine.fuse_objects([obj_t1], world_t0)
    
    fused_forklift = world_t1["obj_forklift_1"]
    assert fused_forklift.velocity_x > 1.0, f"Expected velocity_x > 1.0 m/s, got {fused_forklift.velocity_x}"
    print(f"✅ Velocity Estimation passed! Computed Velocity X: {fused_forklift.velocity_x} m/s")


if __name__ == "__main__":
    test_single_glass_object_insertion()
    test_multi_glass_fusion_and_confidence_boost()
    test_velocity_estimation_during_fusion()
    print("\n🎉 ALL PHASE 7 TESTS PASSED SUCCESSFULLY!")
