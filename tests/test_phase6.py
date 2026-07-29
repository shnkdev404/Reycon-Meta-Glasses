"""
Automated unit and integration tests for Phase 6: Shared World Model & Perception Orchestrator.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from datetime import datetime, timedelta
from app.models.glass import GlassState, GlassPose
from app.models.object import Detection2D, BoundingBox2D, WorldObject
from app.services.world_manager import world_manager


def test_world_manager_telemetry_pipeline():
    print("--- 1. Testing WorldManager Telemetry Pipeline ---")
    asyncio.run(_run_telemetry_pipeline_test())


async def _run_telemetry_pipeline_test():
    world_manager.reset_world_state()
    
    # Glass A update with vehicle detection
    glass_A = GlassState(
        glass_id="glass_A",
        pose=GlassPose(x=0.0, y=0.0, z=1.65, heading=0.0)
    )
    dets_A = [
        Detection2D(label="vehicle", confidence=0.95, distance=5.0, bearing=0.0)
    ]
    
    threats = await world_manager.update_glass_telemetry(glass_A, dets_A)
    
    # Verify glass registration
    retrieved_glass = world_manager.get_glass("glass_A")
    assert retrieved_glass is not None, "Glass A failed to register in WorldManager"
    assert retrieved_glass.pose.x == 0.0
    
    # Verify trajectory recording
    traj = world_manager.get_glass_trajectory("glass_A")
    assert len(traj) == 1, f"Expected 1 trajectory point, got {len(traj)}"
    
    # Verify world objects creation
    world_objs = world_manager.get_world_objects()
    assert len(world_objs) >= 1, "Failed to create WorldObjects from detections"
    print(f"✅ WorldManager Telemetry Pipeline passed! Active objects count: {len(world_objs)}")


def test_stale_object_pruning():
    print("\n--- 2. Testing Stale Object Pruning ---")
    world_manager.reset_world_state()
    
    import time
    stale_glass = GlassState(
        glass_id="stale_device",
        timestamp=time.time() - 10.0
    )
    world_manager.active_glasses["stale_device"] = stale_glass
    
    assert "stale_device" in world_manager.active_glasses
    
    # Prune objects older than 5 seconds
    world_manager.prune_stale_world_objects(max_age_seconds=5.0)
    
    assert "stale_device" not in world_manager.active_glasses, "Stale object failed to prune"
    print("✅ Stale Object Pruning passed! Expired object correctly removed.")


def test_full_world_state_serialization():
    print("\n--- 3. Testing Full World State Serialization ---")
    asyncio.run(_run_serialization_test())


async def _run_serialization_test():
    glass = GlassState(glass_id="glass_test", pose=GlassPose(x=1.0, y=2.0, heading=45.0))
    await world_manager.update_glass_telemetry(glass, [])
    
    state_json = await world_manager.get_full_world_state()
    assert state_json["active_glasses_count"] >= 1
    assert "glass_test" in state_json["glasses"]
    assert "timestamp" in state_json
    print(f"✅ World State Serialization passed! Active glasses: {state_json['active_glasses_count']}")


if __name__ == "__main__":
    test_world_manager_telemetry_pipeline()
    test_stale_object_pruning()
    test_full_world_state_serialization()
    print("\n🎉 ALL PHASE 6 TESTS PASSED SUCCESSFULLY!")
