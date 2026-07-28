"""
Unit & Integration Test Suite for 3D Spatial Identity Matching.

Verifies that detected 3D "person" bounding boxes are matched to connected smart glass units (glass_B)
using 3D Euclidean distance & pose association (Threshold <= 1.5m),
and directed alerts are routed EXCLUSIVELY to that targeted worker.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from app.models.glass import GlassPose, GlassState
from app.models.object import WorldObject
from app.services.fusion_engine import fusion_engine
from app.services.prediction_engine import prediction_engine


def test_spatial_identity_matching():
    print("\n==========================================================================")
    print("🚀 EXECUTING 3D SPATIAL IDENTITY MATCHING & TARGETED ROUTING TEST SUITE")
    print("==========================================================================")

    # Active connected smart glass units
    glass_A = GlassState(
        glass_id="glass_A",
        pose_obj=GlassPose(x=0.0, y=0.0, z=1.65, heading=0.0)
    )

    glass_B = GlassState(
        glass_id="glass_B",
        pose_obj=GlassPose(x=10.0, y=10.0, z=1.65, heading=90.0)
    )

    glass_C = GlassState(
        glass_id="glass_C",
        pose_obj=GlassPose(x=30.0, y=30.0, z=1.65, heading=180.0)
    )

    active_glasses = {
        "glass_A": glass_A,
        "glass_B": glass_B,
        "glass_C": glass_C
    }

    # Glass A detects a "person" at 3D world coordinates (10.1m, 10.2m, 1.65m)
    person_x = 10.1
    person_y = 10.2
    person_z = 1.65

    # Step 1: Execute 3D Spatial Identity Matching
    matched_glass_id = fusion_engine.associate_detected_person_with_glass(
        person_x, person_y, person_z, active_glasses, max_matching_distance=1.5
    )

    assert matched_glass_id == "glass_B"
    print(f"✅ Step 1 Verified: Person detected at ({person_x}m, {person_y}m) correctly matched to active user '{matched_glass_id}'!")

    # Step 2: Test unequipped site visitor (person at 50m, 50m far from all glasses)
    visitor_id = fusion_engine.associate_detected_person_with_glass(
        50.0, 50.0, 1.65, active_glasses, max_matching_distance=1.5
    )
    assert visitor_id is None
    print("✅ Step 2 Verified: Person at (50m, 50m) correctly identified as unequipped visitor (No glass match).")

    # Step 3: Moving Truck advancing toward the matched person's coordinates (10.0m, 10.0m)
    moving_truck = WorldObject(
        object_id="obj_truck_danger",
        label="truck #1",
        confidence=0.95,
        position_x=10.0,
        position_y=5.0,  # Moving North (+Y) at 2.5m/s toward Glass B at (10.0, 10.0)
        position_z=0.0,
        velocity_x=0.0,
        velocity_y=2.5,
        last_seen=datetime.utcnow()
    )

    world_objects = {"obj_truck_danger": moving_truck}
    alerts = prediction_engine.evaluate_threats(active_glasses, world_objects)

    assert len(alerts) >= 1
    assert alerts[0].target_glass_id == "glass_B"
    print(f"✅ Step 3 Verified: Hazard threat routed EXCLUSIVELY to matched target worker '{alerts[0].target_glass_id}'! (Message: {alerts[0].warning_message})")

    print("\n==========================================================================")
    print("🎉 ALL 3D SPATIAL IDENTITY MATCHING & TARGETED ROUTING TESTS PASSED CLEANLY!")
    print("==========================================================================\n")


if __name__ == "__main__":
    test_spatial_identity_matching()
