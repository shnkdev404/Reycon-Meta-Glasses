"""
Phase 6 Unit & Integration Test Suite: Collaborative Threat Detection & Directed Non-Broadcast Alert Router.
Verifies symmetric protection where Glass A protects Glass B (and vice-versa) with zero broadcast noise.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from datetime import datetime
from app.models.glass import GlassPose, GlassState
from app.models.object import WorldObject
from app.models.threat import ThreatAlert, ThreatLevel, ThreatType
from app.services.prediction_engine import prediction_engine
from app.services.alert_engine import alert_engine
from app.services.world_manager import world_manager


def test_phase_6_collaborative_threats():
    print("\n==========================================================================")
    print("🚀 EXECUTING PHASE 6: COLLABORATIVE THREAT DETECTION & DIRECTED ALERT ROUTING")
    print("==========================================================================")

    asyncio.run(_async_phase6_test())


async def _async_phase6_test():
    alert_engine.clear_alert_history()
    world_manager.reset_world_state()

    # Define connected workers
    glass_A = GlassState(
        glass_id="glass_A",
        pose_obj=GlassPose(x=0.0, y=0.0, z=1.65, heading=0.0)
    )

    glass_B = GlassState(
        glass_id="glass_B",
        pose_obj=GlassPose(x=20.0, y=0.0, z=1.65, heading=90.0)
    )

    world_manager.active_glasses["glass_A"] = glass_A
    world_manager.active_glasses["glass_B"] = glass_B

    # SCENARIO 1: Glass A observes a truck moving toward Glass B
    # Truck at (25.0, 0.0) moving left (-2.5 m/s) toward Glass B at (20.0, 0.0)
    truck_observed_by_A = WorldObject(
        object_id="obj_truck_A",
        label="truck #1",
        confidence=0.95,
        position_x=25.0,
        position_y=0.0,
        position_z=0.0,
        velocity_x=-2.5,  # Moving left toward Glass B
        velocity_y=0.0,
        velocity_z=0.0,
        source_glasses=["glass_A"],
        last_seen=datetime.utcnow()
    )

    # Evaluate threats centrally on server
    world_objects_s1 = {"obj_truck_A": truck_observed_by_A}
    active_threats_s1 = prediction_engine.evaluate_threats(world_manager.active_glasses, world_objects_s1)

    # Dispatch alerts using Non-Broadcast Directed Router
    dispatched_s1 = await alert_engine.dispatch_alerts(active_threats_s1)

    # Audit alerts received by Glass A vs Glass B
    alerts_for_B = alert_engine.get_alert_history("glass_B")
    alerts_for_A = alert_engine.get_alert_history("glass_A")

    assert len(alerts_for_B) >= 1
    assert len(alerts_for_A) == 0  # Glass A received NOTHING!
    assert alerts_for_B[0]["target_glass_id"] == "glass_B"
    print("✅ Scenario 1 Verified: Glass A observes truck -> ONLY Glass B receives directed alert! Glass A receives 0 alert noise.")

    # SCENARIO 2: Later, Glass B observes an excavator moving toward Glass A
    alert_engine.clear_alert_history()

    # Excavator at (-5.0, 0.0) moving right (+2.0 m/s) toward Glass A at (0.0, 0.0)
    excavator_observed_by_B = WorldObject(
        object_id="obj_excavator_B",
        label="excavator #2",
        confidence=0.92,
        position_x=-5.0,
        position_y=0.0,
        position_z=0.0,
        velocity_x=2.0,  # Moving right toward Glass A
        velocity_y=0.0,
        velocity_z=0.0,
        source_glasses=["glass_B"],
        last_seen=datetime.utcnow()
    )

    world_objects_s2 = {"obj_excavator_B": excavator_observed_by_B}
    active_threats_s2 = prediction_engine.evaluate_threats(world_manager.active_glasses, world_objects_s2)
    dispatched_s2 = await alert_engine.dispatch_alerts(active_threats_s2)

    alerts_for_A_s2 = alert_engine.get_alert_history("glass_A")
    alerts_for_B_s2 = alert_engine.get_alert_history("glass_B")

    assert len(alerts_for_A_s2) >= 1
    assert len(alerts_for_B_s2) == 0  # Glass B received NOTHING!
    assert alerts_for_A_s2[0]["target_glass_id"] == "glass_A"
    print("✅ Scenario 2 Verified: Glass B observes excavator -> ONLY Glass A receives directed alert! Glass B receives 0 alert noise.")

    print("\n==========================================================================")
    print("🎉 ALL PHASE 6 COLLABORATIVE THREAT DETECTION & DIRECTED ALERT ROUTER TESTS PASSED CLEANLY!")
    print("==========================================================================\n")


if __name__ == "__main__":
    test_phase_6_collaborative_threats()
