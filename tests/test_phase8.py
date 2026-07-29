"""
Automated unit and integration tests for Phase 8: Threat Prediction Engine.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from app.models.glass import GlassState, GlassPose
from app.models.object import WorldObject
from app.models.threat import ThreatLevel, ThreatType
from app.services.prediction_engine import ThreatPredictionEngine


def test_critical_head_on_threat():
    print("--- 1. Testing CRITICAL Head-On Collision Prediction ---")
    engine = ThreatPredictionEngine(cooldown_seconds=0.0)
    
    # Glass user standing at (0, 0) facing North (0°)
    glass = GlassState(
        glass_id="glass_user",
        pose=GlassPose(x=0.0, y=0.0, heading=0.0)
    )
    
    # Approaching forklift at (0, 3) moving South towards user (-3 m/s on Y)
    forklift = WorldObject(
        object_id="obj_forklift_1",
        label="forklift #1",
        confidence=0.95,
        position_x=0.0, position_y=3.0, position_z=0.0,
        velocity_x=0.0, velocity_y=-3.0, velocity_z=0.0,
        last_seen=datetime.utcnow()
    )
    
    threats = engine.evaluate_threats({"glass_user": glass}, {"obj_forklift_1": forklift})
    assert len(threats) == 1, f"Expected 1 threat alert, got {len(threats)}"
    
    alert = threats[0]
    assert alert.target_glass_id == "glass_user"
    assert alert.threat_level == ThreatLevel.CRITICAL
    assert alert.threat_type == ThreatType.FORKLIFT_APPROACH
    assert alert.time_to_collision == 1.0 # 3.0m / 3.0m/s = 1.0s
    print(f"✅ CRITICAL Head-On Threat passed! Alert: {alert.warning_message} (TTC: {alert.time_to_collision}s)")


def test_blind_spot_rear_threat():
    print("\n--- 2. Testing Blind Spot Rear Threat Detection ---")
    engine = ThreatPredictionEngine(cooldown_seconds=0.0)
    
    # Glass user standing at (0, 0) facing North (0°)
    glass = GlassState(
        glass_id="worker_1",
        pose=GlassPose(x=0.0, y=0.0, heading=0.0)
    )
    
    # Hazard behind user at (0, -4.0) -> Relative bearing 180° (Behind Blind Spot)
    hazard = WorldObject(
        object_id="obj_hazard_1",
        label="obstacle",
        confidence=0.90,
        position_x=0.0, position_y=-4.0, position_z=0.0,
        velocity_x=0.0, velocity_y=1.0, velocity_z=0.0,
        last_seen=datetime.utcnow()
    )
    
    threats = engine.evaluate_threats({"worker_1": glass}, {"obj_hazard_1": hazard})
    assert len(threats) == 1
    
    alert = threats[0]
    assert alert.threat_type == ThreatType.BLIND_SPOT_OBSTACLE
    assert "Behind (Blind Spot)" in alert.warning_message
    print(f"✅ Blind Spot Rear Threat passed! Alert message: '{alert.warning_message}'")


def test_alert_cooldown_suppression():
    print("\n--- 3. Testing Alert Cooldown Suppression ---")
    engine = ThreatPredictionEngine(cooldown_seconds=2.0)
    
    glass = GlassState(glass_id="g1", pose=GlassPose(x=0.0, y=0.0, heading=0.0))
    car = WorldObject(object_id="car_1", label="car", confidence=0.9, position_x=0.0, position_y=4.0, last_seen=datetime.utcnow())
    
    # 1st evaluation -> Should generate alert
    alerts1 = engine.evaluate_threats({"g1": glass}, {"car_1": car})
    assert len(alerts1) == 1
    
    # 2nd evaluation immediately after -> Should be suppressed by cooldown
    alerts2 = engine.evaluate_threats({"g1": glass}, {"car_1": car})
    assert len(alerts2) == 0, "Cooldown failed to suppress duplicate immediate alert!"
    print("✅ Alert Cooldown Suppression passed! Duplicate spam suppressed.")


if __name__ == "__main__":
    test_critical_head_on_threat()
    test_blind_spot_rear_threat()
    test_alert_cooldown_suppression()
    print("\n🎉 ALL PHASE 8 TESTS PASSED SUCCESSFULLY!")
