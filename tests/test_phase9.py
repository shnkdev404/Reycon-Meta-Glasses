"""
Automated unit and integration tests for Phase 9: Directed Alert Decision Engine.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from datetime import datetime
from app.models.threat import ThreatAlert, ThreatLevel, ThreatType
from app.services.alert_engine import alert_engine


def test_directed_alert_dispatch():
    print("--- 1. Testing Directed Non-Broadcast Alert Dispatching ---")
    asyncio.run(_run_dispatch_test())


async def _run_dispatch_test():
    alert_engine.clear_alert_history()
    
    # Create targeted alert for Glass B (Glass A unaffected)
    alert = ThreatAlert(
        alert_id="alt_123",
        target_glass_id="glass_B",
        trigger_object_id="obj_forklift_1",
        threat_type=ThreatType.FORKLIFT_APPROACH,
        threat_level=ThreatLevel.CRITICAL,
        time_to_collision=1.2,
        distance=3.5,
        bearing=180.0, # Behind user
        warning_message="CRITICAL: Forklift detected at Behind (Blind Spot) (3.5m away!)",
        timestamp=datetime.utcnow()
    )
    
    count = await alert_engine.dispatch_alerts([alert])
    assert count == 1, f"Expected 1 alert dispatched, got {count}"
    
    history = alert_engine.get_alert_history("glass_B")
    assert len(history) == 1
    assert history[0]["target_glass_id"] == "glass_B"
    assert history[0]["threat_level"] == "CRITICAL"
    print(f"✅ Directed Alert Dispatch passed! Dispatched ONLY to 'glass_B' (Message: '{history[0]['warning_message']}')")


def test_alert_frequency_throttling():
    print("\n--- 2. Testing Alert Frequency Throttling ---")
    asyncio.run(_run_throttling_test())


async def _run_throttling_test():
    alert_engine.clear_alert_history()
    
    alert = ThreatAlert(
        alert_id="alt_456",
        target_glass_id="glass_B",
        trigger_object_id="obj_car_1",
        threat_type=ThreatType.VEHICLE_APPROACH,
        threat_level=ThreatLevel.HIGH,
        time_to_collision=3.0,
        distance=5.0,
        bearing=0.0,
        warning_message="HIGH: Vehicle detected at Front (5.0m away!)",
        timestamp=datetime.utcnow()
    )
    
    count1 = await alert_engine.dispatch_alerts([alert])
    assert count1 == 1, "First alert failed to dispatch"
    
    # Immediate second dispatch should be throttled
    count2 = await alert_engine.dispatch_alerts([alert])
    assert count2 == 0, "Frequency throttling failed to suppress repeated alert"
    print("✅ Alert Frequency Throttling passed! Repeated alert throttled.")


if __name__ == "__main__":
    test_directed_alert_dispatch()
    test_alert_frequency_throttling()
    print("\n🎉 ALL PHASE 9 TESTS PASSED SUCCESSFULLY!")
