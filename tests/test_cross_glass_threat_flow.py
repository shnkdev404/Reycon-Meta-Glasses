"""
Integration test for Cross-Glass Threat Broadcast Flow:
1. Glass B connects and reports a detected truck threat.
2. SharedWorldManager registers the threat globally.
3. Glass A connects and receives alerts for threats detected by Glass B!
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.shared_world_manager import SharedWorldManager, Position3D


def test_cross_glass_threat_flow():
    # 1. Initialize clean SharedWorldManager instance
    wm = SharedWorldManager()
    wm.reset()

    # 2. Register Glass A and Glass B positions
    glass_a_pos = Position3D(x=0.0, y=0.0, z=0.0)
    glass_b_pos = Position3D(x=40.0, y=0.0, z=0.0)

    wm.register_glass("glass_a", glass_a_pos)
    wm.register_glass("glass_b", glass_b_pos)

    # 3. Glass B detects a truck threat at (x=30, y=0, z=0)
    # Distance from Glass A = 30m (which falls into 20m-50m SHARED priority window)
    threat_pos = Position3D(x=30.0, y=0.0, z=0.0)
    threat = wm.add_threat(
        threat_id="threat_001",
        object_type="truck",
        position=threat_pos,
        detected_by_glass_id="glass_b",
        confidence=0.95
    )

    assert threat is not None
    assert len(wm.threats) == 1

    # 4. Glass A queries its alerts
    alerts_a = wm.get_alerts_for_glass("glass_a")

    assert len(alerts_a) == 1
    alert = alerts_a[0]

    # Verify alert fields match expected cross-glass threat payload
    assert alert["threat_id"] == "threat_001"
    assert alert["type"] == "truck"
    assert alert["distance"] == 30.0  # 30 meters away from Glass A
    assert alert["detected_by"] == "glass_b"  # FROM OTHER GLASS!
    assert alert["priority"] == "SHARED"

    print("✅ TEST PASSED: Glass A successfully notified of SHARED threat detected by Glass B!")
    print(f"🚨 Notification Payload for Glass A: {alert}")


if __name__ == "__main__":
    test_cross_glass_threat_flow()
