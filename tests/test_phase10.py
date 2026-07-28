"""
Automated unit and integration tests for Phase 10: Meta Wearable SDK Integration & Adapters.
"""
import asyncio
from datetime import datetime
from app.meta.camera_adapter import MetaCameraAdapter
from app.meta.imu_adapter import MetaIMUAdapter
from app.meta.pose_adapter import MetaPoseAdapter
from app.meta.alert_adapter import MetaAlertAdapter
from app.models.threat import ThreatAlert, ThreatLevel, ThreatType


def test_meta_camera_adapter():
    print("--- 1. Testing MetaCameraAdapter ---")
    cam = MetaCameraAdapter("glass_meta_01")
    assert cam.start_stream() is True
    
    frame = cam.get_latest_frame()
    assert frame is not None
    assert frame["width"] == 1920
    assert frame["height"] == 1080
    assert frame["source"] == "MetaWearableSDK"
    
    cam.stop_stream()
    assert cam.get_latest_frame() is None
    print("✅ MetaCameraAdapter passed!")


def test_meta_imu_adapter():
    print("\n--- 2. Testing MetaIMUAdapter ---")
    imu = MetaIMUAdapter("glass_meta_01")
    readings = imu.read_imu()
    assert readings.accel_z == 9.81
    print(f"✅ MetaIMUAdapter passed! Accel Z: {readings.accel_z} m/s^2")


def test_meta_pose_adapter():
    print("\n--- 3. Testing MetaPoseAdapter ---")
    pose_adapter = MetaPoseAdapter("glass_meta_01", x=5.0, y=10.0, heading=90.0)
    pose = pose_adapter.get_current_pose()
    assert pose.x == 5.0
    assert pose.y == 10.0
    assert pose.heading == 90.0
    print(f"✅ MetaPoseAdapter passed! Pose: ({pose.x}, {pose.y}, {pose.z}) Heading: {pose.heading}°")


def test_meta_alert_adapter():
    print("\n--- 4. Testing MetaAlertAdapter ---")
    asyncio.run(_run_alert_adapter_test())


async def _run_alert_adapter_test():
    alert_adapter = MetaAlertAdapter("glass_meta_01")
    alert = ThreatAlert(
        alert_id="alt_meta_1",
        target_glass_id="glass_meta_01",
        trigger_object_id="obj_forklift_9",
        threat_type=ThreatType.FORKLIFT_APPROACH,
        threat_level=ThreatLevel.CRITICAL,
        time_to_collision=1.0,
        distance=2.5,
        bearing=270.0, # Left side (270° = -90° azimuth)
        warning_message="CRITICAL: Forklift detected at Left (2.5m away!)",
        timestamp=datetime.utcnow()
    )
    
    success = await alert_adapter.send_alert(alert)
    assert success is True
    
    azimuth = alert_adapter.calculate_audio_azimuth(alert.bearing)
    assert azimuth == 90.0 or azimuth == -90.0 # 270 deg normalized azimuth
    print(f"✅ MetaAlertAdapter passed! Spatial Audio Azimuth: {azimuth}°")


if __name__ == "__main__":
    test_meta_camera_adapter()
    test_meta_imu_adapter()
    test_meta_pose_adapter()
    test_meta_alert_adapter()
    print("\n🎉 ALL PHASE 10 TESTS PASSED SUCCESSFULLY!")
