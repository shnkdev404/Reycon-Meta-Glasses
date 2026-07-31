"""
Master Test Runner for Shared Perception Stack (Phases 1 through 11).
Executes end-to-end unit, integration, and orchestration tests across all system phases simultaneously.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import asyncio
from datetime import datetime, timedelta, timezone

# Phase 1
from app.services.connection_manager import connection_manager

# Phase 2
from app.sensors import (
    SimulatedCameraSensor,
    SimulatedIMUSensor,
    SimulatedGPSSensor,
    SimulatedHeadPoseSensor,
    SimulatedDepthSensor
)

# Phase 3
from app.vision.detector import YOLOWrapper
from app.vision.tracker import TrackManager
from app.vision.depth import DepthEstimatorWrapper
from app.services.tracking_manager import TrackingManager

# Phase 4
from app.vision.slam import SLAMManager

# Phase 5
from app.models.glass import GlassPose, GlassState, GlassSensors
from app.models.object import BoundingBox2D, Detection2D, WorldObject
from app.services.geometry import (
    polar_to_cartesian_relative,
    camera_to_world_2d,
    camera_to_world_3d,
    world_to_camera_3d,
    world_to_relative_polar,
    gps_to_enu
)
from app.services.coordinate_transform import coordinate_transformer

# Phase 6
from app.services.world_manager import world_manager

# Phase 7
from app.services.fusion_engine import fusion_engine

# Phase 8
from app.models.threat import ThreatAlert, ThreatLevel, ThreatType
from app.services.prediction_engine import ThreatPredictionEngine, prediction_engine

# Phase 9
from app.services.alert_engine import alert_engine

# Phase 10
from app.meta import MetaCameraAdapter, MetaIMUAdapter, MetaPoseAdapter, MetaAlertAdapter

# Phase 11
from app.api.routes import home, health, get_world_state, get_glasses, get_threats
from app.dashboard.visualizer import render_dashboard


def run_phase_1_tests():
    print("\n==================================================")
    print("🚀 TESTING PHASE 1: Networking & Connection Manager")
    print("==================================================")
    cm = connection_manager
    assert len(cm.active_connections) == 0
    print("✅ Phase 1 ConnectionManager verified!")


def run_phase_2_tests():
    print("\n==================================================")
    print("📡 TESTING PHASE 2: Sensor Interface Layer")
    print("==================================================")
    cam = SimulatedCameraSensor("glass_01")
    assert cam.start() is True
    assert cam.read_frame()["width"] == 1920
    assert SimulatedIMUSensor("glass_01").read_imu().accel_z == 9.81
    assert SimulatedGPSSensor("glass_01").read_gps().latitude == 37.7749
    assert SimulatedHeadPoseSensor("glass_01").read_head_pose().z == 1.65
    assert SimulatedDepthSensor("glass_01").read_depth_map()["min_depth_m"] == 0.5
    print("✅ Phase 2 Sensors (Camera, IMU, GPS, HeadPose, Depth) verified!")


def run_phase_3_tests():
    print("\n==================================================")
    print("👁️ TESTING PHASE 3: Vision & Multi-Object Tracking")
    print("==================================================")
    yolo = YOLOWrapper()
    dets = yolo.detect({})
    assert len(dets) >= 1
    
    tracker = TrackManager()
    tracked = tracker.update(dets)
    assert "#1" in tracked[0].label
    
    dest = DepthEstimatorWrapper(focal_length_px=800.0)
    bbox = BoundingBox2D(xmin=100, ymin=100, xmax=200, ymax=300)
    assert 6.0 <= dest.estimate_depth(None, bbox, "person") <= 7.5
    
    tm = TrackingManager()
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(seconds=1.0)
    o0 = {"v1": WorldObject(object_id="v1", label="car", confidence=0.9, position_x=0.0, position_y=0.0, last_seen=t0)}
    o1 = {"v1": WorldObject(object_id="v1", label="car", confidence=0.9, position_x=3.0, position_y=4.0, last_seen=t1)}
    tm.update_tracks(o0)
    upd = tm.update_tracks(o1)
    assert upd["v1"].velocity_x > 1.0
    print("✅ Phase 3 Vision & Tracking pipeline verified!")


def run_phase_4_tests():
    print("\n==================================================")
    print("🎯 TESTING PHASE 4: Pose Estimation & Visual SLAM")
    print("==================================================")
    slam = SLAMManager()
    assert slam.get_pose().z == 1.65
    sensors = GlassSensors(accel_x=1.5, accel_y=0.0, accel_z=9.81, gyro_z=0.05)
    slam.track_pose(imu_data=sensors, dt=0.1)
    assert slam.get_pose().x > 0.0 or slam.get_pose().y > 0.0
    slam.reset_origin(10.0, 20.0, 1.7, 90.0)
    assert slam.get_pose().x == 10.0 and slam.get_pose().y == 20.0
    print("✅ Phase 4 VIO Dead-Reckoning & SLAM engine verified!")


def run_phase_5_tests():
    print("\n==================================================")
    print("📐 TESTING PHASE 5: Spatial Geometry & Transformations")
    print("==================================================")
    pose = GlassPose(x=10.0, y=20.0, z=1.65, heading=90.0)
    wx, wy, wz = camera_to_world_3d(rel_x=0.0, rel_y=5.0, rel_z=0.0, glass_pose=pose)
    assert abs(wx - 15.0) < 0.01 and abs(wy - 20.0) < 0.01
    
    dist, bearing = world_to_relative_polar(15.0, 20.0, pose)
    assert round(dist, 1) == 5.0 and round(bearing, 1) == 0.0
    
    east, north, up = gps_to_enu(37.7759, -122.4194, 10.0)
    assert 100.0 <= north <= 120.0
    
    gstate = GlassState(glass_id="g1", pose=pose)
    det = Detection2D(label="forklift #1", confidence=0.9, distance=5.0, bearing=0.0)
    wobj = coordinate_transformer.transform_detection_to_world(det, gstate)
    assert wobj.object_id == "obj_forklift_1"
    print("✅ Phase 5 3D Spatial Geometry & Coordinate Transformer verified!")


def run_phase_6_tests():
    print("\n==================================================")
    print("🌐 TESTING PHASE 6: Shared World Model Orchestrator")
    print("==================================================")
    asyncio.run(_async_phase_6_test())


async def _async_phase_6_test():
    world_manager.reset_world_state()
    glass_alpha = GlassState(glass_id="glass_alpha", pose=GlassPose(x=0.0, y=0.0, heading=0.0))
    glass_beta = GlassState(glass_id="glass_beta", pose=GlassPose(x=10.0, y=10.0, heading=180.0))
    
    dets_alpha = [Detection2D(label="vehicle", confidence=0.92, distance=4.0, bearing=0.0)]
    dets_beta = [Detection2D(label="person", confidence=0.85, distance=3.0, bearing=0.0)]
    
    await world_manager.update_glass_telemetry(glass_alpha, dets_alpha)
    await world_manager.update_glass_telemetry(glass_beta, dets_beta)
    
    full_state = await world_manager.get_full_world_state()
    assert full_state["active_glasses_count"] == 2
    assert len(full_state["world_objects"]) >= 2
    world_manager.prune_stale_world_objects(max_age_seconds=0.001)
    print("✅ Phase 6 Central World Model Orchestrator verified!")


def run_phase_7_tests():
    print("\n==================================================")
    print("🔗 TESTING PHASE 7: Perception Fusion Engine")
    print("==================================================")
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(seconds=0.5)
    obj_a = WorldObject(object_id="car_1", label="vehicle #1", confidence=0.85, position_x=5.0, position_y=10.0, source_glasses=["glass_A"], last_seen=t0)
    obj_b = WorldObject(object_id="car_2", label="vehicle #2", confidence=0.90, position_x=5.2, position_y=10.2, source_glasses=["glass_B"], last_seen=t1)
    
    world0 = fusion_engine.fuse_objects([obj_a], {})
    world1 = fusion_engine.fuse_objects([obj_b], world0)
    assert len(world1) == 1
    assert world1["car_1"].confidence == 1.0
    print("✅ Phase 7 Perception Fusion Engine (Multi-Glass Clustering & Confidence Boost) verified!")


def run_phase_8_tests():
    print("\n==================================================")
    print("🎯 TESTING PHASE 8: Threat Prediction Engine")
    print("==================================================")
    pengine = ThreatPredictionEngine(cooldown_seconds=0.0)
    glass = GlassState(glass_id="worker_rear", pose=GlassPose(x=0.0, y=0.0, heading=0.0))
    hazard = WorldObject(object_id="obj_rear", label="obstacle", confidence=0.9, position_x=0.0, position_y=-3.0, last_seen=datetime.now(timezone.utc))
    
    threats = pengine.evaluate_threats({"worker_rear": glass}, {"obj_rear": hazard})
    assert len(threats) == 1
    assert threats[0].threat_type == ThreatType.BLIND_SPOT_OBSTACLE
    assert threats[0].threat_level == ThreatLevel.CRITICAL
    print("✅ Phase 8 Threat Prediction Engine (4-Tier & Blind Spot Detection) verified!")


def run_phase_9_tests():
    print("\n==================================================")
    print("📡 TESTING PHASE 9: Directed Alert Decision Engine")
    print("==================================================")
    asyncio.run(_async_phase_9_test())


async def _async_phase_9_test():
    alert_engine.clear_alert_history()
    alert = ThreatAlert(
        alert_id="alt_phase9", target_glass_id="glass_target", trigger_object_id="obj_forklift_9",
        threat_type=ThreatType.FORKLIFT_APPROACH, threat_level=ThreatLevel.CRITICAL,
        time_to_collision=1.0, distance=2.5, bearing=180.0,
        warning_message="CRITICAL: Forklift detected at Behind (Blind Spot) (2.5m away!)", timestamp=datetime.now(timezone.utc)
    )
    dispatched = await alert_engine.dispatch_alerts([alert])
    assert dispatched == 1
    assert len(alert_engine.get_alert_history("glass_target")) == 1
    print("✅ Phase 9 Directed Alert Decision Engine (Non-broadcast routing) verified!")


def run_phase_10_tests():
    print("\n==================================================")
    print("🕶️ TESTING PHASE 10: Meta Wearable SDK Adapters")
    print("==================================================")
    asyncio.run(_async_phase_10_test())


async def _async_phase_10_test():
    cam = MetaCameraAdapter("meta_01")
    assert cam.start_stream() is True
    assert cam.get_latest_frame()["source"] == "MetaWearableSDK"
    
    imu = MetaIMUAdapter("meta_01")
    assert imu.read_imu().accel_z == 9.81
    
    pose = MetaPoseAdapter("meta_01", x=1.0, y=2.0, heading=45.0)
    assert pose.get_current_pose().heading == 45.0
    
    alert_adapter = MetaAlertAdapter("meta_01")
    alert = ThreatAlert(
        alert_id="alt_m10", target_glass_id="meta_01", trigger_object_id="obj_car",
        threat_type=ThreatType.VEHICLE_APPROACH, threat_level=ThreatLevel.CRITICAL,
        time_to_collision=1.5, distance=3.0, bearing=90.0, warning_message="CRITICAL: Car at Right", timestamp=datetime.now(timezone.utc)
    )
    assert await alert_adapter.send_alert(alert) is True
    print("✅ Phase 10 Meta Wearable SDK Hardware Adapters (Camera, IMU, Pose, HUD/Haptics) verified!")


def run_phase_11_tests():
    print("\n==================================================")
    print("💻 TESTING PHASE 11: Real-time Debug Dashboard & REST APIs")
    print("==================================================")
    assert home()["status"] == "active"
    assert health()["status"] == "OK"
    wstate = asyncio.run(get_world_state())
    assert "active_glasses_count" in wstate
    dash = asyncio.run(render_dashboard())
    assert "REYCON" in dash.body.decode("utf-8") or "Command" in dash.body.decode("utf-8")
    print("✅ Phase 11 Real-time Debug Dashboard Visualizer & REST APIs verified!")


def main():
    print("\n==========================================================================")
    print("🧪 EXECUTING MASTER SYSTEM TEST SUITE (PHASES 1 THROUGH 11 SIMULTANEOUSLY)")
    print("==========================================================================")
    
    run_phase_1_tests()
    run_phase_2_tests()
    run_phase_3_tests()
    run_phase_4_tests()
    run_phase_5_tests()
    run_phase_6_tests()
    run_phase_7_tests()
    run_phase_8_tests()
    run_phase_9_tests()
    run_phase_10_tests()
    run_phase_11_tests()
    
    print("\n==========================================================================")
    print("🎉 ALL PHASES (1 THROUGH 11) PASSED SIMULTANEOUSLY!")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
