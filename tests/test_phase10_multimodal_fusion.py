"""
Phase 10: Multi-Modal Perception Fusion Unit & Integration Tests.

Tests:
1. Audio MFCC Feature Extraction & Audio Threat Classification (Gunshot, Scream, Siren, Footsteps, Ambient).
2. IMU Pattern Matching & Magnitude Calculation (Person vs Vehicle motion, Impact/Fall, Gait).
3. Pressure & Footstep Impulse Telemetry Analysis.
4. Vision-IMU Correlation (`correlate_with_vision`) when IMU magnitude > threshold.
5. Multi-Modal Unified Threat Fusion Engine.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.multimodal_fusion import (
    MultiModalFusionEngine,
    AudioThreatDetector,
    IMUPatternAnalyzer,
    PressureFootstepAnalyzer
)


def test_audio_mfcc_and_classification():
    print("--- 1. Testing Audio MFCC Feature Extraction & Threat Classifier ---")
    detector = AudioThreatDetector()

    # 1a. Quiet ambient audio
    quiet_buffer = np.sin(np.linspace(0, 10, 16000)) * 0.001
    mfcc_quiet = detector.extract_mfcc(quiet_buffer)
    res_quiet = detector.classify_audio_threat(quiet_buffer)
    assert len(mfcc_quiet) == 13
    assert res_quiet["event"] == "ambient_speech" or res_quiet["event"] == "ambient_noise"
    assert res_quiet["is_acoustic_threat"] is False

    # 1b. Loud impulse gunshot audio signal
    loud_gunshot_buffer = np.zeros(16000, dtype=np.float32)
    loud_gunshot_buffer[100:500] = 5.0  # Impulse transient
    mfcc_gunshot = detector.extract_mfcc(loud_gunshot_buffer)
    res_gunshot = detector.classify_audio_threat(loud_gunshot_buffer)
    
    assert len(mfcc_gunshot) == 13
    assert res_gunshot["event"] == "gunshot_explosion"
    assert res_gunshot["threat_score"] >= 0.95
    assert res_gunshot["is_acoustic_threat"] is True

    # 1c. Direct usage of extract_mfcc + audio_classifier pattern matching prompt
    audio_features = detector.extract_mfcc(loud_gunshot_buffer)
    classified_res = detector.audio_classifier(audio_features, db_level=105.0)
    assert classified_res["event"] == "gunshot_explosion"
    assert classified_res["threat_score"] == 0.98

    print(f"✅ Audio MFCC & Acoustic Classification passed! Gunshot threat score: {res_gunshot['threat_score']}, Event: {res_gunshot['event']}")


def test_imu_pattern_matching_person_vs_vehicle():
    print("\n--- 2. Testing IMU Acceleration Pattern Matching (Person vs Vehicle) ---")
    analyzer = IMUPatternAnalyzer()

    # 2a. Person Impact / Fall (mag > 22.0)
    ax, ay, az = 15.0, 15.0, 9.81
    mag = analyzer.calculate_magnitude(ax, ay, az)
    assert mag > 22.0
    res_fall = analyzer.analyze_imu(ax, ay, az)
    assert res_fall["pattern"] == "impact_fall"
    assert res_fall["motion_type"] == "person"
    assert res_fall["is_moving"] is True

    # 2b. Person Running (mag between 13.0 and 22.0)
    ax, ay, az = 8.0, 10.0, 9.81
    res_run = analyzer.analyze_imu(ax, ay, az)
    assert res_run["pattern"] == "running"
    assert res_run["motion_type"] == "person"

    # 2c. Vehicle Travel (steady continuous high magnitude with low gyro variance)
    ax, ay, az = 10.0, 5.0, 9.81
    res_vehicle = analyzer.analyze_imu(ax, ay, az, gx=0.01, gy=0.01, gz=0.01, imu_type_hint="vehicle")
    assert "vehicle" in res_vehicle["pattern"]
    assert res_vehicle["motion_type"] == "vehicle"

    # 2d. Stationary (mag approx 9.81)
    res_static = analyzer.analyze_imu(0.0, 0.0, 9.81)
    assert res_static["pattern"] == "stationary"
    assert res_static["is_moving"] is False

    print(f"✅ IMU Pattern Matching passed! Fall mag: {res_fall['imu_magnitude']} m/s^2, Vehicle pattern: {res_vehicle['pattern']}")


def test_pressure_footstep_analysis():
    print("\n--- 3. Testing Pressure & Footstep Telemetry Analysis ---")
    analyzer = PressureFootstepAnalyzer()

    # 3a. No footsteps
    res_none = analyzer.analyze_pressure_footsteps(None)
    assert res_none["step_count"] == 0
    assert res_none["has_footsteps"] is False

    # 3b. Rapid approaching footsteps (high pressure impulse peaks)
    pressure_buffer = np.zeros(200, dtype=np.float32)
    pressure_buffer[10] = 0.9
    pressure_buffer[30] = 0.85
    pressure_buffer[50] = 0.95
    pressure_buffer[70] = 0.88
    pressure_buffer[90] = 0.92
    pressure_buffer[110] = 0.90

    res_steps = analyzer.analyze_pressure_footsteps(pressure_buffer)
    assert res_steps["step_count"] >= 5
    assert res_steps["has_footsteps"] is True
    assert res_steps["footstep_threat_score"] > 0.50

    print(f"✅ Footstep Analysis passed! Step count: {res_steps['step_count']}, Cadence: {res_steps['cadence_hz']} Hz, Threat score: {res_steps['footstep_threat_score']}")


def test_vision_imu_correlation_and_multimodal_fusion():
    print("\n--- 4. Testing Vision-IMU Correlation & Multi-Modal Perception Fusion ---")
    engine = MultiModalFusionEngine(movement_threshold=10.2)

    # 4a. High threat multi-modal scenario (Gunshot + Running IMU + High Vision score)
    audio_gunshot = np.zeros(16000, dtype=np.float32)
    audio_gunshot[100:500] = 4.0
    imu_running = {"ax": 9.0, "ay": 10.0, "az": 9.81}

    fused_res = engine.fuse_multimodal_perception(
        vision_threat_score=0.85,
        audio_buffer=audio_gunshot,
        imu_reading=imu_running
    )

    assert fused_res["fused_threat_level"] in ["CRITICAL", "HIGH"]
    assert fused_res["multi_modal_score"] >= 0.75
    assert fused_res["vision_correlation"]["is_correlated"] is True
    assert fused_res["audio_res"]["event"] == "gunshot_explosion"
    assert fused_res["imu_res"]["pattern"] == "running"

    # 4b. Low threat ambient scenario
    quiet_audio = np.sin(np.linspace(0, 10, 16000)) * 0.001
    imu_static = {"ax": 0.0, "ay": 0.0, "az": 9.81}

    safe_res = engine.fuse_multimodal_perception(
        vision_threat_score=0.10,
        audio_buffer=quiet_audio,
        imu_reading=imu_static
    )

    assert safe_res["fused_threat_level"] == "SAFE"
    assert safe_res["multi_modal_score"] < 0.25

    print(f"✅ Multi-Modal Perception Fusion passed! Fused score: {fused_res['multi_modal_score']}, Level: {fused_res['fused_threat_level']}")


if __name__ == "__main__":
    test_audio_mfcc_and_classification()
    test_imu_pattern_matching_person_vs_vehicle()
    test_pressure_footstep_analysis()
    test_vision_imu_correlation_and_multimodal_fusion()
    print("\n🎉 ALL PHASE 10 MULTI-MODAL FUSION TESTS PASSED SUCCESSFULLY!")
