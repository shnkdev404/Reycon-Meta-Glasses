"""
Phase 12 & Phase 13: Anomaly Detection and Multi-Factor Threat Scoring Unit Tests.

Tests:
1. Trajectory & Behavioral Anomaly Detection using IsolationForest (Normal vs Person Frozen vs Sudden Acceleration vs Erratic path).
2. Multi-Factor Risk Weighted Threat Scoring Formula:
   threat_score = 0.4 * confidence + 0.3 * (1 / distance) + 0.2 * size_ratio + 0.1 * velocity_magnitude
3. Prediction Engine ThreatAlert generation with multi-factor risk scores and anomaly detection.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.glass import GlassState, GlassPose
from app.models.object import WorldObject
from app.models.threat import ThreatLevel, ThreatType
from app.services.anomaly_detector import TrajectoryAnomalyDetector, anomaly_detector
from app.services.threat_scorer import MultiFactorThreatScorer, threat_scorer
from app.services.prediction_engine import ThreatPredictionEngine


def test_isolation_forest_anomaly_detection():
    print("--- 1. Testing IsolationForest Trajectory & Behavioral Anomaly Detection ---")
    detector = TrajectoryAnomalyDetector()

    # 1a. Normal trajectory history (walking continuously 1.2 m/s)
    normal_history = [(0.0, 0.0, 0.0), (1.2, 0.1, 0.0), (2.4, 0.2, 0.0), (3.6, 0.3, 0.0)]
    res_normal = detector.detect_trajectory_anomaly(normal_history)
    assert res_normal["is_anomaly"] is False
    assert res_normal["anomaly_type"] == "NORMAL"

    # 1b. Frozen Person anomaly (stopped moving for extended duration)
    frozen_features = np.array([[0.01, 0.0, 0.0, 0.0, 6.0]], dtype=np.float32)
    res_frozen = detector.predict_anomaly(frozen_features)
    assert res_frozen["is_anomaly"] is True
    assert res_frozen["anomaly_type"] == "PERSON_FROZEN"

    # 1c. Sudden Acceleration anomaly (extreme accel & jerk spike)
    accel_features = np.array([[4.5, 4.2, 3.5, 10.0, 0.0]], dtype=np.float32)
    res_accel = detector.predict_anomaly(accel_features)
    assert res_accel["is_anomaly"] is True
    assert res_accel["anomaly_type"] == "SUDDEN_ACCELERATION"

    # 1d. Erratic Trajectory anomaly (sharp turn angles > 60 degrees)
    erratic_features = np.array([[2.5, 2.0, 1.5, 85.0, 0.0]], dtype=np.float32)
    res_erratic = detector.predict_anomaly(erratic_features)
    assert res_erratic["is_anomaly"] is True
    assert res_erratic["anomaly_type"] == "ERRATIC_TRAJECTORY"

    print(f"✅ Anomaly Detection passed! Frozen anomaly: {res_frozen['anomaly_type']}, Accel anomaly: {res_accel['anomaly_type']}, Erratic anomaly: {res_erratic['anomaly_type']}")


def test_multi_factor_threat_scoring():
    print("\n--- 2. Testing Multi-Factor Threat Scoring Formula --- Grand Formula Verification ---")
    scorer = MultiFactorThreatScorer()

    # Case A: High risk close proximity hazard
    # threat_score = 0.4*0.9 + 0.3*(1/1.0) + 0.2*0.3 + 0.1*(2.5/5.0)
    # = 0.36 + 0.30 + 0.06 + 0.05 = 0.77 -> CRITICAL
    score_high, comps_high = scorer.compute_threat_score(
        confidence=0.90,
        distance=1.0,
        person_size_ratio=0.30,
        velocity_magnitude=2.5
    )

    assert 0.75 <= score_high <= 1.0
    assert comps_high["confidence_component"] == 0.36
    assert comps_high["proximity_component"] == 0.30
    assert comps_high["size_ratio_component"] == 0.06
    assert comps_high["velocity_component"] == 0.05
    assert scorer.score_to_threat_level(score_high) == ThreatLevel.CRITICAL

    # Case B: Low risk distant object
    # threat_score = 0.4*0.8 + 0.3*(1/10.0) + 0.2*0.05 + 0.1*(0.0/5.0)
    # = 0.32 + 0.03 + 0.01 + 0.0 = 0.36 -> MEDIUM
    score_low, comps_low = scorer.compute_threat_score(
        confidence=0.80,
        distance=10.0,
        person_size_ratio=0.05,
        velocity_magnitude=0.0
    )

    assert 0.25 <= score_low <= 0.45
    assert scorer.score_to_threat_level(score_low) in [ThreatLevel.MEDIUM, ThreatLevel.LOW]

    print(f"✅ Multi-Factor Threat Scoring passed! Close hazard score: {score_high} ({scorer.score_to_threat_level(score_high).value}), Distant score: {score_low}")


def test_prediction_engine_multi_factor_threat_alerts():
    print("\n--- 3. Testing Threat Prediction Engine Integration with Threat Scoring ---")
    engine = ThreatPredictionEngine(cooldown_seconds=0.0)

    glass = GlassState(
        glass_id="glass_test_01",
        pose_obj=GlassPose(x=0.0, y=0.0, z=0.0, heading=0.0)
    )

    # Approaching Forklift 3.0 meters away
    forklift_obj = WorldObject(
        object_id="obj_forklift_99",
        label="forklift #99",
        confidence=0.92,
        position_x=0.0,
        position_y=3.0,
        position_z=0.0,
        velocity_x=0.0,
        velocity_y=-2.0,
        velocity_z=0.0
    )

    alerts = engine.evaluate_threats({"glass_test_01": glass}, {"obj_forklift_99": forklift_obj})
    assert len(alerts) == 1
    alert = alerts[0]

    assert alert.target_glass_id == "glass_test_01"
    assert alert.threat_type == ThreatType.FORKLIFT_APPROACH
    assert alert.threat_score > 0.40
    assert "confidence_component" in alert.score_components
    assert "proximity_component" in alert.score_components

    print(f"✅ Threat Prediction Engine Multi-Factor Alert verified! Generated Alert ID: {alert.alert_id}, Threat Score: {alert.threat_score}, Level: {alert.threat_level.value}")


if __name__ == "__main__":
    test_isolation_forest_anomaly_detection()
    test_multi_factor_threat_scoring()
    test_prediction_engine_multi_factor_threat_alerts()
    print("\n🎉 ALL PHASE 12 & 13 ANOMALY & THREAT SCORING TESTS PASSED SUCCESSFULLY!")
