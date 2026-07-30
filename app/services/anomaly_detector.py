"""
Phase 12: Anomaly Detection Engine.

Detects unusual trajectories and behavioral anomalies (e.g., person frozen, abnormal movement,
sudden acceleration, erratic path) using scikit-learn IsolationForest.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("AnomalyDetector")


class TrajectoryAnomalyDetector:
    """
    ML-based Trajectory & Behavioral Anomaly Detector using Isolation Forest.
    """

    def __init__(self, contamination: str | float = "auto", random_state: int = 42):
        self.contamination = contamination
        self.clf = IsolationForest(contamination=contamination, random_state=random_state)
        self._is_fitted = False
        self._fit_baseline_model()

    def _fit_baseline_model(self) -> None:
        """
        Fits baseline IsolationForest on normal trajectory feature distributions.
        Normal movement: velocity 0.2-2.5 m/s, low acceleration, smooth turning.
        """
        np.random.seed(42)
        n_samples = 300
        
        # Features: [vel_mag, accel_mag, jerk_mag, turn_angle_deg, stillness_duration]
        vel_mag = np.random.normal(1.2, 0.5, n_samples).clip(0.1, 2.5)
        accel_mag = np.random.normal(0.2, 0.3, n_samples).clip(0.0, 1.2)
        jerk_mag = np.random.normal(0.1, 0.2, n_samples).clip(0.0, 0.6)
        turn_angle = np.random.normal(10.0, 15.0, n_samples).clip(0.0, 35.0)
        stillness = np.random.normal(0.5, 0.5, n_samples).clip(0.0, 1.5)

        normal_features = np.column_stack([vel_mag, accel_mag, jerk_mag, turn_angle, stillness])
        
        # Include explicit steady walking / standing inliers
        inliers = np.array([
            [1.2, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 5.0, 0.0],
            [1.5, 0.2, 0.1, 10.0, 0.0],
            [0.8, 0.1, 0.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        training_data = np.vstack([normal_features, inliers])
        self.clf.fit(training_data)
        self._is_fitted = True
        logger.info("✅ IsolationForest Trajectory Anomaly Detector baseline model trained.")

    def extract_trajectory_features(self, position_history: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Extracts 5 key numerical features from 3D spatial position history list:
          1. Average Velocity Magnitude (m/s)
          2. Acceleration Magnitude (m/s^2)
          3. Jerk Magnitude (m/s^3)
          4. Max Turn Angle (degrees)
          5. Stillness Duration (seconds)
        """
        if not position_history or len(position_history) < 2:
            return np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        pts = np.array(position_history, dtype=np.float32)
        diffs = np.diff(pts, axis=0) # Velocity vectors assuming 1 sec intervals
        velocities = np.linalg.norm(diffs, axis=1)

        avg_vel = float(np.mean(velocities))
        
        if len(velocities) >= 2:
            accels = np.diff(velocities)
            avg_accel = float(np.mean(np.abs(accels)))
        else:
            avg_accel = 0.0

        if len(velocities) >= 3:
            jerks = np.diff(accels)
            avg_jerk = float(np.mean(np.abs(jerks)))
        else:
            avg_jerk = 0.0

        # Calculate turning angle
        if len(diffs) >= 2:
            v1 = diffs[-2]
            v2 = diffs[-1]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-3 and n2 > 1e-3:
                cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                turn_angle = float(np.degrees(np.arccos(cos_theta)))
            else:
                turn_angle = 0.0
        else:
            turn_angle = 0.0

        # Calculate stillness duration (consecutive low velocity steps < 0.1 m/s)
        still_count = 0
        for v in reversed(velocities):
            if v < 0.1:
                still_count += 1
            else:
                break
        stillness_duration = float(still_count)

        return np.array([[avg_vel, avg_accel, avg_jerk, turn_angle, stillness_duration]], dtype=np.float32)

    def predict_anomaly(self, trajectory_features: np.ndarray) -> Dict[str, Any]:
        """
        Detects anomalies using IsolationForest:
          is_anomaly = clf.predict(trajectory_features) == -1
        """
        if not self._is_fitted:
            self._fit_baseline_model()

        features = np.asarray(trajectory_features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Isolation Forest prediction (-1 for outlier/anomaly, 1 for normal)
        pred = self.clf.predict(features)[0]
        decision_score = float(self.clf.decision_function(features)[0])
        is_anomaly = bool(pred == -1)

        vel_mag, accel_mag, jerk_mag, turn_angle, stillness = features[0][:5]

        # Determine detailed anomaly classification type
        if is_anomaly:
            if stillness > 4.0 or (vel_mag < 0.05 and stillness > 2.5):
                anomaly_type = "PERSON_FROZEN"
            elif accel_mag > 3.0 or jerk_mag > 2.0:
                anomaly_type = "SUDDEN_ACCELERATION"
            elif turn_angle > 60.0:
                anomaly_type = "ERRATIC_TRAJECTORY"
            else:
                anomaly_type = "ABNORMAL_MOVEMENT"
        else:
            anomaly_type = "NORMAL"

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(1.0 - decision_score, 4),
            "anomaly_type": anomaly_type,
            "features": {
                "velocity_mag": round(float(vel_mag), 2),
                "accel_mag": round(float(accel_mag), 2),
                "jerk_mag": round(float(jerk_mag), 2),
                "turn_angle_deg": round(float(turn_angle), 1),
                "stillness_sec": round(float(stillness), 1)
            }
        }

    def detect_trajectory_anomaly(self, position_history: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        Convenience wrapper extracting features and predicting anomaly from 3D position history.
        """
        features = self.extract_trajectory_features(position_history)
        return self.predict_anomaly(features)


anomaly_detector = TrajectoryAnomalyDetector()
