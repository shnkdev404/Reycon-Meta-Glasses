"""
Tracking & Trajectory Anomaly Detection Module.
Provides 1D Kalman Filter bounding box smoothing (SmoothTrack) and trajectory anomaly detection.
"""
import logging
from typing import Tuple, Dict, List, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class KalmanFilter1D:
    """Simple 1D Kalman filter for smoothing object trajectories."""
    
    def __init__(self, process_variance: float = 0.1, measurement_variance: float = 2.0, initial_value: float = 0.0, initial_estimate_error: float = 1.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = initial_estimate_error
    
    def update(self, measurement: float) -> float:
        """Update filter with new measurement."""
        # Predict
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate


class SmoothTrack:
    """Smooth detected bounding boxes across frames."""
    
    def __init__(self, smoothing_factor: float = 0.7):
        self.smoothing_factor = smoothing_factor
        self.prev_boxes: Dict[int, Tuple[float, float, float, float]] = {}
        self.kalman_filters: Dict[int, List[KalmanFilter1D]] = {}
    
    def smooth_detection(self, track_id: int, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Apply Kalman smoothing to bounding box."""
        x1, y1, x2, y2 = bbox
        
        if track_id not in self.kalman_filters:
            # Initialize Kalman filters for this track
            self.kalman_filters[track_id] = [
                KalmanFilter1D(process_variance=0.1, measurement_variance=2.0, initial_value=x)
                for x in [x1, y1, x2, y2]
            ]
        
        # Update each dimension
        filters = self.kalman_filters[track_id]
        smooth_box = tuple(kf.update(coord) for kf, coord in zip(filters, [x1, y1, x2, y2]))
        
        return smooth_box


class TrajectoryAnomalyDetector:
    """Detect unusual object trajectories."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.trajectories: Dict[int, deque] = {}
    
    def update_trajectory(self, track_id: int, position: Tuple[float, float, float]):
        """Add position to trajectory."""
        if track_id not in self.trajectories:
            self.trajectories[track_id] = deque(maxlen=self.window_size)
        
        self.trajectories[track_id].append(position)
    
    def is_anomalous(self, track_id: int, threshold: float = 2.0) -> Tuple[bool, float]:
        """Check if trajectory is anomalous."""
        if track_id not in self.trajectories:
            return False, 0.0
        
        traj_list = list(self.trajectories[track_id])
        
        if len(traj_list) < 3:
            return False, 0.0
        
        traj = np.array(traj_list)
        if len(traj) < 2:
            return False, 0.0
        
        velocity = np.diff(traj, axis=0)
        if len(velocity) < 2:
            return False, 0.0
        
        acceleration = np.diff(velocity, axis=0)
        
        features = np.array([
            np.linalg.norm(velocity.mean(axis=0)),
            np.linalg.norm(acceleration.mean(axis=0)) if len(acceleration) > 0 else 0.0
        ]).reshape(1, -1)
        
        accel_magnitude = features[0, 1]
        
        anomaly_score = min(1.0, float(accel_magnitude / 5.0))
        is_anomalous_bool = anomaly_score > threshold / 10.0
        
        return is_anomalous_bool, round(float(anomaly_score), 3)
