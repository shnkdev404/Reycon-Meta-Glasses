"""
Reycon Meta Glasses - Performance & Accuracy Optimization Code Examples.
Practical implementations for key improvements:
1. Kalman Filter for Temporal Smoothing
2. Non-Maximum Suppression (Soft-NMS)
3. Depth Sensor Integration
4. Optical Flow for Velocity Estimation
5. Multi-Factor Threat Scoring
6. Confidence Calibration
7. Lightweight Anomaly Detection
8. Adaptive Frame Skipping for Speed
9. Batch Processing for GPU
10. Integrated Production Optimization Pipeline
"""
import time
import math
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import deque
import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 1. KALMAN FILTER FOR TEMPORAL SMOOTHING
# ============================================================================

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


# ============================================================================
# 2. NON-MAXIMUM SUPPRESSION (NMS) POST-PROCESSING
# ============================================================================

def soft_nms(boxes: List[Any], scores: List[float], iou_threshold: float = 0.5, sigma: float = 0.5, score_threshold: float = 0.01) -> Tuple[List[Any], List[float]]:
    """
    Soft-NMS: Instead of removing overlapping boxes, reduce their confidence.
    Fixes: Multiple detections of same object
    """
    if len(boxes) == 0:
        return boxes, scores
    
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)
    
    # Sort by score descending
    sorted_idx = np.argsort(-scores_np).tolist()
    
    # Compute IoU matrix
    def iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    keep = []
    for i in sorted_idx:
        if scores_np[i] < score_threshold:
            continue
        
        keep_box = True
        for j in keep:
            iou_score = iou(boxes_np[i], boxes_np[j])
            
            if iou_score > iou_threshold:
                # Reduce confidence via Gaussian
                scores_np[i] *= np.exp(-iou_score ** 2 / sigma)
                
                if scores_np[i] < score_threshold:
                    keep_box = False
                    break
        
        if keep_box:
            keep.append(i)
    
    return boxes_np[keep].tolist(), scores_np[keep].tolist()


# ============================================================================
# 3. DEPTH SENSOR INTEGRATION
# ============================================================================

class DepthEstimator:
    """Refine distance estimates using actual depth sensor data."""
    
    def __init__(self, focal_length: float = 600.0, baseline: float = 0.065):
        """
        Args:
            focal_length: Camera focal length in pixels
            baseline: Stereo baseline in meters (for stereo cameras)
        """
        self.focal_length = focal_length
        self.baseline = baseline
    
    def get_distance_from_depth(self, bbox: Tuple[float, float, float, float], depth_map: np.ndarray) -> Optional[float]:
        """
        Extract actual depth from depth map within bounding box.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Clamp to valid range
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1], x2)
        y2 = min(depth_map.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract depth region
        roi = depth_map[y1:y2, x1:x2]
        
        # Filter invalid pixels (0 or very large values)
        valid_depth = roi[(roi > 0.1) & (roi < 50.0)]
        
        if len(valid_depth) == 0:
            return None
        
        # Return median depth
        return float(np.median(valid_depth))
    
    def fallback_distance(self, bbox: Tuple[float, float, float, float], frame_height: int) -> float:
        """Fallback pinhole model if depth unavailable."""
        x1, y1, x2, y2 = bbox
        box_height = max(1.0, y2 - y1)
        # Assume 1.7m average height for person
        distance = (self.focal_length * 1.7) / box_height
        return min(50.0, max(0.5, distance))


# ============================================================================
# 4. OPTICAL FLOW FOR VELOCITY ESTIMATION
# ============================================================================

class OpticalFlowVelocity:
    """Estimate object velocity from optical flow."""
    
    def __init__(self, method: str = 'farneback'):
        self.method = method
        self.prev_gray: Optional[np.ndarray] = None
        self.flow: Optional[np.ndarray] = None
    
    def compute_flow(self, frame: np.ndarray) -> np.ndarray:
        """Compute optical flow between current and previous frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
        
        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        else:
            flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
        
        self.prev_gray = gray
        self.flow = flow
        return flow
    
    def get_roi_velocity(self, bbox: Tuple[float, float, float, float], flow: np.ndarray) -> Tuple[float, float, float]:
        """
        Get average velocity magnitude and direction within bbox region.
        Returns: (velocity_x, velocity_y, magnitude)
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(flow.shape[1], x2)
        y2 = min(flow.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, 0.0
        
        roi_flow = flow[y1:y2, x1:x2]
        
        # Average flow within ROI
        avg_flow_x = np.mean(roi_flow[..., 0])
        avg_flow_y = np.mean(roi_flow[..., 1])
        
        magnitude = np.sqrt(avg_flow_x ** 2 + avg_flow_y ** 2)
        
        return float(avg_flow_x), float(avg_flow_y), float(magnitude)


# ============================================================================
# 5. BETTER THREAT SCORING
# ============================================================================

@dataclass
class ThreatAssessment:
    """Multi-factor threat score."""
    base_score: float        # 0-1, from model confidence
    proximity_score: float   # Weight by distance (closer = more threat)
    velocity_score: float    # Weight by movement speed toward camera
    size_score: float        # Weight by object size relative to frame
    pose_score: float        # Weight by body pose (standing > sitting)
    anomaly_score: float     # Unusual behavior detected
    
    @property
    def total_score(self) -> float:
        """Weighted combination of all factors."""
        return (
            0.30 * self.base_score +       # Confidence
            0.25 * self.proximity_score +  # Proximity (1/distance)
            0.20 * self.velocity_score +   # Speed toward camera
            0.15 * self.size_score +       # Relative size
            0.05 * self.pose_score +       # Body pose
            0.05 * self.anomaly_score      # Anomalous behavior
        )
    
    @property
    def threat_level(self) -> str:
        """Classify threat level."""
        score = self.total_score
        if score < 0.3:
            return "LOW"
        elif score < 0.6:
            return "MEDIUM"
        elif score < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"


class ThreatScorer:
    """Compute threat scores for detected objects."""
    
    def __init__(self):
        self.distance_weights = {
            "person": {"close": (0, 2), "medium": (2, 8), "far": (8, 30)},
            "vehicle": {"close": (0, 5), "medium": (5, 20), "far": (20, 50)},
        }
    
    def compute_threat(
        self,
        class_name: str,
        confidence: float,
        distance: float,
        velocity: float,
        bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int],
        pose_data: Optional[Dict[str, Any]] = None
    ) -> ThreatAssessment:
        """Compute comprehensive threat score."""
        # Base confidence score
        base_score = min(1.0, confidence)
        
        # Proximity score: closer = higher threat
        proximity_score = 1.0 / (1.0 + distance / 2.0)
        proximity_score = min(1.0, proximity_score)
        
        # Velocity score: faster toward camera = higher threat
        velocity_score = min(1.0, velocity / 10.0)
        
        # Size score: larger object = more threatening (vehicle vs toy)
        x1, y1, x2, y2 = bbox
        bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
        frame_area = max(1.0, float(frame_shape[0] * frame_shape[1]))
        size_ratio = bbox_area / frame_area
        size_score = min(1.0, size_ratio * 50)
        
        # Pose score: standing person > sitting person
        pose_score = 0.5
        if pose_data and "action" in pose_data:
            action_threats = {"running": 0.9, "attacking": 1.0, "standing": 0.5, "sitting": 0.2}
            pose_score = action_threats.get(pose_data["action"], 0.5)
        
        anomaly_score = 0.0
        
        return ThreatAssessment(
            base_score=base_score,
            proximity_score=proximity_score,
            velocity_score=velocity_score,
            size_score=size_score,
            pose_score=pose_score,
            anomaly_score=anomaly_score
        )


# ============================================================================
# 6. CONFIDENCE CALIBRATION
# ============================================================================

class ConfidenceCalibrator:
    """Calibrate model confidence to match true probabilities."""
    
    def __init__(self):
        self.calibration_data: List[Tuple[float, float]] = []
    
    def record_prediction(self, predicted_confidence: float, actual_correct: bool):
        """Record prediction for later calibration."""
        self.calibration_data.append((predicted_confidence, float(actual_correct)))
    
    def calibrate(self) -> Any:
        """Fit calibration curve and return calibration function."""
        if len(self.calibration_data) < 10:
            return lambda x: x
        
        confidences, correctness = zip(*self.calibration_data)
        conf_arr = np.array(confidences)
        corr_arr = np.array(correctness)
        
        from sklearn.isotonic import IsotonicRegression
        
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(conf_arr, corr_arr)
        
        return lambda x: float(calibrator.predict([x])[0])


# ============================================================================
# 7. LIGHTWEIGHT ANOMALY DETECTION
# ============================================================================

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


# ============================================================================
# 8. FRAME SKIPPING FOR SPEED
# ============================================================================

class AdaptiveFrameProcessor:
    """Process frames at variable rates based on scene complexity."""
    
    def __init__(self, base_skip: int = 2):
        self.base_skip = base_skip
        self.frame_count = 0
        self.scene_complexity = 0.5
    
    def should_process(self) -> bool:
        """Decide whether to process current frame."""
        self.frame_count += 1
        skip_rate = max(1, int(self.base_skip / (self.scene_complexity + 0.1)))
        return self.frame_count % skip_rate == 0
    
    def update_complexity(self, num_detections: int):
        """Update scene complexity based on detection count."""
        self.scene_complexity = min(1.0, num_detections / 10.0)


# ============================================================================
# 9. BATCH PROCESSING FOR GPU
# ============================================================================

class BatchProcessor:
    """Accumulate frames for batch inference."""
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.frame_buffer: List[np.ndarray] = []
        self.timestamp_buffer: List[float] = []
    
    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Add frame to buffer."""
        self.frame_buffer.append(frame)
        self.timestamp_buffer.append(timestamp)
    
    def get_batch(self) -> Tuple[Optional[List[np.ndarray]], Optional[List[float]]]:
        """Get batch when ready."""
        if len(self.frame_buffer) >= self.batch_size:
            batch = self.frame_buffer[:self.batch_size]
            times = self.timestamp_buffer[:self.batch_size]
            
            self.frame_buffer = self.frame_buffer[self.batch_size:]
            self.timestamp_buffer = self.timestamp_buffer[self.batch_size:]
            
            return batch, times
        
        return None, None
    
    def flush(self) -> Tuple[Optional[List[np.ndarray]], Optional[List[float]]]:
        """Get remaining frames even if batch not full."""
        if len(self.frame_buffer) > 0:
            batch = self.frame_buffer.copy()
            times = self.timestamp_buffer.copy()
            self.frame_buffer = []
            self.timestamp_buffer = []
            return batch, times
        
        return None, None


# ============================================================================
# 10. INTEGRATED OPTIMIZATION PIPELINE
# ============================================================================

class OptimizedDetectionPipeline:
    """Combines all optimizations for production use."""
    
    def __init__(self, model_path: str = "yolo11n.pt", enable_gpu: bool = True):
        from app.services.detector import model_manager, detector
        
        self.model_path = model_path
        self.detector_service = detector
        self.model = model_manager.get_model(model_path)
        
        # Initialize components
        self.smoother = SmoothTrack(smoothing_factor=0.7)
        self.depth_est = DepthEstimator()
        self.optical_flow = OpticalFlowVelocity()
        self.threat_scorer = ThreatScorer()
        self.anomaly_detector = TrajectoryAnomalyDetector()
        self.frame_processor = AdaptiveFrameProcessor()
        self.batch_processor = BatchProcessor(batch_size=4)
        
        self.prev_detections: Dict[int, Any] = {}
        self.track_id_counter = 0
    
    def process_frame(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Process single frame through optimized pipeline.
        
        Returns:
            {
                "detections": [...],
                "latency_ms": float,
                "frame_shape": tuple,
                "num_detections": int
            }
        """
        start_time = time.time()
        
        # 1. Frame skipping for speed
        if not self.frame_processor.should_process():
            return {
                "detections": [],
                "latency_ms": 0.0,
                "frame_shape": frame.shape if frame is not None and hasattr(frame, 'shape') else (0, 0, 0),
                "num_detections": 0,
                "skipped": True
            }
        
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {"detections": [], "latency_ms": 0.0, "num_detections": 0}

        frame_h, frame_w = frame.shape[:2]
        
        # 2. Run YOLO inference
        raw_dets_objects = self.detector_service.detect_frame(frame, depth_map=depth_map, force_inference=True)
        
        raw_detections = []
        for det in raw_dets_objects:
            raw_detections.append({
                "bbox": det.bbox if det.bbox else [10.0, 10.0, 100.0, 100.0],
                "confidence": float(det.confidence),
                "class_name": str(det.class_name),
                "cls_id": 0
            })
        
        # 4. Apply Soft-NMS to remove duplicates
        if raw_detections:
            boxes = [d["bbox"] for d in raw_detections]
            scores = [d["confidence"] for d in raw_detections]
            
            nms_boxes, nms_scores = soft_nms(boxes, scores, iou_threshold=0.5)
            
            nms_dets = []
            for box, score in zip(nms_boxes, nms_scores):
                for det in raw_detections:
                    if det["bbox"] == box and abs(det["confidence"] - score) < 0.01:
                        det["confidence"] = score
                        nms_dets.append(det)
                        break
            if nms_dets:
                raw_detections = nms_dets
        
        # 5. Compute optical flow velocity
        flow = self.optical_flow.compute_flow(frame)
        
        # 6. Build final detections with depth, smoothing, threat scoring
        final_detections = []
        
        for det in raw_detections:
            track_id = self.track_id_counter
            self.track_id_counter += 1
            
            bbox = det["bbox"]
            
            # Get depth (prefer sensor depth if available)
            if depth_map is not None:
                distance = self.depth_est.get_distance_from_depth(bbox, depth_map)
                if distance is None:
                    distance = self.depth_est.fallback_distance(bbox, frame_h)
            else:
                distance = self.depth_est.fallback_distance(bbox, frame_h)
            
            # Apply temporal smoothing
            smooth_bbox = self.smoother.smooth_detection(track_id, bbox)
            
            # Get velocity
            vel_x, vel_y, vel_mag = self.optical_flow.get_roi_velocity(smooth_bbox, flow)
            
            # Update trajectory
            center_x = (smooth_bbox[0] + smooth_bbox[2]) / 2.0
            center_y = (smooth_bbox[1] + smooth_bbox[3]) / 2.0
            self.anomaly_detector.update_trajectory(track_id, (center_x, center_y, distance))
            
            # Check for anomalies
            is_anomalous, anomaly_score = self.anomaly_detector.is_anomalous(track_id)
            
            # Compute threat score
            threat = self.threat_scorer.compute_threat(
                class_name=det["class_name"],
                confidence=det["confidence"],
                distance=distance,
                velocity=vel_mag,
                bbox=smooth_bbox,
                frame_shape=(frame_h, frame_w)
            )
            
            final_detections.append({
                "track_id": track_id,
                "class_name": det["class_name"],
                "confidence": round(det["confidence"], 3),
                "bbox": [round(x, 1) for x in smooth_bbox],
                "distance": round(distance, 2),
                "velocity": {"x": round(vel_x, 2), "y": round(vel_y, 2), "magnitude": round(vel_mag, 2)},
                "threat_score": round(threat.total_score, 3),
                "threat_level": threat.threat_level,
                "is_anomalous": is_anomalous,
                "anomaly_score": round(anomaly_score, 3)
            })
        
        # Update scene complexity
        self.frame_processor.update_complexity(len(final_detections))
        
        elapsed_ms = (time.time() - start_time) * 1000.0
        
        return {
            "detections": final_detections,
            "latency_ms": round(elapsed_ms, 2),
            "frame_shape": frame.shape,
            "num_detections": len(final_detections)
        }
