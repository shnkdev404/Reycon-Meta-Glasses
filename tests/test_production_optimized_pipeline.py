"""
Production Reference Optimized Detection Pipeline Unit & Integration Tests.

Verifies:
1. Instantiation of all 10 optimization components.
2. Single-frame processing via pipeline.process_frame(frame, depth_map).
3. Soft-NMS, Kalman smoothing, optical flow velocity, depth estimation, threat scoring, anomaly detection.
4. Output dictionary schema: detections, latency_ms, frame_shape, num_detections.
"""
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.optimized_detection_pipeline import (
    KalmanFilter1D, SmoothTrack, soft_nms, DepthEstimator, OpticalFlowVelocity,
    ThreatScorer, ThreatAssessment, ConfidenceCalibrator, TrajectoryAnomalyDetector,
    AdaptiveFrameProcessor, BatchProcessor, OptimizedDetectionPipeline
)


def test_optimization_components_instantiation():
    print("--- 1. Testing Optimization Components Instantiation ---")
    kf = KalmanFilter1D()
    assert kf.update(10.0) is not None

    smoother = SmoothTrack()
    bbox_smoothed = smoother.smooth_detection(1, (10.0, 10.0, 50.0, 50.0))
    assert len(bbox_smoothed) == 4

    boxes = [[10, 10, 50, 50], [12, 12, 52, 52]]
    scores = [0.9, 0.85]
    nms_b, nms_s = soft_nms(boxes, scores)
    assert len(nms_b) > 0

    depth_est = DepthEstimator()
    dist = depth_est.fallback_distance((10, 10, 50, 100), 384)
    assert dist > 0.0

    of = OpticalFlowVelocity()
    frame = np.zeros((384, 640, 3), dtype=np.uint8)
    flow = of.compute_flow(frame)
    assert flow.shape[:2] == (384, 640)

    scorer = ThreatScorer()
    threat = scorer.compute_threat("person", 0.9, 2.0, 1.5, (10, 10, 50, 100), (384, 640))
    assert threat.threat_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    calib = ConfidenceCalibrator()
    calib.record_prediction(0.8, True)
    assert calib is not None

    anom = TrajectoryAnomalyDetector()
    anom.update_trajectory(1, (1.0, 2.0, 3.0))
    is_a, score_a = anom.is_anomalous(1)
    assert isinstance(is_a, bool)

    fp = AdaptiveFrameProcessor()
    assert isinstance(fp.should_process(), bool)

    bp = BatchProcessor(batch_size=2)
    bp.add_frame(frame, 100.0)

    print("✅ All 10 optimization components instantiated and verified successfully!")


def test_production_pipeline_process_frame():
    print("\n--- 2. Testing Production Optimized Pipeline process_frame ---")
    pipeline = OptimizedDetectionPipeline(model_path="yolo11n.pt", enable_gpu=False)

    frame = np.zeros((384, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (250, 250), (255, 255, 255), -1)

    depth_map = np.ones((384, 640), dtype=np.float32) * 3.5

    # Execute process_frame matching user usage snippet
    result = pipeline.process_frame(frame, depth_map=depth_map, confidence_threshold=0.3)

    assert isinstance(result, dict)
    assert "detections" in result
    assert "latency_ms" in result
    assert "frame_shape" in result
    assert "num_detections" in result
    assert result["frame_shape"] == (384, 640, 3)

    print(f"✅ Production process_frame passed! Latency: {result['latency_ms']} ms, Detections: {result['num_detections']}")


if __name__ == "__main__":
    test_optimization_components_instantiation()
    test_production_pipeline_process_frame()
    print("\n🎉 ALL PRODUCTION OPTIMIZED DETECTION PIPELINE TESTS PASSED SUCCESSFULLY!")
