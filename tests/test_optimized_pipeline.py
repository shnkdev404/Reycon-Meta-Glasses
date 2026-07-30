"""
Optimized Multi-Stage Detection Pipeline Unit & Integration Tests.

Verifies:
1. Instantiation of OptimizedDetectionPipeline components: ByteTrack, KalmanFilter, DepthSensor, PoseEstimator, OpticalFlow.
2. Multi-frame batch processing: process_frame_batch(frames_batch) on 4-8 frame inputs.
3. Output dictionary fields: detections, poses, velocities, threat_scores, latency_ms.
4. Execution latency timing.
"""
import sys
import os
import asyncio
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.optimized_pipeline import OptimizedDetectionPipeline, ByteTrack, KalmanFilter, DepthSensor, PoseEstimator, OpticalFlow


def test_pipeline_component_instantiation():
    print("--- 1. Testing Optimized Pipeline Component Instantiation ---")
    pipeline = OptimizedDetectionPipeline()
    assert pipeline is not None
    assert isinstance(pipeline.tracker, ByteTrack)
    assert isinstance(pipeline.kalman, KalmanFilter)
    assert isinstance(pipeline.depth_sensor, DepthSensor)
    assert isinstance(pipeline.pose_model, PoseEstimator)
    assert isinstance(pipeline.flow, OpticalFlow)
    print("✅ All pipeline components (ByteTrack, Kalman, Depth, Pose, OpticalFlow) instantiated successfully!")


def test_async_process_frame_batch():
    print("\n--- 2. Testing Async Multi-Frame Batch Processing ---")
    pipeline = OptimizedDetectionPipeline()

    # Create synthetic 4-frame batch
    frames_batch = []
    for i in range(4):
        frame = np.zeros((384, 640, 3), dtype=np.uint8)
        # Add visual shapes
        cv2.circle(frame, (150 + i * 20, 150), 40, (0, 255, 0), -1)
        cv2.rectangle(frame, (300, 200), (400, 300), (0, 0, 255), -1)
        frames_batch.append(frame)

    # Run async pipeline matching prompt signature:
    # result = await pipeline.process_frame_batch(frames_batch)
    result = asyncio.run(pipeline.process_frame_batch(frames_batch))

    assert isinstance(result, dict)
    assert "detections" in result
    assert "poses" in result
    assert "velocities" in result
    assert "threat_scores" in result
    assert "latency_ms" in result

    assert len(result["detections"]) == 4
    assert len(result["poses"]) == 4
    assert len(result["velocities"]) == 4
    assert len(result["threat_scores"]) == 4
    assert result["latency_ms"] >= 0.0

    print(f"✅ process_frame_batch passed! Processed 4 frames in {result['latency_ms']} ms.")
    print(f"   Outputs -> Detections: {len(result['detections'])}, Poses: {len(result['poses'])}, Velocities: {len(result['velocities'])}, Threat Scores: {len(result['threat_scores'])}")


if __name__ == "__main__":
    test_pipeline_component_instantiation()
    test_async_process_frame_batch()
    print("\n🎉 ALL OPTIMIZED MULTI-STAGE PIPELINE TESTS PASSED SUCCESSFULLY!")
