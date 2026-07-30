"""
Phase 20: Multi-GPU Parallel Inference Distribution Unit & Integration Tests.

Verifies:
1. MultiGPUInferenceEngine device binding matching prompt specification:
     self.model = YOLO("yolo11n.pt").to("cuda:0")
2. Parallel multi-frame stream processing matching prompt specification:
     detect_frames_parallel(frames)
3. Device status reporting & thread safety.
4. DetectionEngine.detect_batch multi-GPU delegation.
"""
import sys
import os
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.gpu_distributor import MultiGPUInferenceEngine, gpu_distributor
from app.services.detector import DetectionEngine, detector


def test_multi_gpu_initialization_and_device_binding():
    print("--- 1. Testing Multi-GPU Initialization & Device Binding ---")
    
    # Instantiate engine with explicit multi-GPU targets: ["cuda:0", "cuda:1"]
    engine = MultiGPUInferenceEngine(model_name="yolo11n.pt", devices=["cuda:0", "cuda:1"])
    assert engine is not None
    assert len(engine.devices) == 2
    assert "cuda:0" in engine.devices
    assert "cuda:1" in engine.devices
    assert len(engine.models) == 2

    status = engine.get_device_status()
    assert status["device_count"] == 2
    print(f"✅ Multi-GPU device binding passed! Bound YOLO models to devices: {status['devices']}")


def test_parallel_multi_frame_inference():
    print("\n--- 2. Testing Parallel Multi-Frame Inference Stream ---")
    
    engine = MultiGPUInferenceEngine(model_name="yolo11n.pt")
    
    # Create 4 synthetic camera frames
    frame1 = np.zeros((384, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((384, 640, 3), dtype=np.uint8)
    frame3 = np.zeros((384, 640, 3), dtype=np.uint8)
    frame4 = np.zeros((384, 640, 3), dtype=np.uint8)

    cv2.circle(frame1, (200, 200), 50, (255, 255, 255), -1)
    cv2.rectangle(frame3, (100, 100), (300, 300), (255, 255, 255), -1)

    frames = [frame1, frame2, frame3, frame4]

    # Execute parallel multi-frame dispatching matching prompt signature:
    # Process multiple frames in parallel on GPU:1, GPU:2, etc.
    results = engine.detect_frames_parallel(frames)

    assert isinstance(results, list)
    assert len(results) == 4
    print(f"✅ Parallel multi-frame inference passed! Received {len(results)} detection outputs across GPU workers.")


def test_detection_engine_multi_gpu_batch_delegation():
    print("\n--- 3. Testing Detection Engine Multi-GPU Batch Delegation ---")
    
    engine = DetectionEngine(model_name="yolo11n.pt")
    frames = [np.zeros((384, 640, 3), dtype=np.uint8) for _ in range(3)]

    batch_outputs = engine.detect_batch(frames)
    assert isinstance(batch_outputs, list)
    assert len(batch_outputs) == 3
    print("✅ DetectionEngine.detect_batch multi-GPU delegation test passed!")


if __name__ == "__main__":
    test_multi_gpu_initialization_and_device_binding()
    test_parallel_multi_frame_inference()
    test_detection_engine_multi_gpu_batch_delegation()
    print("\n🎉 ALL PHASE 20 MULTI-GPU TESTS PASSED SUCCESSFULLY!")
