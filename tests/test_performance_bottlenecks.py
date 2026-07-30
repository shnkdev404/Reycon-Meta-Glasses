"""
Unit tests for Performance Bottlenecks & Pipeline Optimizations:
1. Issue 1.1: ModelManager Singleton Pattern & Caching
2. Issue 1.2: Batch Processing (detect_batch)
3. Issue 1.3: Adaptive Resolution Downscaling (640x384)
4. Issue 1.4: Async Thread Pool Execution
"""
import sys
import os
import time
import asyncio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.detector import DetectionEngine, ModelManager, model_manager, _downscale_frame
from app.vision.detector import YOLOWrapper


def test_model_manager_singleton():
    print("--- 1. Testing ModelManager Singleton & Caching ---")
    mgr1 = ModelManager()
    mgr2 = ModelManager()
    assert mgr1 is mgr2, "ModelManager must be a singleton instance!"

    m1 = mgr1.get_model("yolo11n.pt")
    m2 = mgr2.get_model("yolo11n.pt")
    assert m1 is m2, "Model instance must be cached and re-used!"
    print("✅ ModelManager singleton pattern passed! Zero model reload overhead.")


def test_adaptive_resolution_downscaling():
    print("\n--- 2. Testing Adaptive Resolution Downscaling (640x384) ---")
    # High-resolution HD camera frame (1920x1080)
    full_hd = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    downscaled, sx, sy = _downscale_frame(full_hd, target_w=640, target_h=384)
    assert downscaled.shape == (384, 640, 3), f"Expected (384, 640, 3), got {downscaled.shape}"
    assert abs(sx - 3.0) < 0.01, f"Expected scale_x = 3.0, got {sx}"
    assert abs(sy - 2.8125) < 0.01, f"Expected scale_y = 2.8125, got {sy}"
    print(f"✅ Resolution downscaling passed! Frame downscaled from 1920x1080 to {downscaled.shape[1]}x{downscaled.shape[0]}. Scale: ({sx:.2f}x, {sy:.2f}y)")


def test_batch_processing():
    print("\n--- 3. Testing Batch Processing (detect_batch) ---")
    engine = DetectionEngine(confidence_threshold=0.5)
    
    frame1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    t_start = time.perf_counter()
    batch_results = engine.detect_batch([frame1, frame2])
    dt_ms = (time.perf_counter() - t_start) * 1000.0

    assert len(batch_results) == 2, f"Expected 2 batch results, got {len(batch_results)}"
    print(f"✅ Batch processing passed! Batched multi-frame inference execution: {dt_ms:.2f}ms")


def test_async_threadpool_execution():
    print("\n--- 4. Testing Async Thread Pool Execution ---")
    from app.api.websocket import vision_executor, decode_base64_and_detect
    import base64
    import cv2

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", dummy_frame)
    b64_str = base64.b64encode(buffer).decode("utf-8")

    async def run_async_test():
        loop = asyncio.get_running_loop()
        t_start = time.perf_counter()
        dets = await loop.run_in_executor(vision_executor, decode_base64_and_detect, b64_str)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        return dets, dt_ms

    dets, dt_ms = asyncio.run(run_async_test())
    print(f"✅ Async Thread Pool execution passed! Time: {dt_ms:.2f}ms")


if __name__ == "__main__":
    test_model_manager_singleton()
    test_adaptive_resolution_downscaling()
    test_batch_processing()
    test_async_threadpool_execution()
    print("\n🎉 ALL PERFORMANCE OPTIMIZATION TESTS PASSED SUCCESSFULLY!")
