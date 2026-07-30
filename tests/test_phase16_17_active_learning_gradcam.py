"""
Phase 16 & Phase 17: Active Learning & Grad-CAM Visualization Unit & Integration Tests.

Tests:
1. Active Learning & Hard Example Mining logging when 0.3 < confidence < 0.5:
     if 0.3 < confidence < 0.5:
         save_hard_example(frame, bbox, class_name)
2. Manifest JSON tracking and cropped image generation.
3. Grad-CAM visual feature attribution map generation matching prompt API specification:
     cam = GradCAM(model=yolo_model)
     attribution_map = cam(frame)
     visualize_attribution(frame, attribution_map)
"""
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.active_learning import HardExampleMiner, save_hard_example, hard_example_miner
from app.services.cam_visualizer import GradCAM, visualize_attribution
from app.services.detector import model_manager


def test_active_learning_hard_example_mining():
    print("--- 1. Testing Active Learning & Hard Example Mining (0.3 < confidence < 0.5) ---")
    miner = HardExampleMiner(output_dir="data/hard_examples_test")
    miner.clear_hard_examples()

    frame = np.zeros((384, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (250, 250), (0, 255, 0), -1)

    # Test prompt signature: if 0.3 < confidence < 0.5: save_hard_example(frame, bbox, class_name)
    confidence = 0.42
    bbox = [100.0, 100.0, 250.0, 250.0]
    class_name = "person"

    if 0.3 < confidence < 0.5:
        record = miner.save_hard_example(frame, bbox, class_name, confidence)

    assert record is not None
    assert record["class_name"] == "person"
    assert record["confidence"] == 0.42
    assert os.path.exists(record["crop_path"])
    assert os.path.exists(record["full_path"])

    hard_examples = miner.list_hard_examples()
    assert len(hard_examples) == 1
    assert hard_examples[0]["example_id"] == record["example_id"]

    print(f"✅ Active Learning Hard Example Mining passed! Saved record '{record['example_id']}' to '{record['crop_path']}'.")
    miner.clear_hard_examples()


def test_grad_cam_and_attribution_visualization():
    print("\n--- 2. Testing Grad-CAM Explainability & Heatmap Visualization ---")
    
    yolo_model = model_manager.get_model("yolo11n.pt")
    
    # Instantiate matching prompt: cam = GradCAM(model=yolo_model)
    cam = GradCAM(model=yolo_model)
    assert cam is not None

    frame = np.zeros((384, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 192), 60, (255, 255, 255), -1)

    # Execute matching prompt: attribution_map = cam(frame)
    attribution_map = cam(frame)
    assert attribution_map is not None
    assert isinstance(attribution_map, np.ndarray)
    assert attribution_map.shape[:2] == (384, 640)
    assert 0.0 <= np.min(attribution_map) <= np.max(attribution_map) <= 1.0

    # Execute matching prompt: visualize_attribution(frame, attribution_map)
    visualization = visualize_attribution(frame, attribution_map)
    assert visualization is not None
    assert isinstance(visualization, np.ndarray)
    assert visualization.shape == (384, 640, 3)

    print(f"✅ Grad-CAM & Attribution Visualization passed! Attribution map shape: {attribution_map.shape}, Visualized output shape: {visualization.shape}")


if __name__ == "__main__":
    test_active_learning_hard_example_mining()
    test_grad_cam_and_attribution_visualization()
    print("\n🎉 ALL PHASE 16 & 17 ACTIVE LEARNING & GRAD-CAM TESTS PASSED SUCCESSFULLY!")
