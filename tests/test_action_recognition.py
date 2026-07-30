"""
Unit tests for Temporal Action Recognition Engine:
1. ActionRecognitionEngine 8-frame sliding buffer.
2. Action classification (standing_still, walking, running, attacking).
3. Hazardous action flag detection.
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision import ActionRecognitionEngine


def test_action_recognition_still_and_walking():
    print("--- 1. Testing Action Recognition (Standing Still vs Walking) ---")
    engine = ActionRecognitionEngine(buffer_size=8)

    # 1. Standing Still (8 identical static frames)
    static_frame = np.zeros((384, 640, 3), dtype=np.uint8)
    cv2.rectangle(static_frame, (100, 100), (200, 300), (255, 255, 255), -1)

    for _ in range(8):
        engine.add_frame(static_frame)

    res_still = engine.classify_action()
    assert res_still["action"] == "standing_still", f"Expected standing_still, got {res_still['action']}"
    assert res_still["is_hazardous_action"] is False
    print(f"✅ Standing still test passed! Action: '{res_still['action']}', Hazard: {res_still['is_hazardous_action']}")

    # 2. Walking sequence (8 slowly moving frames)
    engine_walk = ActionRecognitionEngine(buffer_size=8)
    for i in range(8):
        f = np.zeros((384, 640, 3), dtype=np.uint8)
        shift = i * 4  # 4-px shift for walking
        cv2.rectangle(f, (100 + shift, 100), (200 + shift, 300), (255, 255, 255), -1)
        engine_walk.add_frame(f)

    res_walk = engine_walk.classify_action()
    assert res_walk["action"] in ["walking", "running"], f"Expected walking/running, got {res_walk['action']}"
    print(f"✅ Walking sequence test passed! Action: '{res_walk['action']}', Motion intensity: {res_walk['motion_intensity']}")


def test_action_recognition_running_and_attacking():
    print("\n--- 2. Testing Action Recognition (Running / Aggressive Threat Motion) ---")
    engine_run = ActionRecognitionEngine(buffer_size=8)

    # 8 rapidly moving / fluctuating frames (simulating fast approach / attack motion)
    for i in range(8):
        f = np.zeros((384, 640, 3), dtype=np.uint8)
        shift = (i % 2) * 80  # Rapid alternating high-amplitude motion
        cv2.rectangle(f, (100 + shift, 100), (250 + shift, 350), (255, 255, 255), -1)
        engine_run.add_frame(f)

    res_threat = engine_run.classify_action()
    assert res_threat["is_hazardous_action"] is True, f"Expected hazardous action flag, got {res_threat}"
    assert res_threat["action"] in ["running", "attacking"]
    print(f"✅ Fast threat motion test passed! Action: '{res_threat['action']}', Hazard Flag: {res_threat['is_hazardous_action']}")


def test_person_bboxes_action_detection():
    print("\n--- 3. Testing Person Bounding Box Action Detection ---")
    engine = ActionRecognitionEngine(buffer_size=8)
    frame = np.zeros((384, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 300), (255, 255, 255), -1)

    bboxes = [[100, 100, 200, 300], [400, 150, 500, 350]]
    person_actions = engine.detect_person_actions(frame, bboxes)

    assert len(person_actions) == 2, f"Expected 2 person action results, got {len(person_actions)}"
    assert person_actions[0]["bbox_index"] == 0
    assert person_actions[1]["bbox_index"] == 1
    print(f"✅ Person bboxes action detection passed! Processed {len(person_actions)} person tracks.")


if __name__ == "__main__":
    test_action_recognition_still_and_walking()
    test_action_recognition_running_and_attacking()
    test_person_bboxes_action_detection()
    print("\n🎉 ALL ACTION RECOGNITION TESTS PASSED SUCCESSFULLY!")
