"""
Unit tests for Advanced Perception Suite:
1. Panoptic Segmentation (PanopticSegmentationEngine)
2. 3D Gaze Estimation & Eye Contact Threat (GazeEstimationEngine)
3. 17-Keypoint Body Pose Estimation (PoseEstimationEngine)
4. Multi-Modal Vision + Audio (MFCC) + IMU Fusion (MultiModalFusionEngine)
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision import PanopticSegmentationEngine, GazeEstimationEngine, PoseEstimationEngine
from app.services.multimodal_fusion import MultiModalFusionEngine, AudioThreatDetector, IMUPatternAnalyzer


def test_panoptic_segmentation():
    print("--- 1. Testing Panoptic Segmentation (Things + Stuff + Crowd) ---")
    engine = PanopticSegmentationEngine(confidence_threshold=0.5)
    frame = np.zeros((384, 640, 3), dtype=np.uint8)

    res = engine.segment_panoptic(frame)
    assert "instances" in res
    assert "semantic_stuff" in res
    assert "crowd_density" in res
    assert "is_crowded_scene" in res

    print(f"✅ Panoptic segmentation passed! Instances: {len(res['instances'])}, Crowd Density: {res['crowd_density'] * 100:.1f}%")


def test_3d_gaze_estimation():
    print("\n--- 2. Testing 3D Gaze Estimation & Eye Contact Threat ---")
    gaze_engine = GazeEstimationEngine(eye_contact_threshold_deg=15.0)
    frame = np.zeros((384, 640, 3), dtype=np.uint8)
    
    # Person at center (looking directly at camera)
    bbox_center = [270.0, 100.0, 370.0, 300.0]
    gaze_res = gaze_engine.estimate_gaze(frame, bbox_center)

    assert "gaze_vector" in gaze_res
    assert len(gaze_res["gaze_vector"]) == 3
    assert gaze_res["is_eye_contact_threat"] is True, f"Center gaze should be eye contact threat! Got {gaze_res}"

    # Person far off-center (looking away)
    bbox_off = [50.0, 100.0, 150.0, 300.0]
    gaze_off = gaze_engine.estimate_gaze(frame, bbox_off)
    assert gaze_off["is_eye_contact_threat"] is False
    print(f"✅ 3D gaze estimation passed! Center Gaze vector: {gaze_res['gaze_vector']}, Direct Eye Contact: {gaze_res['is_eye_contact_threat']}")


def test_17_keypoint_pose_estimation():
    print("\n--- 3. Testing 17-Keypoint Body Pose Estimation ---")
    pose_engine = PoseEstimationEngine(confidence_threshold=0.5)
    frame = np.zeros((384, 640, 3), dtype=np.uint8)

    poses = pose_engine.estimate_pose(frame)
    assert len(poses) > 0, "Pose estimation returned no poses!"
    first_pose = poses[0]

    assert "keypoints" in first_pose
    assert len(first_pose["keypoints"]) == 17, f"Expected 17 keypoints, got {len(first_pose['keypoints'])}"
    assert "posture" in first_pose
    assert "is_threat_posture" in first_pose

    print(f"✅ 17-Keypoint pose estimation passed! Extracted {len(first_pose['keypoints'])} COCO keypoints. Posture: '{first_pose['posture']}'")


def test_multimodal_vision_audio_imu_fusion():
    print("\n--- 4. Testing Multi-Modal Fusion (Vision + Audio MFCC + IMU) ---")
    fusion_engine = MultiModalFusionEngine()

    # Synthetic audio signal (loud acoustic event > 85dB scream/siren)
    audio_buffer = np.sin(np.linspace(0, 500, 16000)) * 0.8
    # Wearable IMU acceleration reading (running / motion spike: 15.0 m/s^2)
    imu_reading = {"ax": 8.0, "ay": 10.0, "az": 9.81}
    vision_threat_score = 0.80

    fused_res = fusion_engine.fuse_multimodal_perception(
        vision_threat_score=vision_threat_score,
        audio_buffer=audio_buffer,
        imu_reading=imu_reading
    )

    assert "fused_threat_level" in fused_res
    assert "multi_modal_score" in fused_res
    assert fused_res["fused_threat_level"] in ["HIGH", "CRITICAL"]
    assert fused_res["multi_modal_score"] >= 0.70

    print(f"✅ Multi-modal perception fusion passed! Fused Level: '{fused_res['fused_threat_level']}', Score: {fused_res['multi_modal_score']}")


if __name__ == "__main__":
    test_panoptic_segmentation()
    test_3d_gaze_estimation()
    test_17_keypoint_pose_estimation()
    test_multimodal_vision_audio_imu_fusion()
    print("\n🎉 ALL ADVANCED MULTI-MODAL PERCEPTION TESTS PASSED SUCCESSFULLY!")
