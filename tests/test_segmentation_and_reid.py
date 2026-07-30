"""
Unit tests for Semantic/Instance Segmentation & Person Re-ID (ReID):
1. Instance Segmentation mask extraction (YOLOSegmentationEngine)
2. 512-D Person Re-ID feature vector extraction (PersonReIDExtractor)
3. Cosine similarity matching across camera viewpoints (compute_cosine_similarity)
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision import YOLOSegmentationEngine, PersonReIDExtractor, compute_cosine_similarity


def test_instance_segmentation():
    print("--- 1. Testing Instance Segmentation (yolo11n-seg.pt) ---")
    seg_engine = YOLOSegmentationEngine(confidence_threshold=0.5)
    
    # Process test synthetic frame (480x640x3)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    segments = seg_engine.segment(dummy_frame)

    assert len(segments) > 0, "Instance segmentation returned no segments!"
    first_seg = segments[0]
    assert "label" in first_seg
    assert "confidence" in first_seg
    assert "bbox" in first_seg
    assert "mask_polygon" in first_seg
    assert "area_px" in first_seg
    print(f"✅ Instance segmentation passed! Detected {len(segments)} objects with masks. First label: '{first_seg['label']}'")


def test_person_reid_feature_extraction_and_cosine_matching():
    print("\n--- 2. Testing Person Re-ID (512-D Feature Extraction & Cosine Matching) ---")
    reid = PersonReIDExtractor()

    # Synthetic frame 1 (Camera Viewpoint A)
    frame_a = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame_a, (100, 100), (250, 400), (0, 0, 255), -1)  # Red coat person
    bbox_a = [100, 100, 250, 400]

    # Synthetic frame 2 (Camera Viewpoint B - Same Person)
    frame_b = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame_b, (120, 90), (270, 390), (0, 0, 255), -1)  # Same red coat person
    bbox_b = [120, 90, 270, 390]

    # Synthetic frame 3 (Camera Viewpoint B - Different Person)
    frame_c = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame_c, (50, 50), (200, 350), (255, 0, 0), -1)  # Blue coat person
    bbox_c = [50, 50, 200, 350]

    # Extract 512-D feature embeddings
    feat_a = reid.extract_features(frame_a, bbox_a)
    feat_b = reid.extract_features(frame_b, bbox_b)
    feat_c = reid.extract_features(frame_c, bbox_c)

    assert feat_a.shape == (512,), f"Expected 512-D vector, got {feat_a.shape}"
    assert feat_b.shape == (512,)
    assert feat_c.shape == (512,)

    # Verify unit L2 normalization
    norm_a = float(np.linalg.norm(feat_a))
    assert abs(norm_a - 1.0) < 1e-3, f"Expected unit L2 norm (1.0), got {norm_a}"

    # Compute cosine similarity
    sim_same = compute_cosine_similarity(feat_a, feat_b)
    sim_diff = compute_cosine_similarity(feat_a, feat_c)

    print(f"  Similarity (Same Person across Viewpoints A & B): {sim_same:.4f}")
    print(f"  Similarity (Different Person across Viewpoints A & C): {sim_diff:.4f}")

    assert sim_same > sim_diff, "Same person across viewpoints must have higher cosine similarity than different persons!"
    
    # Test identity matching
    candidates = {"person_101": feat_b, "person_102": feat_c}
    matched = reid.match_person_identity(feat_a, candidates, similarity_threshold=0.60)
    assert matched is not None, "Failed to match person identity across viewpoints!"
    assert matched[0] == "person_101", f"Expected match person_101, got {matched[0]}"
    print(f"✅ Person Re-ID matching passed! Matched target to '{matched[0]}' with similarity {matched[1]:.4f}")


if __name__ == "__main__":
    test_instance_segmentation()
    test_person_reid_feature_extraction_and_cosine_matching()
    print("\n🎉 ALL SEGMENTATION & RE-ID TESTS PASSED SUCCESSFULLY!")
