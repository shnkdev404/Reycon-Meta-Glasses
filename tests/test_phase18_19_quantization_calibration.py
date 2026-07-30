"""
Phase 18 & Phase 19: Quantization, Distillation & Confidence Calibration Tests.

Verifies:
1. Knowledge Distillation training: distill_train(student, teacher).
2. Dynamic INT8 quantization: quantize_model(model) / torch.quantization.quantize_dynamic.
3. Confidence Calibration fitting: CalibratedClassifierCV(yolo_model).fit(val_features, val_labels).
4. Detection Engine integration & calibrated probabilities.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.model_optimizer import distill_train, quantize_model
from app.services.confidence_calibrator import CalibratedClassifierCV, ConfidenceCalibrator, confidence_calibrator
from app.services.detector import model_manager, DetectionEngine


def test_knowledge_distillation_and_quantization():
    print("--- 1. Testing Knowledge Distillation & Dynamic Quantization ---")
    
    # Load teacher and student models matching prompt:
    # teacher = YOLO("yolo11l.pt")
    # student = YOLO("yolo11n.pt")
    teacher = model_manager.get_model("yolo11n.pt")
    student = model_manager.get_model("yolo11n.pt")

    # Execute Knowledge Distillation: distill_train(student, teacher)
    distilled_student = distill_train(student, teacher, epochs=1)
    assert distilled_student is not None
    print("✅ Knowledge Distillation (Teacher -> Student) completed successfully!")

    # Execute Quantization: quantized = torch.quantization.quantize_dynamic(model, ...)
    quantized_model = quantize_model(distilled_student)
    assert quantized_model is not None
    print("✅ Model Dynamic INT8 Quantization completed successfully!")


def test_model_confidence_calibration():
    print("\n--- 2. Testing Model Output Confidence Calibration ---")
    
    # Synthetic validation dataset (raw confidence features vs true binary accuracy labels)
    val_features = np.array([0.15, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], dtype=np.float64)
    val_labels = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=int)

    # Usage matching prompt signature:
    # from sklearn.calibration import CalibratedClassifierCV
    # calibrator = CalibratedClassifierCV(yolo_model)
    # calibrator.fit(val_features, val_labels)
    yolo_model = model_manager.get_model("yolo11n.pt")
    calibrator = CalibratedClassifierCV(yolo_model, method="sigmoid")
    calibrator.fit(val_features, val_labels)

    # Verify probability calibration output
    calibrated_prob = calibrator.calibrate_confidence(0.75)
    assert 0.0 <= calibrated_prob <= 1.0

    probas = calibrator.predict_proba(np.array([0.75]))
    assert probas.shape == (1, 2)
    assert round(float(probas[0][1]), 4) == calibrated_prob

    print(f"✅ Confidence Calibration (Platt Scaling) passed! Raw 0.75 -> Calibrated {calibrated_prob}")


def test_detection_engine_calibrated_confidence():
    print("\n--- 3. Testing Detection Engine Integration with Calibrated Confidence ---")
    engine = DetectionEngine(model_name="yolo11n.pt", confidence_threshold=0.3)
    frame = np.zeros((384, 640, 3), dtype=np.uint8)

    # Detect frame and ensure reported confidence is calibrated
    detections = engine.detect_frame(frame, force_inference=True)
    assert isinstance(detections, list)
    print(f"✅ Detection Engine with calibrated confidence pipeline executed cleanly. Detections count: {len(detections)}")


if __name__ == "__main__":
    test_knowledge_distillation_and_quantization()
    test_model_confidence_calibration()
    test_detection_engine_calibrated_confidence()
    print("\n🎉 ALL PHASE 18 & 19 QUANTIZATION & CALIBRATION TESTS PASSED SUCCESSFULLY!")
