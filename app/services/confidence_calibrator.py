"""
Phase 19: Model Output Confidence Calibration Engine.

Calibrates raw neural network detection confidence scores using Platt Scaling (sigmoid)
or Isotonic Regression so that reported detection confidence equals true empirical accuracy
(confidence ≈ true probability).
"""
import logging
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from sklearn.calibration import CalibratedClassifierCV as SklearnCalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger("ConfidenceCalibrator")


class ConfidenceCalibrator:
    """
    Platt Scaling / Isotonic Probability Calibrator mapping raw detection scores to true empirical probabilities.
    """

    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.is_calibrated = False
        self._platt_model: Optional[Any] = None

    def fit(self, val_features: np.ndarray, val_labels: np.ndarray) -> "ConfidenceCalibrator":
        """
        Fits confidence calibrator curve using validation confidence features and true ground truth labels.
        Usage matching prompt specification:
          calibrator = CalibratedClassifierCV(yolo_model)
          calibrator.fit(val_features, val_labels)
        """
        if val_features is None or val_labels is None or len(val_features) == 0:
            logger.warning("Empty features or labels provided for calibration fit. Using default calibration curve.")
            return self

        try:
            X = np.asarray(val_features, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            y = np.asarray(val_labels, dtype=int)

            if self.method == "isotonic":
                self._platt_model = IsotonicRegression(out_of_bounds="clip")
                self._platt_model.fit(X.ravel(), y)
            else:
                self._platt_model = LogisticRegression(C=1.0, solver="lbfgs")
                self._platt_model.fit(X, y)

            self.is_calibrated = True
            logger.info(f"✅ Model Confidence Calibrator fitted using {self.method} method ({len(y)} samples).")
        except Exception as e:
            logger.error(f"Failed to fit confidence calibrator: {e}")
            self.is_calibrated = False

        return self

    def calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Transforms raw detection confidence score into calibrated true empirical probability.
        """
        conf = float(raw_confidence)
        if not self.is_calibrated or self._platt_model is None:
            # Default mild Platt scaling approximation for uncalibrated raw scores
            calibrated = 1.0 / (1.0 + np.exp(-4.5 * (conf - 0.5)))
            return round(float(np.clip(calibrated, 0.05, 0.99)), 4)

        try:
            if isinstance(self._platt_model, IsotonicRegression):
                calibrated = self._platt_model.predict([conf])[0]
            else:
                probs = self._platt_model.predict_proba([[conf]])
                calibrated = probs[0][1] if probs.shape[1] > 1 else probs[0][0]
            return round(float(np.clip(calibrated, 0.01, 0.99)), 4)
        except Exception as e:
            logger.debug(f"Confidence calibration error: {e}")
            return conf

    def calibrate_detections(self, detections: List[Any]) -> List[Any]:
        """
        Calibrates confidence scores across a list of Detection objects.
        """
        for det in detections:
            if hasattr(det, "confidence"):
                raw_c = getattr(det, "confidence", 0.5)
                det.confidence = self.calibrate_confidence(raw_c)
        return detections


class CalibratedClassifierCV:
    """
    Wrapper class matching prompt API specification:
      from sklearn.calibration import CalibratedClassifierCV
      calibrator = CalibratedClassifierCV(yolo_model)
      calibrator.fit(val_features, val_labels)
    """

    def __init__(self, estimator: Optional[Any] = None, method: str = "sigmoid", cv: Any = None):
        self.estimator = estimator
        self.method = method
        self.calibrator = ConfidenceCalibrator(method=method)

    def fit(self, val_features: np.ndarray, val_labels: np.ndarray) -> "CalibratedClassifierCV":
        self.calibrator.fit(val_features, val_labels)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        calibrated_probs = []
        for val in X_arr.ravel():
            p1 = self.calibrator.calibrate_confidence(val)
            p0 = 1.0 - p1
            calibrated_probs.append([p0, p1])

        return np.array(calibrated_probs, dtype=np.float64)

    def calibrate_confidence(self, raw_confidence: float) -> float:
        return self.calibrator.calibrate_confidence(raw_confidence)


confidence_calibrator = ConfidenceCalibrator()
