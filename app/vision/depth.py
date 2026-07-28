"""
Phase 3: Monocular Depth Estimation Interface & Wrappers.

Abstract contract & wrappers for monocular/binaural depth estimation.
TODO: Connect real depth neural networks (Depth Anything V2, MiDaS, ZoeDepth).
"""
from abc import ABC, abstractmethod
from typing import Any, Tuple
from app.models.object import BoundingBox2D


class BaseDepthEstimator(ABC):
    """Abstract contract for monocular depth estimation engines."""

    @abstractmethod
    def estimate_depth(self, frame: Any, bbox: BoundingBox2D) -> float:
        """Estimate 3D metric distance (in meters) for an object in a bounding box."""
        pass


class DepthEstimatorWrapper(BaseDepthEstimator):
    """
    Depth estimation wrapper compatible with Depth Anything V2, MiDaS, ZoeDepth,
    and pinhole monocular camera geometry.
    """

    # Estimated real-world reference heights in meters for standard object classes
    REFERENCE_HEIGHTS = {
        "person": 1.7,
        "vehicle": 1.5,
        "car": 1.5,
        "truck": 2.5,
        "forklift": 2.2,
        "bicycle": 1.0,
        "dog": 0.5,
    }

    def __init__(self, model_type: str = "DepthAnythingV2", focal_length_px: float = 800.0):
        self.model_type = model_type
        self.focal_length_px = focal_length_px
        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        """Attempts to initialize torch / DepthAnything model weights if available."""
        try:
            import torch
            # Optional neural depth estimation initialization
            self._model = None
        except Exception:
            self._model = None

    def estimate_depth(self, frame: Any, bbox: BoundingBox2D, label: str = "person") -> float:
        """
        Compute metric depth in meters from frame depth map or pinhole camera geometry.
        """
        # Step 1: If frame contains a depth map buffer (from Phase 2 Depth Sensor)
        if isinstance(frame, dict) and "depth_map" in frame:
            depth_map = frame["depth_map"]
            if hasattr(depth_map, "__getitem__") and bbox:
                cx = int((bbox.xmin + bbox.xmax) / 2.0)
                cy = int((bbox.ymin + bbox.ymax) / 2.0)
                try:
                    depth_val = float(depth_map[cy][cx])
                    if depth_val > 0.0:
                        return round(depth_val, 2)
                except Exception:
                    pass

        # Step 2: Monocular pinhole geometry fallback: depth = (f * H_real) / h_pixel
        if bbox:
            box_h = max(1.0, bbox.ymax - bbox.ymin)
            ref_h = self.REFERENCE_HEIGHTS.get(label.lower(), 1.5)
            estimated_depth = (self.focal_length_px * ref_h) / box_h
            return max(0.5, round(estimated_depth, 2))

        return 5.0

