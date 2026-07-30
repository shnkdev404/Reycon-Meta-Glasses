"""
Phase 16: Active Learning & Hard Example Mining Engine.

Logs low-confidence and uncertain predictions (0.3 < confidence < 0.5) to disk
for offline dataset curation, hard example mining, and model retraining.
"""
import os
import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("ActiveLearning")


class HardExampleMiner:
    """
    Hard Example Miner logging uncertain bounding box predictions and cropped frame images.
    """

    def __init__(self, output_dir: str = "data/hard_examples", min_conf: float = 0.3, max_conf: float = 0.5):
        self.output_dir = output_dir
        self.images_dir = os.path.join(self.output_dir, "images")
        self.annotations_dir = os.path.join(self.output_dir, "annotations")
        self.manifest_file = os.path.join(self.output_dir, "hard_examples_manifest.json")
        
        self.min_conf = min_conf
        self.max_conf = max_conf
        self.hard_examples: List[Dict[str, Any]] = []

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        self._load_manifest()

    def _load_manifest(self):
        """Load manifest list from disk if exists."""
        try:
            if os.path.exists(self.manifest_file):
                with open(self.manifest_file, "r") as f:
                    self.hard_examples = json.load(f)
                logger.info(f"📂 Loaded {len(self.hard_examples)} hard examples from manifest.")
        except Exception as e:
            logger.error(f"Error loading hard examples manifest: {e}")

    def _save_manifest(self):
        """Save updated manifest list to disk."""
        try:
            with open(self.manifest_file, "w") as f:
                json.dump(self.hard_examples, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving hard examples manifest: {e}")

    def save_hard_example(
        self,
        frame: np.ndarray,
        bbox: List[float],
        class_name: str,
        confidence: float,
        reason: str = "uncertain_prediction"
    ) -> Optional[Dict[str, Any]]:
        """
        Saves hard example frame crop, full image, annotation metadata, and updates manifest.
        Usage pattern matching user prompt:
          if 0.3 < confidence < 0.5:
              save_hard_example(frame, bbox, class_name)
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        example_id = f"hex_{uuid.uuid4().hex[:8]}"
        now_ts = time.time()

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1_c, x2_c = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1_c, y2_c = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))

        # Save cropped ROI image
        crop_filename = f"{example_id}_crop.jpg"
        crop_path = os.path.join(self.images_dir, crop_filename)
        
        if x2_c > x1_c and y2_c > y1_c:
            crop_img = frame[y1_c:y2_c, x1_c:x2_c]
            cv2.imwrite(crop_path, crop_img)
        else:
            cv2.imwrite(crop_path, frame)

        # Save full annotated image frame
        full_filename = f"{example_id}_full.jpg"
        full_path = os.path.join(self.images_dir, full_filename)
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1_c, y1_c), (x2_c, y2_c), (0, 165, 255), 2)
        cv2.putText(annotated, f"{class_name}: {confidence:.2f}", (x1_c, max(15, y1_c - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.imwrite(full_path, annotated)

        # Record metadata JSON annotation
        record = {
            "example_id": example_id,
            "class_name": class_name,
            "confidence": round(float(confidence), 4),
            "bbox": [x1, y1, x2, y2],
            "frame_width": w,
            "frame_height": h,
            "crop_path": crop_path,
            "full_path": full_path,
            "reason": reason,
            "timestamp": now_ts
        }

        ann_file = os.path.join(self.annotations_dir, f"{example_id}.json")
        with open(ann_file, "w") as f:
            json.dump(record, f, indent=2)

        self.hard_examples.append(record)
        self._save_manifest()
        logger.info(f"💾 Hard Example mined: '{class_name}' (conf: {confidence:.2f}) saved to '{crop_path}'.")
        return record

    def evaluate_and_log_detections(self, frame: np.ndarray, raw_detections: List[Any]) -> List[Dict[str, Any]]:
        """
        Scans raw detection candidates and automatically logs any hard examples
        where 0.3 < confidence < 0.5.
        """
        mined = []
        for det in raw_detections:
            conf = float(getattr(det, "confidence", 0.0))
            if self.min_conf < conf < self.max_conf:
                bbox = getattr(det, "bbox", [0, 0, 100, 100])
                cls_name = getattr(det, "class_name", getattr(det, "label", "unknown"))
                record = self.save_hard_example(frame, bbox, cls_name, conf)
                if record:
                    mined.append(record)
        return mined

    def list_hard_examples(self) -> List[Dict[str, Any]]:
        """Return all logged hard examples."""
        return self.hard_examples

    def clear_hard_examples(self):
        """Clear all logged hard examples."""
        self.hard_examples.clear()
        self._save_manifest()


hard_example_miner = HardExampleMiner()

# Alias wrapper function matching user prompt: save_hard_example(frame, bbox, class_name)
def save_hard_example(frame: np.ndarray, bbox: List[float], class_name: str, confidence: float = 0.45) -> Optional[Dict[str, Any]]:
    return hard_example_miner.save_hard_example(frame, bbox, class_name, confidence)
