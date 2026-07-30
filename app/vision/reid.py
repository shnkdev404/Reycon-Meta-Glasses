"""
Phase 3: Person Re-Identification (ReID) Engine.

Extracts 512-D visual feature embeddings from person image crops
and performs cosine similarity matching to track person identities
across multiple smart glass camera viewpoints.
"""
import logging
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np

logger = logging.getLogger("PersonReID")


def compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Computes cosine similarity between two 512-D feature vectors.
    similarity = (v1 . v2) / (||v1|| * ||v2||)
    Returns float in range [-1.0, 1.0].
    """
    if feat1 is None or feat2 is None:
        return 0.0
    v1 = np.asarray(feat1, dtype=np.float32).flatten()
    v2 = np.asarray(feat2, dtype=np.float32).flatten()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    dot = float(np.dot(v1, v2))
    sim = dot / (norm1 * norm2)
    return round(float(np.clip(sim, -1.0, 1.0)), 4)


class PersonReIDExtractor:
    """
    Person Re-Identification (ReID) Feature Extractor.
    Extracts 512-D normalized visual feature embeddings for person image crops.
    """

    def __init__(self, model_name: str = "osnet_x1_0", embedding_dim: int = 512):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._extractor = None
        self._initialize_extractor()

    def _initialize_extractor(self):
        """Attempts to load torchreid FeatureExtractor if available."""
        try:
            import torch
            from torchreid.utils import FeatureExtractor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._extractor = FeatureExtractor(
                model_name=self.model_name,
                device=device
            )
            logger.info(f"Loaded torchreid model '{self.model_name}' on device '{device}'.")
        except Exception as e:
            logger.info(f"Torchreid unavailable ({e}). Using lightweight 512-D spatial feature encoder.")
            self._extractor = None

    def crop_person(self, frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """Crops image region inside bounding box."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or not bbox:
            return None
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))

            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2]
        except Exception:
            pass
        return None

    def extract_features(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        Extracts 512-D unit L2-normalized feature vector for a person detection crop.
        """
        crop = self.crop_person(frame, bbox)

        if self._extractor is not None and crop is not None:
            try:
                # Run torchreid extractor
                feat = self._extractor(crop)
                if hasattr(feat, "cpu"):
                    feat = feat.cpu().numpy()
                feat = np.asarray(feat, dtype=np.float32).flatten()
                if feat.size >= self.embedding_dim:
                    feat = feat[:self.embedding_dim]
                    norm = np.linalg.norm(feat)
                    if norm > 1e-6:
                        return feat / norm
            except Exception as e:
                logger.error(f"Torchreid feature extraction error: {e}")

        # Lightweight 512-D L2-normalized spatial color/texture descriptor encoder
        return self._fallback_512d_encoder(crop, bbox)

    def _fallback_512d_encoder(self, crop: Optional[np.ndarray], bbox: list) -> np.ndarray:
        """Generates deterministic 512-D unit L2-normalized feature vector from crop visual properties."""
        feat = np.zeros(self.embedding_dim, dtype=np.float32)

        if crop is not None and crop.size > 0:
            try:
                resized = cv2.resize(crop, (16, 16))
                flattened = resized.astype(np.float32).flatten()  # 16x16x3 = 768 elements
                feat[:min(512, len(flattened))] = flattened[:512]
            except Exception:
                pass

        if np.all(feat == 0):
            # Seed from bounding box geometry
            seed = int(abs(bbox[0] * 31 + bbox[1] * 17 + bbox[2] * 13 + bbox[3] * 7)) % 10000
            rng = np.random.RandomState(seed)
            feat = rng.randn(self.embedding_dim).astype(np.float32)

        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        return feat

    def match_person_identity(
        self,
        target_feature: np.ndarray,
        candidate_features: Dict[str, np.ndarray],
        similarity_threshold: float = 0.65
    ) -> Optional[Tuple[str, float]]:
        """
        Matches target person 512-D feature against candidates across different viewpoint camera feeds.
        Returns (matched_person_id, cosine_similarity) if similarity >= similarity_threshold.
        """
        if target_feature is None or not candidate_features:
            return None

        best_id = None
        best_sim = -1.0

        for candidate_id, cand_feat in candidate_features.items():
            sim = compute_cosine_similarity(target_feature, cand_feat)
            if sim > best_sim:
                best_sim = sim
                best_id = candidate_id

        if best_sim >= similarity_threshold:
            return best_id, best_sim
        return None
