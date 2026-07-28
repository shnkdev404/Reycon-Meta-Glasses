"""
Phase 3: Multi-Object Tracking (MOT) Interface & Wrappers.

Abstract contract & wrappers for 2D/3D temporal object tracking.
TODO: Connect real tracking algorithms (ByteTrack, DeepSORT, BoT-SORT).
"""
from abc import ABC, abstractmethod
from typing import List, Any
from app.models.object import Detection2D


class BaseTracker(ABC):
    """Abstract contract for multi-object tracking across video frames."""

    @abstractmethod
    def update(self, detections: List[Detection2D], frame: Any = None) -> List[Detection2D]:
        """Update tracker trajectories with new frame detections."""
        pass


import math
from typing import List, Any, Dict, Optional
from app.models.object import Detection2D, BoundingBox2D


def compute_iou(box1: Optional[BoundingBox2D], box2: Optional[BoundingBox2D]) -> float:
    """Compute Intersection-over-Union (IoU) between two 2D bounding boxes."""
    if not box1 or not box2:
        return 0.0
    
    x1 = max(box1.xmin, box2.xmin)
    y1 = max(box1.ymin, box2.ymin)
    x2 = min(box1.xmax, box2.xmax)
    y2 = min(box1.ymax, box2.ymax)

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    box2_area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class TrackState:
    """State record for an active track persistent across video frames."""

    def __init__(self, track_id: int, detection: Detection2D):
        self.track_id = track_id
        self.label = detection.label
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        self.distance = detection.distance
        self.bearing = detection.bearing
        self.age = 1
        self.missed = 0

    def update(self, detection: Detection2D):
        self.confidence = detection.confidence
        if detection.bbox:
            self.bbox = detection.bbox
        self.distance = detection.distance
        self.bearing = detection.bearing
        self.age += 1
        self.missed = 0


class TrackManager(BaseTracker):
    """
    Multi-Object Tracking (MOT) data association engine compatible with ByteTrack / DeepSORT principles.
    Uses IoU and spatial distance matching across consecutive frames to maintain persistent track IDs.
    """

    def __init__(self, algorithm: str = "ByteTrack", iou_threshold: float = 0.3, max_missed: int = 5):
        self.algorithm = algorithm
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self.active_tracks: Dict[int, TrackState] = {}

    def update(self, detections: List[Detection2D], frame: Any = None) -> List[Detection2D]:
        """
        Associate new frame detections with existing active tracks or initialize new tracks.
        Returns detections populated with persistent tracking metadata (label formatted with track_id).
        """
        matched_track_ids = set()
        matched_det_indices = set()

        # Step 1: Match existing tracks with detections via IoU and Label matching
        for track_id, track in list(self.active_tracks.items()):
            best_iou = 0.0
            best_det_idx = -1

            for idx, det in enumerate(detections):
                if idx in matched_det_indices:
                    continue

                # Category match
                if det.label != track.label:
                    continue

                # Calculate 2D IoU if bounding boxes exist
                iou = compute_iou(track.bbox, det.bbox)
                
                # Spatial distance match fallback if IoU is zero
                spatial_dist = abs(det.distance - track.distance) + abs(det.bearing - track.bearing) * 0.1
                is_spatial_match = spatial_dist < 2.0

                if iou >= self.iou_threshold or is_spatial_match:
                    score = iou if iou > 0 else (1.0 / (1.0 + spatial_dist))
                    if score > best_iou:
                        best_iou = score
                        best_det_idx = idx

            if best_det_idx != -1:
                track.update(detections[best_det_idx])
                matched_track_ids.add(track_id)
                matched_det_indices.add(best_det_idx)

        # Step 2: Increment missed frames for unmatched active tracks and prune expired tracks
        for track_id, track in list(self.active_tracks.items()):
            if track_id not in matched_track_ids:
                track.missed += 1
                if track.missed > self.max_missed:
                    del self.active_tracks[track_id]

        # Step 3: Initialize new tracks for unmatched detections
        tracked_detections: List[Detection2D] = []
        for idx, det in enumerate(detections):
            if idx in matched_det_indices:
                # Find matching track_id
                matched_id = next((tid for tid, trk in self.active_tracks.items() if trk.bbox == det.bbox or (trk.label == det.label and abs(trk.distance - det.distance) < 0.1)), None)
                assigned_id = matched_id or self._next_id
            else:
                assigned_id = self._next_id
                self.active_tracks[assigned_id] = TrackState(assigned_id, det)
                self._next_id += 1

            # Format label with persistent track ID e.g. "vehicle #1"
            tracked_det = det.model_copy(deep=True)
            tracked_det.label = f"{det.label} #{assigned_id}"
            tracked_detections.append(tracked_det)

        return tracked_detections

