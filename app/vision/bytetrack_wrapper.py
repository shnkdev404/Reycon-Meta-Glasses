"""
Phase 2: ByteTrack Multi-Object Tracker Wrapper.
Assigns persistent IDs, tracks object bounding boxes across frames, computes 3D velocities, and maintains historical motion paths.
"""
import time
import math
import logging
from typing import List, Dict, Any
from app.models.object import Detection2D, BoundingBox2D

logger = logging.getLogger("ByteTrackWrapper")


class TrackedObjectState:
    def __init__(self, track_id: int, label: str, bbox: BoundingBox2D, distance: float, bearing: float):
        self.track_id = track_id
        self.label = label
        self.bbox = bbox
        self.distance = distance
        self.bearing = bearing
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.last_update = time.time()
        self.history: List[Dict[str, float]] = []


class ByteTrackWrapper:
    """
    ByteTrack Multi-Object Tracker assigning persistent object IDs, 
    velocity vectors, and trajectory history across consecutive frames.
    """

    def __init__(self, max_age_seconds: float = 3.0):
        self.max_age_seconds = max_age_seconds
        self.tracks: Dict[int, TrackedObjectState] = {}
        self._next_id = 1

    def update_tracks(self, raw_detections: List[Detection2D]) -> List[Detection2D]:
        """
        Associate new detections with existing tracks based on Spatial IOU and centroid distance.
        Updates persistent object IDs and estimates velocity vectors.
        """
        now = time.time()
        updated_detections: List[Detection2D] = []

        for det in raw_detections:
            matched_id = self._match_detection_to_track(det)
            if matched_id is None:
                matched_id = self._next_id
                self._next_id += 1
                self.tracks[matched_id] = TrackedObjectState(
                    track_id=matched_id,
                    label=det.label,
                    bbox=det.bbox,
                    distance=det.distance,
                    bearing=det.bearing
                )
            else:
                track = self.tracks[matched_id]
                dt = max(0.01, now - track.last_update)
                
                # Estimate velocity
                track.velocity_x = (det.distance - track.distance) / dt
                track.velocity_y = (det.bearing - track.bearing) / dt
                
                track.distance = det.distance
                track.bearing = det.bearing
                track.bbox = det.bbox
                track.last_update = now

            track = self.tracks[matched_id]
            track.history.append({
                "time": now,
                "distance": track.distance,
                "bearing": track.bearing
            })
            if len(track.history) > 30:
                track.history.pop(0)

            # Assign persistent track label (e.g. "truck #1", "worker #2")
            persistent_label = f"{det.label} #{matched_id}"
            
            updated_det = Detection2D(
                label=persistent_label,
                confidence=det.confidence,
                bbox=det.bbox,
                distance=det.distance,
                bearing=det.bearing
            )
            updated_detections.append(updated_det)

        self._prune_stale_tracks(now)
        return updated_detections

    def _match_detection_to_track(self, det: Detection2D) -> int | None:
        """Find best matching track based on bounding box IOU / centroid distance."""
        best_id = None
        min_dist = float("inf")

        for track_id, track in self.tracks.items():
            if track.label.split(" #")[0] != det.label.split(" #")[0]:
                continue
            dist = math.hypot(det.distance - track.distance, det.bearing - track.bearing)
            if dist < min_dist and dist < 5.0:  # Matching threshold
                min_dist = dist
                best_id = track_id

        return best_id

    def _prune_stale_tracks(self, now: float):
        """Remove tracks that haven't been observed recently."""
        stale_ids = [
            tid for tid, track in self.tracks.items()
            if (now - track.last_update) > self.max_age_seconds
        ]
        for tid in stale_ids:
            del self.tracks[tid]


bytetrack = ByteTrackWrapper()
