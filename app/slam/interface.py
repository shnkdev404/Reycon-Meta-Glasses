"""
Phase 1: Abstract SLAM Interface Layer.
Decouples SLAM implementation (ORB-SLAM3 Monocular+IMU) from project architecture.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
from app.models.glass import GlassPose, GlassSensors
from app.models.map import LocalMap


class BaseSLAMInterface(ABC):
    """Abstract contract for Smart Glasses SLAM engines."""

    @abstractmethod
    def track_monocular_imu(self, frame: Any, sensors: GlassSensors) -> GlassPose:
        """Process incoming camera frame & IMU telemetry to compute 6DoF camera pose."""
        pass

    @abstractmethod
    def get_camera_pose(self) -> GlassPose:
        """Retrieve current 6DoF camera pose."""
        pass

    @abstractmethod
    def get_local_map(self) -> LocalMap:
        """Retrieve local SLAM map points and keyframes."""
        pass

    @abstractmethod
    def save_map(self, file_path: str) -> bool:
        """Persist SLAM map to file."""
        pass

    @abstractmethod
    def load_map(self, file_path: str) -> bool:
        """Load persistent SLAM map from file."""
        pass
