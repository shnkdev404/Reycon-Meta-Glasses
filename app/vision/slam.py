"""
Phase 4 & Phase 14: Pose Estimation & Visual SLAM Interfaces.

Abstract contracts & wrappers for Visual Odometry, VIO, and Real-Time SLAM systems.
Integrates ORBSLAMWrapper (ORB-SLAM3 Monocular + IMU) engine.
"""
import math
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from app.models.glass import GlassPose, GlassSensors
from app.slam.orbslam3_wrapper import ORBSLAMWrapper, ORBSLAM3Wrapper

logger = logging.getLogger("SLAMManager")


class BaseSLAM(ABC):
    """Abstract contract for Visual Inertial Odometry and 6DoF Pose Tracking."""

    @abstractmethod
    def track_pose(self, frame: Any = None, imu_data: Any = None, dt: float = 0.033) -> GlassPose:
        """Estimate 6DoF camera position and orientation."""
        pass

    @abstractmethod
    def reset_origin(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, heading: float = 0.0):
        """Reset spatial origin anchor."""
        pass


class SLAMManager(BaseSLAM):
    """
    Visual Inertial Odometry (VIO) & SLAM Manager wrapper integrating ORBSLAMWrapper
    (ORB-SLAM3 Monocular + IMU). Real-time 6DoF global coordinate frame tracking,
    loop closure detection, relocalization, and map persistence.
    """

    def __init__(
        self,
        backend: str = "ORBSLAM3",
        vocab_path: str = "ORBvoc.txt",
        config_path: str = "camera.yaml",
        init_pose: Optional[GlassPose] = None
    ):
        self.backend = backend
        self.vocab_path = vocab_path
        self.config_path = config_path
        self._slam_engine: Optional[ORBSLAM3Wrapper] = None
        
        # 6DoF State variables
        self.x = init_pose.x if init_pose else 0.0
        self.y = init_pose.y if init_pose else 0.0
        self.z = init_pose.z if init_pose else 1.65  # Default standing height
        self.heading = init_pose.heading if init_pose else 0.0
        self.pitch = init_pose.pitch if init_pose else 0.0
        self.roll = init_pose.roll if init_pose else 0.0

        # Velocity state (m/s)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self._initialize_backend()

    def _initialize_backend(self):
        """Initializes ORBSLAMWrapper (ORB-SLAM3 Monocular+IMU engine)."""
        try:
            self._slam_engine = ORBSLAMWrapper(
                vocab_path=self.vocab_path,
                config_path=self.config_path,
                glass_id="meta_glass_slam"
            )
            logger.info("✅ SLAMManager bound successfully to ORBSLAMWrapper backend.")
        except Exception as e:
            logger.warning(f"Error binding SLAM backend: {e}")
            self._slam_engine = None

    def track_pose(self, frame: Any = None, imu_data: Any = None, dt: float = 0.033) -> GlassPose:
        """
        Compute updated 6DoF pose from visual-inertial frame features.
        Fuses high-frequency IMU acceleration/gyroscope readings with visual odometry.
        """
        if dt <= 0:
            dt = 0.033

        # Step 1: Process native ORBSLAMWrapper engine if bound
        if self._slam_engine is not None:
            try:
                sensors = None
                if isinstance(imu_data, GlassSensors):
                    sensors = imu_data
                elif isinstance(imu_data, dict):
                    sensors = GlassSensors(
                        accel_x=float(imu_data.get("accel_x", 0.0)),
                        accel_y=float(imu_data.get("accel_y", 0.0)),
                        accel_z=float(imu_data.get("accel_z", 9.81)),
                        gyro_z=float(imu_data.get("gyro_z", 0.0))
                    )

                slam_pose = self._slam_engine.track_mono(frame, sensors=sensors, dt=dt)
                self.x, self.y, self.z = slam_pose.x, slam_pose.y, slam_pose.z
                self.heading, self.pitch, self.roll = slam_pose.heading, slam_pose.pitch, slam_pose.roll
                return slam_pose
            except Exception as e:
                logger.debug(f"SLAM engine tracking fallback: {e}")

        # Step 2: IMU Kinematic Dead Reckoning fallback
        accel_x, accel_y, accel_z = 0.0, 0.0, 9.81
        gyro_z = 0.0

        if isinstance(imu_data, GlassSensors):
            accel_x = imu_data.accel_x
            accel_y = imu_data.accel_y
            accel_z = imu_data.accel_z
            gyro_z = imu_data.gyro_z
        elif isinstance(imu_data, dict):
            accel_x = float(imu_data.get("accel_x", 0.0))
            accel_y = float(imu_data.get("accel_y", 0.0))
            accel_z = float(imu_data.get("accel_z", 9.81))
            gyro_z = float(imu_data.get("gyro_z", 0.0))

        # Integrate gyroscope to update heading orientation (rad to deg conversion)
        heading_delta_deg = math.degrees(gyro_z * dt)
        self.heading = (self.heading + heading_delta_deg + 360.0) % 360.0

        # Remove gravity component from Z acceleration
        net_accel_z = accel_z - 9.81

        # Damping factor to prevent unbounded drift in stationary simulation
        damping = 0.90
        self.vx = (self.vx + accel_x * dt) * damping
        self.vy = (self.vy + accel_y * dt) * damping
        self.vz = (self.vz + net_accel_z * dt) * damping

        # Transform local velocities into global coordinates using heading
        heading_rad = math.radians(self.heading)
        global_vx = self.vx * math.cos(heading_rad) - self.vy * math.sin(heading_rad)
        global_vy = self.vx * math.sin(heading_rad) + self.vy * math.cos(heading_rad)

        # Update 3D position
        self.x += global_vx * dt
        self.y += global_vy * dt
        self.z = max(0.0, self.z + self.vz * dt)

        # Step 3: Visual Odometry Sensor Fusion (EKF Update step)
        if isinstance(frame, dict) and "visual_odometry" in frame:
            vo = frame["visual_odometry"]
            vo_dx = float(vo.get("dx", 0.0))
            vo_dy = float(vo.get("dy", 0.0))
            vo_dheading = float(vo.get("dheading", 0.0))

            self.x += vo_dx * 0.5
            self.y += vo_dy * 0.5
            self.heading = (self.heading + vo_dheading * 0.5 + 360.0) % 360.0

        return self.get_pose()

    def get_all_poses(self) -> List[GlassPose]:
        """Return all historical 6DoF camera poses from SLAM engine."""
        if self._slam_engine is not None:
            return self._slam_engine.get_all_poses()
        return [self.get_pose()]

    def relocalize(self, frame: Any = None, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Perform SLAM relocalization."""
        if self._slam_engine is not None:
            return self._slam_engine.relocalize(frame, timestamp=timestamp)
        return {"success": False, "relocalized_pose": self.get_pose(), "relocalization_count": 0}

    def reset_origin(self, x: float = 0.0, y: float = 0.0, z: float = 1.65, heading: float = 0.0):
        """Reset spatial origin anchor coordinates and heading."""
        self.x = x
        self.y = y
        self.z = z
        self.heading = (heading + 360.0) % 360.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        if self._slam_engine is not None:
            self._slam_engine.current_pose = GlassPose(x=x, y=y, z=z, heading=heading)

    def get_pose(self) -> GlassPose:
        """Return the current estimated 6DoF pose."""
        return GlassPose(
            x=round(self.x, 3),
            y=round(self.y, 3),
            z=round(self.z, 3),
            heading=round(self.heading, 1),
            pitch=round(self.pitch, 1),
            roll=round(self.roll, 1)
        )
