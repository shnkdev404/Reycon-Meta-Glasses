"""
Phase 2: Sensor Interface Layer - IMU (Inertial Measurement Unit) Sensor Interface.

Independent contract and simulated IMU sensor interface reading
accelerometer and gyroscope motion telemetry.
"""
from abc import ABC, abstractmethod
from typing import Dict
from app.models.glass import GlassSensors


class BaseIMUSensor(ABC):
    """Abstract contract for Smart Glasses IMU sensors (Accel + Gyro)."""

    @abstractmethod
    def read_imu(self) -> GlassSensors:
        """Fetch current accelerometer and gyroscope values."""
        pass


class SimulatedIMUSensor(BaseIMUSensor):
    """Synthetic IMU sensor generating simulated accelerometer & gyroscope telemetry."""

    def __init__(self, glass_id: str):
        self.glass_id = glass_id

    def read_imu(self) -> GlassSensors:
        return GlassSensors(
            accel_x=0.01,
            accel_y=-0.02,
            accel_z=9.81,  # Standard gravity (m/s^2)
            gyro_x=0.001,
            gyro_y=0.002,
            gyro_z=0.000
        )
