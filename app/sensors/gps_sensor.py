"""
Phase 2: Sensor Interface Layer - GPS / GNSS Sensor Interface.

Independent contract and simulated GPS interface providing latitude, longitude,
altitude, and satellite location fix precision.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pydantic import BaseModel, Field


class GPSReading(BaseModel):
    latitude: float = Field(..., description="Latitude in decimal degrees")
    longitude: float = Field(..., description="Longitude in decimal degrees")
    altitude_m: float = Field(default=0.0, description="Altitude in meters above sea level")
    accuracy_m: float = Field(default=2.5, description="Horizontal GPS fix accuracy radius in meters")


class BaseGPSSensor(ABC):
    """Abstract contract for GPS / GNSS location sensors."""

    @abstractmethod
    def read_gps(self) -> Optional[GPSReading]:
        """Fetch latest GPS fix coordinates."""
        pass


class SimulatedGPSSensor(BaseGPSSensor):
    """Synthetic GPS sensor producing simulated global positioning fix data."""

    def __init__(self, glass_id: str, init_lat: float = 37.7749, init_lon: float = -122.4194):
        self.glass_id = glass_id
        self.latitude = init_lat
        self.longitude = init_lon

    def read_gps(self) -> GPSReading:
        return GPSReading(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude_m=15.0,
            accuracy_m=1.2
        )
