"""
Phase 10: Meta Wearable SDK 6DoF Pose Adapter.

Adapter interfacing Meta Smart Glasses spatial tracking and 6DoF pose telemetry.
TODO: Plug Meta Wearable SDK Pose API.
"""
from app.models.glass import GlassPose


from app.models.glass import GlassPose
from app.sensors.head_pose_sensor import BaseHeadPoseSensor


class MetaPoseAdapter(BaseHeadPoseSensor):
    """
    Adapter interfacing Meta Smart Glasses spatial tracking and 6DoF pose telemetry.
    Complies with BaseHeadPoseSensor contract.
    """

    def __init__(self, glass_id: str, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        self.glass_id = glass_id
        self.x = x
        self.y = y
        self.heading = heading

    def read_head_pose(self) -> GlassPose:
        """Fetch latest 6DoF head pose orientation and position from Meta Wearable SDK."""
        return GlassPose(
            x=self.x,
            y=self.y,
            z=1.65,  # Standing eye-level height in meters
            heading=self.heading,
            pitch=0.0,
            roll=0.0
        )

    def get_current_pose(self) -> GlassPose:
        """Alias for read_head_pose()."""
        return self.read_head_pose()

