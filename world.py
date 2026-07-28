"""
World Model Module Alias
Exposes the centralized world_manager instance and GlassState model.
"""
from app.models import GlassState, Position, Detection
from app.services.world_manager import world_manager, WorldManager

__all__ = ["GlassState", "Position", "Detection", "world_manager", "WorldManager"]
