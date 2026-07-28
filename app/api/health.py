"""
Health check and REST system metrics API endpoints.
"""
from fastapi import APIRouter
from app.services.world_manager import world_manager
from app.utils.config import settings

router = APIRouter()


@router.get("/health")
def get_health():
    """Return platform status."""
    return {
        "status": "OK",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }


@router.get("/world")
async def get_world():
    """Retrieve full synchronized 3D World Model state."""
    return await world_manager.get_full_world_state()
