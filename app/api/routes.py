from fastapi import APIRouter
from app.services.world_manager import world_manager

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "Shared Perception Server for Ray-Ban Meta Smart Glasses 🚀",
        "status": "active"
    }


@router.get("/health")
def health():
    return {
        "status": "OK"
    }



@router.get("/world")
async def get_world_state():
    """Returns the current synchronized state of all connected smart glasses, world objects, and active threats."""
    return await world_manager.get_full_world_state()


@router.get("/glasses")
async def get_glasses():
    """Returns list of all active connected Ray-Ban Meta Smart Glasses."""
    state = await world_manager.get_full_world_state()
    return state.get("glasses", {})


@router.get("/threats")
async def get_threats():
    """Returns list of currently active spatial threat alerts."""
    return world_manager.get_active_threats()

