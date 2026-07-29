"""
Shared Perception REST API Routes.
Provides REST endpoints for fetching full 3D world state, active threats,
registered glasses, landmark map points, and resetting server state.
"""
import logging
from fastapi import APIRouter
from app.services.shared_world_manager import world_manager

logger = logging.getLogger("SharedRoutes")

router = APIRouter(prefix="/api/shared", tags=["Shared Perception REST API"])


@router.get("/world_state")
async def get_world_state():
    """
    Get full synchronized world model state:
    - Active glasses poses & positions
    - Active threats with detector metadata
    - Persistent 3D map points
    - Telemetry statistics
    """
    with world_manager.lock:
        glasses_dict = {}
        for gid, ginfo in world_manager.glasses.items():
            pos = ginfo.get("position")
            glasses_dict[gid] = {
                "id": gid,
                "position": pos.to_dict() if hasattr(pos, "to_dict") else pos,
                "heading": float(ginfo.get("heading", 0.0)),
                "connected": ginfo.get("connected", True),
                "timestamp": ginfo.get("timestamp")
            }

        return {
            "status": "success",
            "glasses": glasses_dict,
            "threats": world_manager.get_all_threats(),
            "map_points": [p.to_dict() for p in world_manager.get_map_points()],
            "stats": world_manager.get_map_statistics()
        }


@router.get("/map")
async def get_map_points():
    """Get all 3D SLAM map landmarks."""
    return {
        "count": len(world_manager.map_points),
        "map_points": [p.to_dict() for p in world_manager.get_map_points()]
    }


@router.get("/threats")
async def get_threats():
    """Get active threats."""
    return {
        "count": len(world_manager.threats),
        "threats": world_manager.get_all_threats()
    }


@router.get("/glasses")
async def get_glasses():
    """Get all registered glasses poses and positions."""
    with world_manager.lock:
        return {
            "count": len(world_manager.glasses),
            "glasses": [
                {
                    "id": gid,
                    "position": g["position"].to_dict() if hasattr(g["position"], "to_dict") else g["position"],
                    "heading": float(g.get("heading", 0.0)),
                    "connected": g.get("connected", True)
                }
                for gid, g in world_manager.glasses.items()
            ]
        }


@router.post("/reset")
async def reset_state():
    """Reset all world manager data."""
    world_manager.reset()
    return {"status": "success", "message": "Shared world state reset."}
