"""
Shared Perception REST API Routes.
Provides REST endpoints for fetching full 3D world state, active threats,
registered glasses, landmark map points, and resetting server state.
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Body
from app.services.shared_world_manager import world_manager
from app.services.memory_manager import memory_manager
from app.services.connection_manager import connection_manager

logger = logging.getLogger("SharedRoutes")

router = APIRouter(prefix="/api/shared", tags=["Shared Perception REST API"])


@router.get("/world_state")
async def get_world_state():
    """
    Get full synchronized world model state:
    - Active glasses/phones poses & exact 3D positions
    - Active threats with detector metadata
    - Persistent 3D map points & objects
    - Telemetry statistics
    """
    with world_manager.lock:
        world_manager.prune_stale_glasses()
        glasses_dict = {}
        for gid, ginfo in world_manager.glasses.items():
            pos = ginfo.get("position")
            pos_dict = pos.to_dict() if hasattr(pos, "to_dict") else (pos if isinstance(pos, dict) else {"x": 0.0, "y": 0.0, "z": 0.0})
            x_m = pos_dict.get("x", 0.0)
            y_m = pos_dict.get("y", 0.0)
            z_m = pos_dict.get("z", 0.0)
            
            glasses_dict[gid] = {
                "id": gid,
                "position": pos_dict,
                "formatted_position": f"X: {x_m:+.2f}m, Y: {y_m:+.2f}m, Z: {z_m:+.2f}m",
                "heading": float(ginfo.get("heading", 0.0)),
                "connected": ginfo.get("connected", True),
                "timestamp": ginfo.get("timestamp")
            }

        persistent_objs = memory_manager.get_all_persistent_objects()

        return {
            "status": "success",
            "glasses": glasses_dict,
            "threats": world_manager.get_all_threats(),
            "persistent_objects": persistent_objs,
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
    """Get all registered glasses/phones poses, headings, and exact 3D positions."""
    with world_manager.lock:
        world_manager.prune_stale_glasses()
        glasses_list = []
        for gid, g in world_manager.glasses.items():
            pos = g.get("position")
            pos_dict = pos.to_dict() if hasattr(pos, "to_dict") else (pos if isinstance(pos, dict) else {"x": 0.0, "y": 0.0, "z": 0.0})
            x_m = pos_dict.get("x", 0.0)
            y_m = pos_dict.get("y", 0.0)
            z_m = pos_dict.get("z", 0.0)
            glasses_list.append({
                "id": gid,
                "position": pos_dict,
                "formatted_position": f"X: {x_m:+.2f}m, Y: {y_m:+.2f}m, Z: {z_m:+.2f}m",
                "heading": float(g.get("heading", 0.0)),
                "connected": g.get("connected", True),
                "timestamp": g.get("timestamp")
            })

        return {
            "count": len(glasses_list),
            "glasses": glasses_list
        }


@router.get("/persistent_memory")
async def get_persistent_memory():
    """Get all persistent remembered objects with their exact 3D position and correction status."""
    objects = memory_manager.get_all_persistent_objects()
    return {
        "status": "success",
        "count": len(objects),
        "objects": objects
    }


@router.post("/correct_object")
async def correct_object_label(payload: Dict[str, Any] = Body(...)):
    """
    Correct a detected object's label/classification in case of misinterpretation.
    Payload: { "object_id": "string", "new_label": "string" }
    """
    object_id = payload.get("object_id") or payload.get("threat_id")
    new_label = payload.get("new_label") or payload.get("corrected_label")

    if not object_id or not new_label:
        raise HTTPException(status_code=400, detail="Missing object_id or new_label in request body.")

    updated_obj = world_manager.correct_object_label(object_id, new_label)

    if not updated_obj:
        raise HTTPException(status_code=444 if False else 404, detail=f"Object with ID '{object_id}' not found in persistent memory store.")

    # Broadcast correction to all connected WebSocket clients (Mobile & Server Dashboards)
    try:
        correction_event = {
            "event": "object_corrected",
            "object_id": object_id,
            "new_label": new_label,
            "object": updated_obj
        }
        import asyncio
        asyncio.create_task(connection_manager.broadcast(correction_event))
    except Exception as e:
        logger.warning(f"Failed to broadcast object correction event: {e}")

    return {
        "status": "success",
        "message": f"Object '{object_id}' label corrected to '{new_label}'.",
        "object": updated_obj
    }


@router.post("/reset")
async def reset_state():
    """Reset all world manager data and persistent memory store."""
    world_manager.reset(clear_persistent_memory=True)
    from app.services.world_manager import world_manager as wm
    wm.reset_world_state()
    return {"status": "success", "message": "Shared world state and persistent memory reset."}


