"""
Phase 1: High-throughput WebSocket API Endpoint.

Validates incoming smart glass spatial telemetry, coordinates spatial fusion & threat engines,
and delivers direct acknowledgments and non-broadcast targeted alerts.
"""
import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from pydantic import ValidationError
from app.models.glass import GlassState, GlassPose
from app.models.object import Detection2D
from app.services.connection_manager import connection_manager
from app.services.world_manager import world_manager
from app.utils.logger import get_logger

logger = get_logger("WebSocketAPI")
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    glass_id: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Query(default=None)
):
    """
    WebSocket endpoint for Ray-Ban Meta Smart Glasses.
    URL Format: ws://127.0.0.1:8000/ws?glass_id=glass_A
    """
    client_glass_id: str = glass_id or "unknown_glass"
    is_connected = False

    try:
        # Step 1: Pre-parse glass_id if provided in query or wait for first packet
        raw_text = await websocket.receive_text()
        first_payload = json.loads(raw_text)
        
        if "glass_id" in first_payload:
            client_glass_id = first_payload["glass_id"]

        # Step 2: Accept connection & register session
        is_connected = await connection_manager.connect(
            glass_id=client_glass_id,
            websocket=websocket,
            auth_token=auth_token
        )

        if not is_connected:
            return

        # Step 3: Process initial payload
        await _process_telemetry_packet(client_glass_id, first_payload, websocket)

        # Step 4: Stream processing loop
        while True:
            raw_text = await websocket.receive_text()
            connection_manager.update_heartbeat(client_glass_id)
            
            payload = json.loads(raw_text)
            if "glass_id" in payload:
                client_glass_id = payload["glass_id"]
                
            await _process_telemetry_packet(client_glass_id, payload, websocket)

    except WebSocketDisconnect:
        logger.info(f"📌 Client '{client_glass_id}' disconnected gracefully.")
        connection_manager.disconnect(client_glass_id)
        await world_manager.remove_glass(client_glass_id)

    except json.JSONDecodeError:
        logger.error(f"⚠️ JSONDecodeError from '{client_glass_id}'.")
        if is_connected:
            await websocket.send_json({"status": "error", "message": "Malformed JSON payload"})

    except Exception as e:
        logger.error(f"⚠️ WebSocket error from '{client_glass_id}': {e}")
        connection_manager.disconnect(client_glass_id)
        await world_manager.remove_glass(client_glass_id)


async def _process_telemetry_packet(glass_id: str, payload: dict, websocket: WebSocket):
    """
    Validate telemetry schema, update World Model, and return direct acknowledgment.
    Supports both nested Pydantic structure and legacy flat telemetry packets.
    """
    try:
        # Normalize incoming JSON to GlassState & Detection2D models
        if "pose" in payload:
            glass_state = GlassState.model_validate(payload)
        else:
            # Construct GlassState from flat schema (position: {x, y}, heading)
            pos_dict = payload.get("position", {})
            glass_state = GlassState(
                glass_id=glass_id,
                pose=GlassPose(
                    x=float(pos_dict.get("x", 0.0)),
                    y=float(pos_dict.get("y", 0.0)),
                    z=float(pos_dict.get("z", 0.0)),
                    heading=float(payload.get("heading", 0.0))
                )
            )

        # Parse detections
        raw_detections = payload.get("detections", [])
        detections = [Detection2D.model_validate(d) for d in raw_detections]

        # Synchronize with central World Model & evaluate threat engine
        active_threats = await world_manager.update_glass_telemetry(glass_state, detections)

        # Send direct status response back to sending glass
        await websocket.send_json({
            "status": "ok",
            "glass_id": glass_id,
            "active_threats_count": len(active_threats),
            "message": "World model synchronized"
        })

    except ValidationError as ve:
        logger.warning(f"⚠️ Validation error for glass '{glass_id}': {ve.errors()}")
        await websocket.send_json({
            "status": "error",
            "error_type": "ValidationError",
            "details": ve.errors()
        })
