"""
Shared Perception WebSocket Endpoint.
Handles multi-glass connections, spatial telemetry streaming, detection processing,
3D landmark map updates, and real-time cross-glass alert distribution.
"""
import json
import logging
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.shared_world_manager import world_manager, Position3D

logger = logging.getLogger("SharedWebSocket")

router = APIRouter(tags=["Shared WebSocket"])


@router.websocket("/ws/glass/{glass_id}")
async def glass_websocket_endpoint(websocket: WebSocket, glass_id: str):
    """
    WebSocket endpoint for Meta Smart Glasses telemetry.
    
    Expects incoming JSON telemetry payload:
    {
        "type": "telemetry",
        "position": {"x": 10, "y": 5, "z": 0},
        "pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        "detections": [
            {
                "object_type": "truck",
                "position": {"x": 12, "y": 5, "z": 0},
                "confidence": 0.9
            }
        ],
        "map_points": [
            {
                "point_id": "mp_001",
                "position": {"x": 10, "y": 5, "z": 0}
            }
        ]
    }
    """
    await websocket.accept()
    logger.info(f"🕶️ Glass connected: {glass_id}")

    # Register glass in SharedWorldManager
    initial_pos = Position3D(0.0, 0.0, 0.0)
    world_manager.register_glass(glass_id, initial_pos)

    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)

            msg_type = data.get("type", "telemetry")

            if msg_type in ["telemetry", "glass_telemetry"]:
                # 1. Extract position
                pos_dict = data.get("position", {})
                glass_pos = Position3D(
                    x=float(pos_dict.get("x", 0.0)),
                    y=float(pos_dict.get("y", 0.0)),
                    z=float(pos_dict.get("z", 0.0))
                )

                # 2. Extract pose matrix
                raw_pose = data.get("pose")
                if raw_pose:
                    pose_matrix = np.array(raw_pose, dtype=float)
                else:
                    pose_matrix = np.eye(4)

                # Update glass position & pose
                world_manager.update_glass_pose(glass_id, pose_matrix, glass_pos)

                # 3. Process Detections / Threats
                detections = data.get("detections", [])
                for i, det in enumerate(detections):
                    obj_type = det.get("object_type", det.get("label", "hazard"))
                    det_pos_dict = det.get("position", {})
                    det_pos = Position3D(
                        x=float(det_pos_dict.get("x", glass_pos.x)),
                        y=float(det_pos_dict.get("y", glass_pos.y)),
                        z=float(det_pos_dict.get("z", glass_pos.z))
                    )
                    threat_id = det.get("threat_id", f"threat_{glass_id}_{i}_{int(det_pos.x)}")
                    conf = float(det.get("confidence", 0.8))
                    vel = tuple(det.get("velocity", (0.0, 0.0, 0.0)))

                    world_manager.add_threat(
                        threat_id=threat_id,
                        object_type=obj_type,
                        position=det_pos,
                        detected_by_glass_id=glass_id,
                        confidence=conf,
                        velocity=vel
                    )

                # 4. Process Map Points (3D SLAM Landmarks)
                map_points = data.get("map_points", [])
                for mp in map_points:
                    pt_id = mp.get("point_id", f"mp_{glass_id}_{len(world_manager.map_points)}")
                    pt_pos_dict = mp.get("position", {})
                    pt_pos = Position3D(
                        x=float(pt_pos_dict.get("x", 0.0)),
                        y=float(pt_pos_dict.get("y", 0.0)),
                        z=float(pt_pos_dict.get("z", 0.0))
                    )
                    world_manager.add_map_point(pt_id, pt_pos, glass_id)

                # 5. Fetch alerts relevant to this glass (includes cross-glass alerts!)
                alerts = world_manager.get_alerts_for_glass(glass_id)

                # 6. Response payload
                response_payload = {
                    "status": "ok",
                    "glass_id": glass_id,
                    "alerts": alerts,
                    "all_threats": world_manager.get_all_threats(),
                    "map_stats": world_manager.get_map_statistics()
                }

                await websocket.send_text(json.dumps(response_payload))

    except WebSocketDisconnect:
        logger.info(f"🕶️ Glass disconnected: {glass_id}")
        if glass_id in world_manager.glasses:
            world_manager.glasses[glass_id]["connected"] = False
    except Exception as e:
        logger.error(f"Error in Glass WebSocket handler for {glass_id}: {e}")
        await websocket.close()
