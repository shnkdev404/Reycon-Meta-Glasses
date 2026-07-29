"""
Shared Perception WebSocket Endpoint.
Handles multi-glass connections, spatial telemetry streaming, detection processing,
3D landmark map updates, and real-time cross-glass alert distribution.
"""
import math
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
                # 1. Extract position & heading orientation
                pos_dict = data.get("position", {})
                heading = float(data.get("heading", 0.0))
                if not heading and isinstance(data.get("pose"), dict):
                    heading = float(data.get("pose", {}).get("heading", 0.0))

                glass_pos = Position3D(
                    x=float(pos_dict.get("x", 0.0)),
                    y=float(pos_dict.get("y", 0.0)),
                    z=float(pos_dict.get("z", 0.0))
                )

                # 2. Extract pose matrix
                raw_pose = data.get("pose")
                if raw_pose and isinstance(raw_pose, list):
                    pose_matrix = np.array(raw_pose, dtype=float)
                else:
                    pose_matrix = np.eye(4)

                # Update glass position, pose, and heading angle
                world_manager.update_glass_pose(glass_id, pose_matrix, glass_pos, heading=heading)

                # 3. Process Detections / Threats (Mapped 3D relative to Glass position & heading)
                detections = data.get("detections", [])
                heading_rad = math.radians(heading)

                for i, det in enumerate(detections):
                    obj_type = det.get("object_type", det.get("label", det.get("class_name", "hazard")))
                    det_pos_dict = det.get("position", {})
                    
                    if "x" in det_pos_dict and "y" in det_pos_dict:
                        # Absolute position supplied
                        det_pos = Position3D(
                            x=float(det_pos_dict.get("x", 0.0)),
                            y=float(det_pos_dict.get("y", 0.0)),
                            z=float(det_pos_dict.get("z", 0.0))
                        )
                    else:
                        # Map relative distance w.r.t glass position & heading
                        dist = float(det.get("distance", 3.5))
                        bearing_deg = float(det.get("bearing", 0.0))
                        rad = math.radians(heading + bearing_deg)
                        
                        det_x = glass_pos.x + dist * math.cos(rad)
                        det_y = glass_pos.y + dist * math.sin(rad)
                        det_z = glass_pos.z
                        det_pos = Position3D(x=round(det_x, 2), y=round(det_y, 2), z=round(det_z, 2))

                    threat_id = det.get("threat_id", f"threat_{glass_id}_{obj_type}")
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
