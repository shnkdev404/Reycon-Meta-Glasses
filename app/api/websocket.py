import base64
import json
import logging
import time
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import math

from app.models import GlassState, Position, GPSLocation
from app.services.connection_manager import connection_manager
from app.services.world_manager import world_manager
from app.services.detector import detector

logger = logging.getLogger("WebSocketAPI")
router = APIRouter()


def decode_base64_and_detect(b64_string: str):
    """Synchronous CPU worker: Decodes Base64 JPEG and executes YOLO detection."""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            return detector.detect_frame(frame)
    except Exception as e:
        logger.error(f"Error in frame decoding/detection worker: {e}")
    return []


@router.websocket("/ws/mobile")
@router.websocket("/ws")  # Alias for backward compatibility
async def websocket_mobile_endpoint(websocket: WebSocket):
    """
    High-performance zero-lag WebSocket endpoint for mobile clients & smart glasses.
    Telemetry updates process instantly (<1ms).
    YOLO vision inference is offloaded to background worker threads using asyncio.to_thread.
    """
    glass_id = "unknown_device"
    await websocket.accept()

    # Per-connection detection cache to prevent duplicate work
    last_detections = []
    frame_counter = 0

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({"status": "error", "message": "Invalid JSON format"})
                continue

            # Handle special actions such as object correction
            if data.get("action") == "correct_object" or data.get("type") == "correct_object":
                obj_id = data.get("object_id") or data.get("threat_id")
                new_lbl = data.get("new_label") or data.get("corrected_label")
                if obj_id and new_lbl:
                    from app.services.shared_world_manager import world_manager as shared_wm
                    updated_obj = shared_wm.correct_object_label(obj_id, new_lbl)
                    await connection_manager.broadcast({
                        "event": "object_corrected",
                        "object_id": obj_id,
                        "new_label": new_lbl,
                        "object": updated_obj
                    })
                    await websocket.send_json({
                        "status": "ok",
                        "action": "correct_object",
                        "object_id": obj_id,
                        "new_label": new_lbl,
                        "object": updated_obj
                    })
                continue

            glass_id = data.get("glass_id", glass_id)
            from app.services.shared_world_manager import extract_heading_deg, world_manager as shared_wm, Position3D
            heading = extract_heading_deg(data)
            frame_b64 = data.get("frame")

            # Register connection with manager if not already present
            if glass_id not in connection_manager.active_connections:
                connection_manager.active_connections[glass_id] = websocket
                logger.info(f"Registered connection for '{glass_id}'.")

            # Offload heavy YOLO forward pass & image decoding to threadpool so event loop is NEVER blocked!
            if frame_b64:
                frame_counter += 1
                # Run YOLO inference every frame in threadpool
                try:
                    last_detections = await asyncio.to_thread(decode_base64_and_detect, frame_b64)
                except Exception as ex:
                    logger.error(f"Async frame detection error for '{glass_id}': {ex}")

            if frame_b64:
                detections = last_detections
            else:
                raw_dets = data.get("detections", [])
                if isinstance(raw_dets, list) and raw_dets:
                    from app.models import Detection
                    parsed_dets = []
                    for d in raw_dets:
                        if isinstance(d, dict):
                            parsed_dets.append(Detection(
                                class_name=d.get("class_name", d.get("label", d.get("object_type", "truck"))),
                                confidence=float(d.get("confidence", 0.9)),
                                position=Position(
                                    x=float(d.get("position", {}).get("x", 0.0)) if isinstance(d.get("position"), dict) else 0.0,
                                    y=float(d.get("position", {}).get("y", d.get("distance", 5.0))) if isinstance(d.get("position"), dict) else float(d.get("distance", 5.0)),
                                    z=float(d.get("position", {}).get("z", 0.0)) if isinstance(d.get("position"), dict) else 0.0
                                ),
                                direction=d.get("direction", "FRONT"),
                                bbox=d.get("bbox", [0, 0, 100, 100])
                            ))
                        else:
                            parsed_dets.append(d)
                    detections = parsed_dets
                else:
                    detections = []

            # Parse optional GPS location coordinates
            gps_location = None
            gps_raw = data.get("gps")
            if gps_raw and isinstance(gps_raw, dict) and "latitude" in gps_raw and "longitude" in gps_raw:
                try:
                    gps_location = GPSLocation(
                        latitude=float(gps_raw["latitude"]),
                        longitude=float(gps_raw["longitude"]),
                        altitude=float(gps_raw.get("altitude", 0.0) or 0.0),
                        accuracy=float(gps_raw.get("accuracy", 0.0) or 0.0)
                    )
                except Exception as ve:
                    logger.warning(f"Invalid GPS data payload from '{glass_id}': {ve}")

            pos_dict = data.get("position", {})
            pose_dict = data.get("pose", {})
            position = Position(
                x=float(pos_dict.get("x", pose_dict.get("x", 0.0)) if isinstance(pos_dict, dict) and isinstance(pose_dict, dict) else 0.0),
                y=float(pos_dict.get("y", pose_dict.get("y", 0.0)) if isinstance(pos_dict, dict) and isinstance(pose_dict, dict) else 0.0),
                z=float(pos_dict.get("z", pose_dict.get("z", 0.0)) if isinstance(pos_dict, dict) and isinstance(pose_dict, dict) else 0.0)
            )

            uploaded_map = data.get("map") or data.get("local_map")
            tracked_objects = data.get("tracked_objects", [])

            glass_state = GlassState(
                glass_id=glass_id,
                position=position,
                gps=gps_location,
                heading=heading,
                detections=detections,
                tracked_objects=tracked_objects,
                local_map=uploaded_map,
                timestamp=time.time()
            )

            # Update world state and calculate spatial radar blips and threats
            spatial_update = world_manager.update_glass(glass_state)

            # Sync with SharedWorldManager
            raw_pose = data.get("pose")
            if isinstance(raw_pose, list) and len(raw_pose) == 4:
                pose_matrix = np.array(raw_pose, dtype=float)
            else:
                pose_matrix = np.eye(4)

            shared_wm.update_glass_pose(glass_id, pose_matrix, Position3D(position.x, position.y, position.z), heading=heading, gps_info=gps_location)
            
            for d in detections:
                if isinstance(d, dict):
                    det_lbl = d.get('class_name', d.get('label', d.get('object_type', 'hazard')))
                    det_dist = float(d.get('distance', 5.0))
                    det_bearing = float(d.get('bearing', 0.0))
                    det_conf = float(d.get('confidence', 0.9))
                else:
                    det_lbl = getattr(d, 'class_name', getattr(d, 'label', 'hazard'))
                    det_dist = float(getattr(d, 'distance', 5.0))
                    det_bearing = float(getattr(d, 'bearing', 0.0))
                    det_conf = float(getattr(d, 'confidence', 0.9))
                clean_lbl = det_lbl.lower().split(" #")[0].strip()
                if clean_lbl in ["person", "human", "laptop", "phone", "cell phone"]:
                    continue

                rad = math.radians((heading + det_bearing) % 360.0)
                det_x = round(position.x + det_dist * math.sin(rad), 2)
                det_y = round(position.y + det_dist * math.cos(rad), 2)
                det_z = round(position.z, 2)

                threat_id = f"threat_{glass_id}_{det_lbl}"

                shared_wm.add_threat(
                    threat_id=threat_id,
                    object_type=det_lbl,
                    position=Position3D(det_x, det_y, det_z),
                    detected_by_glass_id=glass_id,
                    confidence=det_conf
                )


            # Build response packet for transmitting client
            response_payload = {
                "status": "ok",
                "glass_id": glass_id,
                "heading": heading,
                "gps": gps_location.model_dump() if gps_location else None,
                "detections": [d.model_dump() for d in detections],
                "active_threats": spatial_update["threats"],
                "radar_blips": spatial_update["radar_blips"],
                "all_devices_gps": spatial_update["all_devices_gps"],
                "timestamp": time.time()
            }

            # Send immediate feedback to transmitting client
            await websocket.send_json(response_payload)

            # Broadcast updated World State & Radar Blips to all connected devices
            if spatial_update["threats"] or len(connection_manager.active_connections) > 1:
                await connection_manager.broadcast({
                    "event": "WORLD_STATE_UPDATE",
                    "threats": spatial_update["threats"],
                    "radar_blips": spatial_update["radar_blips"],
                    "all_devices_gps": spatial_update["all_devices_gps"],
                    "timestamp": time.time()
                })

    except WebSocketDisconnect:
        logger.info(f"Client '{glass_id}' disconnected.")
        connection_manager.disconnect(glass_id)
        world_manager.remove_glass(glass_id)
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket loop for '{glass_id}': {e}")
        connection_manager.disconnect(glass_id)
        world_manager.remove_glass(glass_id)
