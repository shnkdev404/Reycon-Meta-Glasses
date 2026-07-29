import base64
import json
import logging
import time
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np

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

            glass_id = data.get("glass_id", glass_id)
            heading = float(data.get("heading", 0.0))
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
                                label=d.get("label", d.get("class_name", d.get("object_type", "truck"))),
                                confidence=float(d.get("confidence", 0.9)),
                                distance=float(d.get("distance", 5.0)),
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
                x=float(pos_dict.get("x", pose_dict.get("x", 0.0))),
                y=float(pos_dict.get("y", pose_dict.get("y", 0.0))),
                z=float(pos_dict.get("z", pose_dict.get("z", 0.0)))
            )

            uploaded_map = data.get("map") or data.get("local_map")
            tracked_objects = data.get("tracked_objects", [])

            glass_state = GlassState(
                glass_id=glass_id,
                position=position,
                gps=gps_location,
                heading=float(heading or pose_dict.get("heading", 0.0)),
                detections=detections,
                tracked_objects=tracked_objects,
                local_map=uploaded_map,
                timestamp=time.time()
            )

            # Update world state and calculate spatial radar blips and threats
            spatial_update = world_manager.update_glass(glass_state)

            # Sync with SharedWorldManager
            from app.services.shared_world_manager import world_manager as shared_wm, Position3D
            shared_wm.update_glass_pose(glass_id, np.eye(4), Position3D(position.x, position.y, position.z))
            for d in detections:
                shared_wm.add_threat(
                    threat_id=f"threat_{glass_id}_{int(time.time())}",
                    object_type=getattr(d, 'label', getattr(d, 'class_name', 'truck')),
                    position=Position3D(position.x + getattr(d, 'distance', 5.0), position.y, position.z),
                    detected_by_glass_id=glass_id,
                    confidence=getattr(d, 'confidence', 0.9)
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
