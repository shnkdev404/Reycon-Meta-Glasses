"""
Live Demo Script Verification:
Launches uvicorn server in a background thread and runs the exact user demo script:
1. Glass B connects & reports truck detection at (12, 0, 0)
2. Glass A connects at (0, 0, 0) with no local detections
3. Glass A receives the alert for the truck detected by Glass B!
"""
import sys
import os
import json
import time
import threading
import asyncio
import websockets
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.shared_world_manager import world_manager

SERVER_PORT = 8009


def start_server():
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT, log_level="error")


async def run_client_demo():
    print("🔄 Resetting SharedWorldManager state...")
    world_manager.reset()

    uri_b = f"ws://127.0.0.1:{SERVER_PORT}/ws/glass/glass_b"
    uri_a = f"ws://127.0.0.1:{SERVER_PORT}/ws/glass/glass_a"

    # Terminal 3 simulation: Glass B connects & reports truck detection
    print("\n[Terminal 3] Glass B connecting & sending telemetry with truck detection...")
    async with websockets.connect(uri_b) as ws_b:
        payload_b = {
            "type": "telemetry",
            "position": {"x": 10, "y": 0, "z": 0},
            "detections": [{
                "object_type": "truck",
                "position": {"x": 12, "y": 0, "z": 0},
                "confidence": 0.9
            }],
            "map_points": []
        }
        await ws_b.send(json.dumps(payload_b))
        res_b = await ws_b.recv()
        print("Glass B response:", json.loads(res_b).get("status"))

    # Terminal 2 simulation: Glass A connects at (0,0,0) with no local detections
    print("\n[Terminal 2] Glass A connecting & sending telemetry (0 local detections)...")
    async with websockets.connect(uri_a) as ws_a:
        payload_a = {
            "type": "telemetry",
            "position": {"x": 0, "y": 0, "z": 0},
            "detections": [],  # A doesn't see threats
            "map_points": []
        }
        await ws_a.send(json.dumps(payload_a))
        res_a = await ws_a.recv()
        alerts_a = json.loads(res_a).get("alerts", [])

        print("\n🎉 Glass A alerts:")
        print(json.dumps(alerts_a, indent=2))

        # Verification
        assert len(alerts_a) > 0, "Glass A should receive alerts!"
        assert alerts_a[0]["type"] == "truck"
        assert alerts_a[0]["detected_by"] == "glass_b"

        print("\nSUCCESS: Glass A knows about threat Glass B detected! 🎉")


def main():
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2.5)  # Wait for uvicorn server startup

    # Run async client demo
    asyncio.run(run_client_demo())


if __name__ == "__main__":
    main()
