"""
Example Client script: Connecting smart glasses to the central server
and sending 3D spatial telemetry, detections, and landmark points.
"""
import asyncio
import json
import websockets

async def glass_telemetry():
    uri = "ws://localhost:8000/ws/glass/glass_001"
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as ws:
        # Send telemetry
        payload = {
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
        
        print("Sending telemetry...")
        await ws.send(json.dumps(payload))
        
        # Receive alerts (including cross-glass threats!)
        response = await ws.recv()
        data = json.loads(response)
        print(f"✅ Response received!")
        print(f"🚨 Alerts ({len(data.get('alerts', []))}): {data.get('alerts')}")
        print(f"📊 Map Stats: {data.get('map_stats')}")

if __name__ == "__main__":
    asyncio.run(glass_telemetry())
