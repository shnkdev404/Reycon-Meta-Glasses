"""
Simulates the Kaya Hackathon Scenario:
1. Glass B connects & stands at (10, 10) facing North (0°). Glass B sees no objects.
2. Glass A connects at (5, 2) facing East (90°). Glass A detects a vehicle heading toward Glass B.
3. Server receives Glass A's telemetry, projects vehicle into World Coordinates at (10, 2),
   computes threat trajectory toward Glass B, and dispatches a warning EXCLUSIVELY to Glass B!
"""
import sys
import asyncio
import json
import websockets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def simulate_glass_b():
    uri = "ws://127.0.0.1:8000/ws?glass_id=glass_B"
    print("👓 [Glass B] Connecting to Shared Perception Server...")
    
    async with websockets.connect(uri) as ws:
        # Glass B telemetry payload (Position: 10, 10, Heading: 0°)
        payload_b = {
            "glass_id": "glass_B",
            "position": {"x": 10.0, "y": 10.0, "z": 0.0},
            "heading": 0.0,  # Facing North
            "detections": [] # Cannot see behind itself
        }
        await ws.send(json.dumps(payload_b))
        resp = await ws.recv()
        print(f"👓 [Glass B] Initialized state: {resp}")

        # Listen for directed threat warnings
        print("👓 [Glass B] Listening for directed threat warnings...")
        try:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if data.get("type") == "THREAT_ALERT":
                    alert = data["alert"]
                    print("\n" + "="*50)
                    print(f"🚨 DIRECTED ALERT RECEIVED BY GLASS B:")
                    print(f"   Message:  {alert['warning_message']}")
                    print(f"   Trigger:  {alert['threat_type']} (Object: {alert['trigger_object_id']})")
                    print(f"   Distance: {alert['distance']}m | Bearing: {alert['bearing']}° | TTC: {alert['time_to_collision']}s")
                    print("="*50 + "\n")
        except websockets.exceptions.ConnectionClosed:
            print("👓 [Glass B] Connection closed.")


async def simulate_glass_a():
    await asyncio.sleep(2)  # Wait for Glass B to connect first
    uri = "ws://127.0.0.1:8000/ws?glass_id=glass_A"
    print("\n👓 [Glass A] Connecting to Shared Perception Server...")

    async with websockets.connect(uri) as ws:
        # Glass A telemetry (Position: 5, 2, Heading: 90° East)
        # Glass A sees a vehicle 5m straight ahead at (10, 2), moving North towards Glass B (10, 10)
        payload_a = {
            "glass_id": "glass_A",
            "position": {"x": 5.0, "y": 2.0, "z": 0.0},
            "heading": 90.0,
            "detections": [
                {
                    "class": "vehicle",
                    "distance": 5.0,
                    "bearing": 0.0,  # Directly ahead of Glass A
                    "confidence": 0.96
                }
            ]
        }
        print(f"👓 [Glass A] Streaming telemetry with vehicle detection...")
        await ws.send(json.dumps(payload_a))
        resp = await ws.recv()
        print(f"👓 [Glass A] Server response: {resp}")
        await asyncio.sleep(3)


async def main():
    print("🚀 Starting Shared Perception Threat Scenario Simulation...\n")
    await asyncio.gather(
        simulate_glass_b(),
        simulate_glass_a()
    )


if __name__ == "__main__":
    asyncio.run(main())
