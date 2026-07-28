"""
Live Collaborative Threat Scenario Simulator with Automatic Reconnection.

Connects Glass A and Glass B to the central server over WebSocket (wss://).
Glass A continuously tracks a moving truck advancing toward Glass B.
The central server predicts collision, calculates TTC, and dispatches directed alerts to Glass B.
"""
import sys
import os
import ssl
import json
import asyncio
import websockets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# SSL Context accepting local self-signed cert
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


async def run_glass_b():
    uri = "wss://127.0.0.1:8000/ws/mobile?glass_id=glass_B"

    while True:
        try:
            print("👓 [Glass B] Connecting to Shared Perception Server...")
            async with websockets.connect(uri, ssl=ssl_context, ping_interval=10, ping_timeout=10) as ws:
                print("👓 [Glass B] Connected! Listening for directed alerts...")
                while True:
                    payload_b = {
                        "glass_id": "glass_B",
                        "pose": {"x": 10.0, "y": 10.0, "z": 1.65, "heading": 0.0},
                        "heading": 0.0,
                        "gps": {"latitude": 28.6139, "longitude": 77.2090},
                        "detections": []
                    }
                    await ws.send(json.dumps(payload_b))

                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.3)
                        data = json.loads(msg)
                        if data.get("type") == "THREAT_ALERT":
                            alert = data.get("alert", {})
                            print("\n" + "="*60)
                            print(f"🚨 DIRECTED ALERT RECEIVED BY GLASS B:")
                            print(f"   Warning:  {alert.get('warning_message')}")
                            print(f"   Level:    {alert.get('threat_level')}")
                            print(f"   Distance: {alert.get('distance')}m | TTC: {alert.get('time_to_collision')}s")
                            print("="*60 + "\n")
                    except asyncio.TimeoutError:
                        pass

                    await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Glass B reconnecting in 2s ({e})...")
            await asyncio.sleep(2)


async def run_glass_a():
    await asyncio.sleep(1.5)  # Wait for Glass B to connect first
    uri = "wss://127.0.0.1:8000/ws/mobile?glass_id=glass_A"
    truck_y = 0.0

    while True:
        try:
            print("\n👓 [Glass A] Connecting to Shared Perception Server...")
            async with websockets.connect(uri, ssl=ssl_context, ping_interval=10, ping_timeout=10) as ws:
                print("👓 [Glass A] Connected! Streaming continuous truck tracking...")
                while True:
                    truck_y += 0.5
                    if truck_y > 12.0:
                        truck_y = 0.0  # Loop trajectory back to start

                    payload_a = {
                        "glass_id": "glass_A",
                        "pose": {"x": 0.0, "y": 0.0, "z": 1.65, "heading": 0.0},
                        "heading": 0.0,
                        "gps": {"latitude": 28.6138, "longitude": 77.2089},
                        "tracked_objects": [
                            {
                                "object_id": "obj_truck_1",
                                "label": "truck #1",
                                "confidence": 0.95,
                                "position_x": 10.0,
                                "position_y": round(truck_y, 2),
                                "position_z": 0.0,
                                "velocity_x": 0.0,
                                "velocity_y": 1.5,
                                "source_glasses": ["glass_A"]
                            }
                        ]
                    }

                    await ws.send(json.dumps(payload_a))
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except asyncio.TimeoutError:
                        pass

                    print(f"📡 [Glass A] Telemetry stream: Truck advancing at (10.0m, {truck_y:.1f}m)...")
                    await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Glass A reconnecting in 2s ({e})...")
            await asyncio.sleep(2)


async def main():
    print("🚀 LAUNCHING LIVE COLLABORATIVE PERCEPTION & THREAT SIMULATION...")
    print("🌐 Server Command Center UI: https://192.168.225.62:8000/server")
    print("📱 Mobile Client HUD:       https://192.168.225.62:8443/mobile_client.html\n")

    await asyncio.gather(
        run_glass_b(),
        run_glass_a()
    )


if __name__ == "__main__":
    asyncio.run(main())
