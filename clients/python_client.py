import asyncio
import json
import websockets


async def main():
    uri = "ws://127.0.0.1:8000/ws"
    glass_id = "glass_A"

    sample_packets = [
        {
            "glass_id": glass_id,
            "position": {"x": 5, "y": 10},
            "heading": 90,
            "detections": [
                {"class": "vehicle", "distance": 8, "bearing": -20, "confidence": 0.95},
                {"class": "person", "distance": 3, "bearing": 10, "confidence": 0.88}
            ]
        },
        {
            "glass_id": glass_id,
            "position": {"x": 6, "y": 11},
            "heading": 95,
            "detections": [
                {"class": "vehicle", "distance": 6, "bearing": -15, "confidence": 0.96}
            ]
        }
    ]

    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to Shared Perception Server as [{glass_id}]!")

            for packet in sample_packets:
                payload = json.dumps(packet)
                print(f"📤 Sending update for {glass_id}: {payload}")
                await websocket.send(payload)
                response = await websocket.recv()
                print(f"📩 Server response: {response}\n")
                await asyncio.sleep(2)

            print("Keep connection open. Type custom JSON or press Ctrl+C to exit.")
            while True:
                user_msg = await asyncio.to_thread(input, "Send telemetry packet (or ENTER for sample): ")
                if not user_msg.strip():
                    packet = {
                        "glass_id": glass_id,
                        "position": {"x": 7, "y": 12},
                        "heading": 100,
                        "detections": []
                    }
                    user_msg = json.dumps(packet)
                
                await websocket.send(user_msg)
                resp = await websocket.recv()
                print(f"📩 Server response: {resp}\n")

    except Exception as e:
        print(f"❌ Connection error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
