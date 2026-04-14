import asyncio
import websockets
import json
import time

URI = "ws://172.26.225.185:8765"

# Simulate a slow sweep of the index joint 0.0 → 1.0 → 0.0
async def send_test_commands():
    print(f"Connecting to {URI}...")
    async with websockets.connect(URI) as websocket:
        print("Connected. Sending test packets...")

        import math
        for i in range(60):
            # Sweep index 0→1→0 over 60 steps
            t = i / 59.0
            index_val = math.sin(t * math.pi)  # smooth 0→1→0 arc

            packet = {
                "timestamp": time.time(),
                "source": "hand_exo",
                "joints": {
                    "wrist": 0.4,
                    "wrist2": 0.4,
                    "thumbadd": 1.0,
                    "thumbrot": 0.0,
                    "thumbflex": 0.25,
                    "index": round(index_val, 4),
                    "middle": 0.07,
                    "ring": 0.5,
                    "pinky": 0.6,
                },
            }
            payload = json.dumps(packet)
            print(f"  step {i:02d} | index={index_val:.4f}")
            await websocket.send(payload)
            await asyncio.sleep(0.05)  # 20 Hz — faster than CMD_INTERVAL to exercise rate limiter

        print("Done.")

if __name__ == "__main__":
    asyncio.run(send_test_commands())