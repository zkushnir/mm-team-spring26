import asyncio
import websockets
import json

async def send_test_command():
    uri = "ws://172.26.225.185:8765" 

    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as websocket:
        # half of tentative input range, maps to 0.6 meters on the lift for now
        test_data = {"lift_input": 100.0}
        payload = json.dumps(test_data)

        print(f"Sending payload: {payload}")
        await websocket.send(payload)
        print("Command sent successfully!")

if __name__ == "__main__":
    asyncio.run(send_test_command())