# test_ws_server.py
import asyncio
from datetime import datetime

import websockets


HOST = "localhost"
PORT = 8765


async def handler(websocket):
    client = websocket.remote_address
    print(f"[CONNECTED] {client}")

    try:
        async for message in websocket:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] RX: {message}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"[DISCONNECTED] {client} | code={e.code} reason={e.reason}")
    except Exception as e:
        print(f"[ERROR] {client} | {e}")
    finally:
        print(f"[CLOSED] {client}")


async def main():
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        print("Waiting for connections...")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")