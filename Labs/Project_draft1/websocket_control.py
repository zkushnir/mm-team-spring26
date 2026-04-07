import json
import asyncio
import websockets
import rclpy
from hello_helpers.hello_misc import HelloNode

class WebsocketStretchControl(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)
        
        # Input range (subject to change)
        self.input_min = 0.0
        self.input_max = 100.0
        
        # Stretch arm lift physical limits
        self.lift_min = 0.2
        self.lift_max = 1.0

    def map_value(self, val):
        # clamp input to not exceed physical limits
        val = max(self.input_min, min(self.input_max, val))
        # Linear interpolation
        mapped_lift = self.lift_min + (val - self.input_min) * (self.lift_max - self.lift_min) / (self.input_max - self.input_min)
        return mapped_lift

    async def ws_handler(self, websocket, *args, **kwargs):
        self.get_logger().info("WebSocket Client Connected!")
        
        async for message in websocket:
            try:
                # Parse incoming data structure, check if expected key is present
                data = json.loads(message)
                
                if 'lift_input' in data:
                    raw_val = float(data['lift_input'])
                    target_lift = self.map_value(raw_val)
                    
                    self.get_logger().info(f"Received Input: {raw_val} | Mapped Lift Command: {target_lift:.3f}m")
                    
                    # Command position the robot
                    self.move_to_pose({'joint_lift': target_lift}, blocking=False)
                    
            except json.JSONDecodeError:
                self.get_logger().warning("Received malformed JSON.")
            except Exception as e:
                self.get_logger().error(f"Error processing message: {e}")

    def main(self):
        # Initialize HelloNode architecture and ROS 2 communication
        HelloNode.main(self, 'ws_stretch_control', 'ws_stretch_control', wait_for_first_pointcloud=False)

        self.stow_the_robot()


async def start_websocket_server(node):
    print("Starting WebSocket Server on ws://0.0.0.0:8765...")
    async with websockets.serve(node.ws_handler, "0.0.0.0", 8765):
        await asyncio.Future()  # Keeps the async server running forever

if __name__ == '__main__':
    # Start the ROS 2 node in the background
    node = WebsocketStretchControl()
    node.main()
    
    # Start the asyncio event loop and WebSocket server on the main thread
    try:
        asyncio.run(start_websocket_server(node))
    except KeyboardInterrupt:
        print("\nShutting down WebSocket server and ROS node...")
    finally:
        # Exit the HelloNode ROS thread and core ROS runtime
        rclpy.try_shutdown()
        if hasattr(node, 'new_thread') and node.new_thread is not None:
            node.new_thread.join()