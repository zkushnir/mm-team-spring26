import json
import asyncio
import time
import websockets
import rclpy
from hello_helpers.hello_misc import HelloNode

# =============================================================================
# MVP TELEOPERATION CONFIG — edit these before each test run
# =============================================================================
EXO_JOINT         = "index"   # which joint drives the lift
LIFT_MIN          = 0.3      # conservative lower bound (m) — above stow height
LIFT_MAX          = 0.1      # conservative upper bound (m) — well below hard stop
INVERT            = False     # True flips open→close to up→down if needed
DEADBAND          = 0.01      # ignore input changes smaller than this (normalized)
SMOOTHING         = 0.25      # exponential smoothing alpha (0=frozen, 1=raw)
CMD_INTERVAL      = 0.15      # minimum seconds between robot commands (~6 Hz)
CHANGE_THRESHOLD  = 0.003     # minimum lift change (m) before sending a new command
STALE_TIMEOUT     = 2.0       # seconds without a packet before logging a warning
# =============================================================================


class WebsocketStretchControl(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        # Smoothed normalized input (0.0–1.0); starts at mid-range
        self._smoothed_input: float = 0.5

        # Last lift position actually commanded to the robot
        self._last_commanded_lift: float | None = None

        # Timestamps for rate limiting and stale detection
        self._last_cmd_time: float = 0.0
        self._last_packet_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _apply_smoothing(self, raw: float) -> float:
        """Exponential moving average. Low alpha = heavy smoothing."""
        self._smoothed_input += SMOOTHING * (raw - self._smoothed_input)
        return self._smoothed_input

    def _map_to_lift(self, normalized: float) -> float:
        """Map a normalized [0, 1] input to the safe lift range."""
        normalized = max(0.0, min(1.0, normalized))   # clamp
        if INVERT:
            normalized = 1.0 - normalized
        return LIFT_MIN + normalized * (LIFT_MAX - LIFT_MIN)

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def ws_handler(self, websocket, *args, **kwargs):
        self.get_logger().info("WebSocket client connected.")

        async for message in websocket:
            try:
                data = json.loads(message)

                # Validate expected structure
                joints = data.get("joints")
                if not isinstance(joints, dict):
                    self.get_logger().warning("Packet missing 'joints' dict — skipping.")
                    continue

                if EXO_JOINT not in joints:
                    self.get_logger().warning(
                        f"Joint '{EXO_JOINT}' not found in packet (keys: {list(joints.keys())}) — skipping."
                    )
                    continue

                raw_input = float(joints[EXO_JOINT])
                self._last_packet_time = time.monotonic()

                # Deadband: ignore tiny fluctuations around the smoothed value
                if abs(raw_input - self._smoothed_input) < DEADBAND:
                    continue

                # Smooth the input
                smoothed = self._apply_smoothing(raw_input)
                target_lift = self._map_to_lift(smoothed)

                # Rate limit
                now = time.monotonic()
                if now - self._last_cmd_time < CMD_INTERVAL:
                    continue

                # Change threshold: skip if the target hasn't moved meaningfully
                if (
                    self._last_commanded_lift is not None
                    and abs(target_lift - self._last_commanded_lift) < CHANGE_THRESHOLD
                ):
                    continue

                # All checks passed — send the command
                self.get_logger().info(
                    f"[{EXO_JOINT}] raw={raw_input:.4f}  smoothed={smoothed:.4f}  "
                    f"lift={target_lift:.4f}m"
                )
                self.move_to_pose({"joint_lift": target_lift}, blocking=False)
                self._last_commanded_lift = target_lift
                self._last_cmd_time = now

            except (json.JSONDecodeError, ValueError) as e:
                self.get_logger().warning(f"Malformed packet: {e}")
            except Exception as e:
                self.get_logger().error(f"Unexpected error in ws_handler: {e}")

    # ------------------------------------------------------------------
    # Stale-packet watchdog (runs as a background asyncio task)
    # ------------------------------------------------------------------

    async def _stale_watchdog(self):
        """Log a warning if no packets arrive within STALE_TIMEOUT seconds."""
        while True:
            await asyncio.sleep(STALE_TIMEOUT)
            age = time.monotonic() - self._last_packet_time
            if age > STALE_TIMEOUT:
                self.get_logger().warning(
                    f"No exo packet received for {age:.1f}s — robot holding last position."
                )

    # ------------------------------------------------------------------
    # Node entry point
    # ------------------------------------------------------------------

    def main(self):
        HelloNode.main(
            self, "ws_stretch_control", "ws_stretch_control",
            wait_for_first_pointcloud=False
        )
        self.stow_the_robot()


# ----------------------------------------------------------------------
# Async server bootstrap
# ----------------------------------------------------------------------

async def start_websocket_server(node):
    print(f"Starting WebSocket server on ws://0.0.0.0:8765")
    print(f"  Driving joint : {EXO_JOINT}")
    print(f"  Lift range    : {LIFT_MIN}m – {LIFT_MAX}m")
    print(f"  Smoothing α   : {SMOOTHING}  |  Rate limit: {CMD_INTERVAL}s")

    async with websockets.serve(node.ws_handler, "0.0.0.0", 8765):
        asyncio.ensure_future(node._stale_watchdog())
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    node = WebsocketStretchControl()
    node.main()

    try:
        asyncio.run(start_websocket_server(node))
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        rclpy.try_shutdown()
        if hasattr(node, "new_thread") and node.new_thread is not None:
            node.new_thread.join()
