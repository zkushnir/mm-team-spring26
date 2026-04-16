import json
import asyncio
import time
import websockets
import rclpy
from hello_helpers.hello_misc import HelloNode

# =============================================================================
# TIMING
# =============================================================================
CMD_INTERVAL  = 0.10   # min seconds between command bursts (~10 Hz)
STALE_TIMEOUT = 2.0    # seconds without a packet → stale warning

# =============================================================================
# MODE SELECTION  (left hand)
# =============================================================================
MODE_ACTIVATE_THRESH = 0.60   # finger flex ≥ this activates that mode
MODE_RELEASE_THRESH  = 0.40   # finger flex < this releases current mode (hysteresis gap)
MODE_DEBOUNCE_TIME   = 0.20   # seconds a candidate mode must be stable before switching
MODE_SWITCH_COOLDOWN = 0.40   # seconds to pause commands after any mode switch

# Priority order (highest first): thumb > ring > middle > index
# Finger keys match packet "left" dict; mode names used internally.
MODE_PRIORITY = [
    ("thumbflex", "gripper"),
    ("ring",      "wrist"),
    ("middle",    "arm"),
    ("index",     "base"),
]

# =============================================================================
# NEUTRAL CALIBRATION  (right hand)
# =============================================================================
NEUTRAL_CALIB_COUNT = 10   # packets to average on startup for right-hand neutral baseline

# =============================================================================
# JOYSTICK PARAMETERS  (right hand)
# =============================================================================
AXIS_DEADBAND  = 0.05   # displacement from neutral below which no command fires
AXIS_SMOOTHING = 0.25   # EMA alpha for right-hand finger values (0=frozen, 1=raw)
JOYSTICK_RANGE = 0.40   # displacement that maps to full-scale max step

# Per-axis max step sizes (physical units per command)
BASE_ROT_MAX_STEP    = 0.06   # radians  — rotate_mobile_base
BASE_TRANS_MAX_STEP  = 0.02   # meters   — translate_mobile_base
LIFT_MAX_STEP        = 0.03   # meters   — joint_lift
ARM_EXT_MAX_STEP     = 0.02   # meters   — wrist_extension
WRIST_PITCH_MAX_STEP = 0.05   # radians  — joint_wrist_pitch
WRIST_YAW_MAX_STEP   = 0.05   # radians  — joint_wrist_yaw
WRIST_ROLL_MAX_STEP  = 0.05   # radians  — joint_wrist_roll
GRIPPER_MAX_STEP     = 0.02   # —         gripper_aperture

# Safe position clamps for incremental joints
LIFT_RANGE        = (0.20, 1.10)
ARM_EXT_RANGE     = (0.00, 0.50)
WRIST_PITCH_RANGE = (-0.90,  0.40)
WRIST_YAW_RANGE   = (-1.40,  1.40)
WRIST_ROLL_RANGE  = (-1.57,  1.57)
GRIPPER_RANGE     = ( 0.00,  0.60)


# =============================================================================
# CONTROL NODE
# =============================================================================

class WebsocketStretchControl(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        # Active mode; None = no mode / no motion
        self._mode: str | None = None
        self._mode_candidate: str | None = None
        self._mode_candidate_since: float = 0.0
        self._mode_switch_time: float = time.monotonic()

        # EMA state for right-hand finger values (keys: "r_index", "r_middle", "r_ring")
        self._smoothed: dict[str, float] = {}

        # Right-hand neutral baseline (populated during startup calibration)
        self._neutral: dict[str, float] = {}
        self._calib_buf: dict[str, list] = {}
        self._calibrated: bool = False

        # Last position commanded per Stretch joint (for incremental control)
        self._last_commanded: dict[str, float] = {}

        self._last_cmd_time: float = 0.0
        self._last_packet_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Low-level: EMA smoother
    # ------------------------------------------------------------------

    def _smooth(self, key: str, raw: float, alpha: float) -> float:
        """Exponential moving average; stores result in self._smoothed[key]."""
        prev = self._smoothed.get(key, raw)
        new = prev + alpha * (raw - prev)
        self._smoothed[key] = new
        return new

    # ------------------------------------------------------------------
    # Neutral calibration
    # ------------------------------------------------------------------

    def _update_calibration(self, right: dict) -> bool:
        """
        Accumulate right-hand samples during startup.
        Returns True once NEUTRAL_CALIB_COUNT packets have been collected and
        the neutral baseline has been locked in.
        """
        for f in ("index", "middle", "ring"):
            if f in right:
                self._calib_buf.setdefault(f, []).append(float(right[f]))

        if all(len(self._calib_buf.get(f, [])) >= NEUTRAL_CALIB_COUNT
               for f in ("index", "middle", "ring")):
            self._neutral = {
                f"r_{f}": sum(self._calib_buf[f][:NEUTRAL_CALIB_COUNT]) / NEUTRAL_CALIB_COUNT
                for f in ("index", "middle", "ring")
            }
            # Seed EMA at neutral so the first steps are smooth
            for k, v in self._neutral.items():
                self._smoothed[k] = v
            self.get_logger().info(
                "Neutral calibrated: "
                + "  ".join(f"{k}={v:.3f}" for k, v in self._neutral.items())
            )
            self._calibrated = True

        return self._calibrated

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------

    def _select_mode(self, left: dict) -> str | None:
        """
        Return the highest-priority active mode based on left-hand flexion.
        Current mode uses release threshold (hysteresis); others use activate threshold.
        Priority: thumb > ring > middle > index.
        """
        for finger, mode_name in MODE_PRIORITY:
            val = float(left.get(finger, 0.0))
            thresh = MODE_RELEASE_THRESH if self._mode == mode_name else MODE_ACTIVATE_THRESH
            if val >= thresh:
                return mode_name
        return None

    def _update_mode(self, left: dict) -> None:
        """Debounce mode transitions; log on change."""
        candidate = self._select_mode(left)
        now = time.monotonic()

        if candidate != self._mode_candidate:
            self._mode_candidate = candidate
            self._mode_candidate_since = now

        if candidate != self._mode and (now - self._mode_candidate_since) >= MODE_DEBOUNCE_TIME:
            prev = self._mode
            self._mode = candidate
            self.get_logger().info(
                f"*** MODE: {(prev or 'NONE').upper()} → {(self._mode or 'NONE').upper()} ***"
            )
            self._mode_switch_time = now

    # ------------------------------------------------------------------
    # Joystick step helper
    # ------------------------------------------------------------------

    def _joystick_step(self, key: str, raw: float, max_step: float) -> float:
        """
        Compute a signed incremental step from a right-hand finger reading.
          key      : smoothed-dict key, e.g. "r_index"
          raw      : latest value from the packet
          max_step : maximum per-command increment at full deflection
        Returns 0.0 when the displacement is within the deadband.
        """
        neutral = self._neutral.get(key, 0.5)
        smoothed = self._smooth(key, raw, AXIS_SMOOTHING)
        displacement = smoothed - neutral

        if abs(displacement) < AXIS_DEADBAND:
            return 0.0

        usable = JOYSTICK_RANGE - AXIS_DEADBAND
        scale = min(1.0, (abs(displacement) - AXIS_DEADBAND) / usable) if usable > 0 else 0.0
        return (1.0 if displacement > 0 else -1.0) * scale * max_step

    # ------------------------------------------------------------------
    # Dispatch guard
    # ------------------------------------------------------------------

    def _can_dispatch(self, label: str) -> bool:
        now = time.monotonic()
        if now - self._last_cmd_time < CMD_INTERVAL:
            return False
        if now - self._mode_switch_time < MODE_SWITCH_COOLDOWN:
            self.get_logger().debug(f"[{label}] Settling after mode switch — skipping.")
            return False
        return True

    # ------------------------------------------------------------------
    # Command dispatch — BASE MODE
    # right index  → rotate_mobile_base  (relative increment)
    # right middle → translate_mobile_base (relative increment)
    # ------------------------------------------------------------------

    def _dispatch_base_mode(self, right: dict) -> None:
        if not self._can_dispatch("BASE"):
            return
        now = time.monotonic()
        sent = False

        rot = self._joystick_step("r_index",  float(right.get("index",  0.5)), BASE_ROT_MAX_STEP)
        if rot != 0.0:
            self.get_logger().info(f"[BASE] rotate={rot:+.4f}rad")
            self.move_to_pose({"rotate_mobile_base": rot}, blocking=False)
            sent = True

        trans = self._joystick_step("r_middle", float(right.get("middle", 0.5)), BASE_TRANS_MAX_STEP)
        if trans != 0.0:
            self.get_logger().info(f"[BASE] translate={trans:+.4f}m")
            self.move_to_pose({"translate_mobile_base": trans}, blocking=False)
            sent = True

        if sent:
            self._last_cmd_time = now

    # ------------------------------------------------------------------
    # Command dispatch — ARM MODE
    # right index  → joint_lift       (incremental absolute)
    # right middle → wrist_extension  (incremental absolute)
    # ------------------------------------------------------------------

    def _dispatch_arm_mode(self, right: dict) -> None:
        if not self._can_dispatch("ARM"):
            return
        now = time.monotonic()

        lift_step = self._joystick_step("r_index",  float(right.get("index",  0.5)), LIFT_MAX_STEP)
        ext_step  = self._joystick_step("r_middle", float(right.get("middle", 0.5)), ARM_EXT_MAX_STEP)

        pose: dict = {}
        if lift_step != 0.0:
            last = self._last_commanded.get("joint_lift", 0.60)
            new  = max(LIFT_RANGE[0], min(LIFT_RANGE[1], last + lift_step))
            if abs(new - last) > 0.001:
                pose["joint_lift"] = new

        if ext_step != 0.0:
            last = self._last_commanded.get("wrist_extension", 0.10)
            new  = max(ARM_EXT_RANGE[0], min(ARM_EXT_RANGE[1], last + ext_step))
            if abs(new - last) > 0.001:
                pose["wrist_extension"] = new

        if pose:
            self.get_logger().info(
                "[ARM] " + "  ".join(f"{k}={v:.4f}" for k, v in pose.items())
            )
            self.move_to_pose(pose, blocking=False)
            self._last_commanded.update(pose)
            self._last_cmd_time = now

    # ------------------------------------------------------------------
    # Command dispatch — WRIST MODE
    # right index  → joint_wrist_pitch (incremental absolute)
    # right middle → joint_wrist_yaw   (incremental absolute)
    # right ring   → joint_wrist_roll  (incremental absolute)
    # ------------------------------------------------------------------

    def _dispatch_wrist_mode(self, right: dict) -> None:
        if not self._can_dispatch("WRIST"):
            return
        now = time.monotonic()

        pitch_step = self._joystick_step("r_index",  float(right.get("index",  0.5)), WRIST_PITCH_MAX_STEP)
        yaw_step   = self._joystick_step("r_middle", float(right.get("middle", 0.5)), WRIST_YAW_MAX_STEP)
        roll_step  = self._joystick_step("r_ring",   float(right.get("ring",   0.5)), WRIST_ROLL_MAX_STEP)

        pose: dict = {}
        if pitch_step != 0.0:
            last = self._last_commanded.get("joint_wrist_pitch", -0.35)
            pose["joint_wrist_pitch"] = max(WRIST_PITCH_RANGE[0], min(WRIST_PITCH_RANGE[1], last + pitch_step))

        if yaw_step != 0.0:
            last = self._last_commanded.get("joint_wrist_yaw", 0.00)
            pose["joint_wrist_yaw"] = max(WRIST_YAW_RANGE[0], min(WRIST_YAW_RANGE[1], last + yaw_step))

        if roll_step != 0.0:
            last = self._last_commanded.get("joint_wrist_roll", 0.00)
            pose["joint_wrist_roll"] = max(WRIST_ROLL_RANGE[0], min(WRIST_ROLL_RANGE[1], last + roll_step))

        if pose:
            self.get_logger().info(
                "[WRIST] " + "  ".join(f"{k}={v:.4f}" for k, v in pose.items())
            )
            self.move_to_pose(pose, blocking=False)
            self._last_commanded.update(pose)
            self._last_cmd_time = now

    # ------------------------------------------------------------------
    # Command dispatch — GRIPPER MODE
    # right index → gripper_aperture (incremental absolute)
    # ------------------------------------------------------------------

    def _dispatch_gripper_mode(self, right: dict) -> None:
        if not self._can_dispatch("GRIPPER"):
            return
        now = time.monotonic()

        step = self._joystick_step("r_index", float(right.get("index", 0.5)), GRIPPER_MAX_STEP)
        if step != 0.0:
            last = self._last_commanded.get("gripper_aperture", 0.25)
            new  = max(GRIPPER_RANGE[0], min(GRIPPER_RANGE[1], last + step))
            if abs(new - last) > 0.001:
                self.get_logger().info(f"[GRIPPER] aperture={new:.4f}")
                self.move_to_pose({"gripper_aperture": new}, blocking=False)
                self._last_commanded["gripper_aperture"] = new
                self._last_cmd_time = now

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def ws_handler(self, websocket):
        self.get_logger().info("WebSocket client connected.")

        async for message in websocket:
            try:
                data = json.loads(message)

                if data.get("side") != "dual":
                    self.get_logger().warning(
                        f"Expected side='dual', got '{data.get('side')}' — skipping."
                    )
                    continue

                left  = data.get("left")
                right = data.get("right")
                if not isinstance(left, dict) or not isinstance(right, dict):
                    self.get_logger().warning("Packet missing 'left' or 'right' dict — skipping.")
                    continue

                self._last_packet_time = time.monotonic()

                # Collect neutral baseline before dispatching any commands
                if not self._calibrated:
                    if not self._update_calibration(right):
                        self.get_logger().debug(
                            f"Calibrating neutral ({sum(len(v) for v in self._calib_buf.values())}"
                            f" / {3 * NEUTRAL_CALIB_COUNT} samples)..."
                        )
                        continue

                # Determine active mode from left hand
                self._update_mode(left)

                self.get_logger().debug(
                    f"mode={self._mode or 'NONE'}  "
                    f"L[idx={left.get('index', 0):.2f} mid={left.get('middle', 0):.2f} "
                    f"ring={left.get('ring', 0):.2f} thumb={left.get('thumbflex', 0):.2f}]  "
                    f"R[idx={right.get('index', 0):.2f} mid={right.get('middle', 0):.2f} "
                    f"ring={right.get('ring', 0):.2f}]"
                )

                if self._mode == "base":
                    self._dispatch_base_mode(right)
                elif self._mode == "arm":
                    self._dispatch_arm_mode(right)
                elif self._mode == "wrist":
                    self._dispatch_wrist_mode(right)
                elif self._mode == "gripper":
                    self._dispatch_gripper_mode(right)
                # else: mode is None → no motion

            except (json.JSONDecodeError, ValueError) as e:
                self.get_logger().warning(f"Malformed packet: {e}")
            except Exception as e:
                self.get_logger().error(f"Unexpected error in ws_handler: {e}")

    # ------------------------------------------------------------------
    # Stale packet watchdog
    # ------------------------------------------------------------------

    async def _stale_watchdog(self):
        """Log a warning if no packets arrive within STALE_TIMEOUT seconds."""
        while True:
            await asyncio.sleep(STALE_TIMEOUT)
            age = time.monotonic() - self._last_packet_time
            if age > STALE_TIMEOUT:
                self.get_logger().warning(
                    f"No exo packet for {age:.1f}s — robot holding last position."
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


# =============================================================================
# Async server bootstrap
# =============================================================================

async def start_websocket_server(node):
    print("Starting WebSocket server on ws://0.0.0.0:8765")
    print(f"  BASE mode    (L index)  : R index → rotation  | R middle → translation")
    print(f"  ARM mode     (L middle) : R index → lift       | R middle → arm extension")
    print(f"  WRIST mode   (L ring)   : R index → pitch      | R middle → yaw | R ring → roll")
    print(f"  GRIPPER mode (L thumb)  : R index → gripper aperture")
    print(f"  Priority: thumb > ring > middle > index")
    print(f"  Neutral calibration : first {NEUTRAL_CALIB_COUNT} packets")
    print(f"  Rate limit          : {CMD_INTERVAL}s  |  Stale timeout: {STALE_TIMEOUT}s")

    async with websockets.serve(node.ws_handler, "0.0.0.0", 8765):
        asyncio.ensure_future(node._stale_watchdog())
        await asyncio.Future()


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
