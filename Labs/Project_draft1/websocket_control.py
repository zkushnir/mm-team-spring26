import json
import asyncio
import time
import websockets
import rclpy
from collections import namedtuple
from hello_helpers.hello_misc import HelloNode

# =============================================================================
# JOINT CONFIG FORMAT
#   stretch  : Stretch joint name passed to move_to_pose()
#   out_min  : minimum output value (meters or radians)
#   out_max  : maximum output value
#   invert   : True → exo 0.0 maps to out_max, exo 1.0 maps to out_min
#   deadband : normalized input change below which the smoothed value is NOT updated
#   smoothing: exponential smoothing alpha (0 = frozen, 1 = raw/no smoothing)
#   delta    : minimum output change (m or rad) required to send a new command
# =============================================================================
JointCfg = namedtuple(
    'JointCfg',
    ['stretch', 'out_min', 'out_max', 'invert', 'deadband', 'smoothing', 'delta']
)

# =============================================================================
# WRIST MODE SWITCHING
# =============================================================================
WRIST_CLOSED_THRESH  = 0.35   # wrist ≤ this → arm control mode
WRIST_OPEN_THRESH    = 0.65   # wrist ≥ this → gripper orientation mode
# The gap [0.35, 0.65] is the hysteresis band — no mode flip inside it
WRIST_SMOOTHING      = 0.20   # low alpha keeps wrist estimate stable
MODE_SWITCH_COOLDOWN = 0.50   # seconds to pause all commands after a mode switch
                              # (also provides a ~0.5 s startup settling window)

# =============================================================================
# GLOBAL TIMING
# =============================================================================
CMD_INTERVAL  = 0.15   # min seconds between command bursts (~6 Hz)
STALE_TIMEOUT = 2.0    # seconds without a packet before a warning is logged

# =============================================================================
# MODE 1 — ARM CONTROL  (wrist ≤ WRIST_CLOSED_THRESH)
#
#   exo joint   Stretch joint        min    max   inv    db     α     Δ
# =============================================================================
ARM_MODE = {
    "index":  JointCfg("joint_lift",       0.45,  0.75, True,  0.01, 0.25, 0.003),
    "middle": JointCfg("wrist_extension",  0.00,  0.30, False, 0.01, 0.25, 0.003),
    # ring  → base rotation via velocity mode (see BASE_ROT_* below)
    # pinky → base translation via velocity mode (see BASE_TRANS_* below)
}

# Ring (arm mode only) — velocity/incremental base rotation
# rotate_mobile_base is a RELATIVE delta, not an absolute position.
# Ring is treated as a joystick: center = stop, deflection = rotate.
BASE_ROT_CENTER    = 0.50   # ring value that means "no rotation"
BASE_ROT_DEADBAND  = 0.15   # normalized dead-zone around center
BASE_ROT_MAX_STEP  = 0.05   # radians per command at full deflection (~3 deg/step)
BASE_ROT_SMOOTHING = 0.25

# Pinky (arm mode only) — velocity/incremental base translation
# translate_mobile_base is a RELATIVE delta, not an absolute position.
# Pinky is treated as a joystick: center = stop, deflection = move.
BASE_TRANS_CENTER    = 0.50   # pinky value that means "no movement"
BASE_TRANS_DEADBAND  = 0.15   # normalized dead-zone around center
BASE_TRANS_MAX_STEP  = 0.02   # meters per command at full deflection
BASE_TRANS_SMOOTHING = 0.25

# =============================================================================
# MODE 2 — GRIPPER ORIENTATION  (wrist ≥ WRIST_OPEN_THRESH)
#
#   exo joint   Stretch joint            min    max   inv    db     α     Δ
# =============================================================================
GRIPPER_MODE = {
    "ring":   JointCfg("joint_wrist_pitch", -0.70,  0.00, False, 0.01, 0.25, 0.010),
    "middle": JointCfg("joint_wrist_roll",  -1.00,  1.00, False, 0.01, 0.25, 0.010),
    "index":  JointCfg("joint_wrist_yaw",   -0.50,  1.20, False, 0.01, 0.25, 0.010),
    "pinky":  JointCfg("gripper_aperture",   0.00,  0.50, False, 0.02, 0.25, 0.005),
}


# =============================================================================
# CONTROL NODE
# =============================================================================

class WebsocketStretchControl(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        self._mode: str = "arm"

        # Initialized to now → first MODE_SWITCH_COOLDOWN seconds no commands fire
        # (acts as a safe startup settling delay)
        self._mode_switch_time: float = time.monotonic()

        # Per-exo-joint exponential moving average, normalized [0, 1]
        self._smoothed: dict[str, float] = {
            j: 0.5 for j in ("wrist", "index", "middle", "ring", "pinky")
        }

        # Per-stretch-joint last value sent; None until the joint is first commanded
        self._last_commanded: dict[str, float] = {}

        self._last_cmd_time: float = 0.0
        self._last_packet_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _smooth(self, joint: str, raw: float, alpha: float) -> float:
        """Exponential moving average update; returns new smoothed value."""
        prev = self._smoothed.get(joint, raw)
        new = prev + alpha * (raw - prev)
        self._smoothed[joint] = new
        return new

    def _map_joint(self, smoothed: float, cfg: JointCfg) -> float:
        """Map a smoothed normalized [0, 1] input to the configured Stretch output range."""
        v = max(0.0, min(1.0, smoothed))
        if cfg.invert:
            v = 1.0 - v
        return cfg.out_min + v * (cfg.out_max - cfg.out_min)

    def _changed(self, stretch_joint: str, new_val: float, threshold: float) -> bool:
        """True if new_val differs from last commanded value by at least threshold."""
        last = self._last_commanded.get(stretch_joint)
        return last is None or abs(new_val - last) >= threshold

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _update_mode(self, wrist_smoothed: float) -> None:
        """Switch mode when wrist crosses a threshold. The gap IS the hysteresis."""
        prev = self._mode
        if self._mode == "arm" and wrist_smoothed >= WRIST_OPEN_THRESH:
            self._mode = "gripper"
        elif self._mode == "gripper" and wrist_smoothed <= WRIST_CLOSED_THRESH:
            self._mode = "arm"

        if self._mode != prev:
            self.get_logger().info(
                f"*** MODE SWITCH: {prev.upper()} → {self._mode.upper()} "
                f"(wrist={wrist_smoothed:.3f}) ***"
            )
            self._mode_switch_time = time.monotonic()

    # ------------------------------------------------------------------
    # Command dispatch — arm control mode
    # ------------------------------------------------------------------

    def _dispatch_arm_mode(self, joints: dict) -> None:
        now = time.monotonic()
        if now - self._last_cmd_time < CMD_INTERVAL:
            return
        if now - self._mode_switch_time < MODE_SWITCH_COOLDOWN:
            self.get_logger().debug("[ARM] Settling after mode switch — holding position.")
            return

        # Build absolute-joint pose dict (lift + arm extension)
        pose: dict = {}
        for exo_j, cfg in ARM_MODE.items():
            if exo_j not in joints:
                continue
            raw = float(joints[exo_j])
            if abs(raw - self._smoothed.get(exo_j, raw)) >= cfg.deadband:
                smoothed = self._smooth(exo_j, raw, cfg.smoothing)
            else:
                smoothed = self._smoothed.get(exo_j, raw)

            target = self._map_joint(smoothed, cfg)
            if self._changed(cfg.stretch, target, cfg.delta):
                pose[cfg.stretch] = target

        sent = False
        if pose:
            self.get_logger().info(
                "[ARM] " + "  ".join(f"{k}={v:.4f}" for k, v in pose.items())
            )
            self.move_to_pose(pose, blocking=False)
            self._last_commanded.update(pose)
            sent = True

        # Ring → base rotation (relative/velocity mode, sent separately)
        if "ring" in joints:
            raw_r = float(joints["ring"])
            if abs(raw_r - self._smoothed.get("ring", raw_r)) >= BASE_ROT_DEADBAND * 0.5:
                r_smooth = self._smooth("ring", raw_r, BASE_ROT_SMOOTHING)
            else:
                r_smooth = self._smoothed.get("ring", raw_r)

            centered_r = r_smooth - BASE_ROT_CENTER
            if abs(centered_r) >= BASE_ROT_DEADBAND:
                scale = (abs(centered_r) - BASE_ROT_DEADBAND) / (0.5 - BASE_ROT_DEADBAND)
                step = (1.0 if centered_r > 0 else -1.0) * scale * BASE_ROT_MAX_STEP
                self.get_logger().info(
                    f"[ARM] rotate_mobile_base={step:+.4f}rad  (ring={r_smooth:.3f})"
                )
                # Note: rotate_mobile_base is a RELATIVE increment, not absolute position
                self.move_to_pose({"rotate_mobile_base": step}, blocking=False)
                sent = True
            else:
                self.get_logger().debug(
                    f"[ARM] ring={r_smooth:.3f} in deadband — rotation skipped"
                )

        # Pinky → base translation (relative/velocity mode, sent separately)
        if "pinky" in joints:
            raw_p = float(joints["pinky"])
            if abs(raw_p - self._smoothed.get("pinky", raw_p)) >= BASE_TRANS_DEADBAND * 0.5:
                p_smooth = self._smooth("pinky", raw_p, BASE_TRANS_SMOOTHING)
            else:
                p_smooth = self._smoothed.get("pinky", raw_p)

            centered = p_smooth - BASE_TRANS_CENTER
            if abs(centered) >= BASE_TRANS_DEADBAND:
                scale = (abs(centered) - BASE_TRANS_DEADBAND) / (0.5 - BASE_TRANS_DEADBAND)
                step = (1.0 if centered > 0 else -1.0) * scale * BASE_TRANS_MAX_STEP
                self.get_logger().info(f"[ARM] translate_mobile_base={step:+.4f}m")
                # Note: translate_mobile_base is a RELATIVE increment, not absolute position
                self.move_to_pose({"translate_mobile_base": step}, blocking=False)
                sent = True

        if sent:
            self._last_cmd_time = now

    # ------------------------------------------------------------------
    # Command dispatch — gripper orientation mode
    # ------------------------------------------------------------------

    def _dispatch_gripper_mode(self, joints: dict) -> None:
        now = time.monotonic()
        if now - self._last_cmd_time < CMD_INTERVAL:
            return
        if now - self._mode_switch_time < MODE_SWITCH_COOLDOWN:
            self.get_logger().debug("[GRIPPER] Settling after mode switch — holding position.")
            return

        # All gripper-mode joints are absolute positions — batch into one call
        pose: dict = {}
        for exo_j, cfg in GRIPPER_MODE.items():
            if exo_j not in joints:
                continue
            raw = float(joints[exo_j])
            if abs(raw - self._smoothed.get(exo_j, raw)) >= cfg.deadband:
                smoothed = self._smooth(exo_j, raw, cfg.smoothing)
            else:
                smoothed = self._smoothed.get(exo_j, raw)

            target = self._map_joint(smoothed, cfg)
            if self._changed(cfg.stretch, target, cfg.delta):
                pose[cfg.stretch] = target

        if pose:
            self.get_logger().info(
                "[GRIPPER] " + "  ".join(f"{k}={v:.4f}" for k, v in pose.items())
            )
            self.move_to_pose(pose, blocking=False)
            self._last_commanded.update(pose)
            self._last_cmd_time = now

    # ------------------------------------------------------------------
    # WebSocket handler — receive path unchanged, dispatch extended
    # ------------------------------------------------------------------

    async def ws_handler(self, websocket):
        self.get_logger().info("WebSocket client connected.")

        async for message in websocket:
            try:
                data = json.loads(message)
                joints = data.get("joints")
                if not isinstance(joints, dict):
                    self.get_logger().warning("Packet missing 'joints' dict — skipping.")
                    continue

                required = ("wrist", "index", "middle", "ring", "pinky")
                missing = [k for k in required if k not in joints]
                if missing:
                    self.get_logger().warning(f"Missing joints {missing} — skipping.")
                    continue

                self._last_packet_time = time.monotonic()

                # Smooth wrist before mode decision to suppress noise-driven flipping
                wrist_s = self._smooth("wrist", float(joints["wrist"]), WRIST_SMOOTHING)
                self._update_mode(wrist_s)

                self.get_logger().debug(
                    f"mode={self._mode}  wrist={wrist_s:.3f}  "
                    f"idx={float(joints['index']):.3f}  mid={float(joints['middle']):.3f}  "
                    f"ring={float(joints['ring']):.3f}  pinky={float(joints['pinky']):.3f}"
                )

                if self._mode == "arm":
                    self._dispatch_arm_mode(joints)
                else:
                    self._dispatch_gripper_mode(joints)

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
    print(f"  ARM mode     (wrist ≤ {WRIST_CLOSED_THRESH}): index→lift | middle→arm | ring→rotation | pinky→translation")
    print(f"  GRIPPER mode (wrist ≥ {WRIST_OPEN_THRESH}): ring→pitch | middle→roll | index→yaw | pinky→gripper")
    print(f"  Hysteresis band : [{WRIST_CLOSED_THRESH}, {WRIST_OPEN_THRESH}]")
    print(f"  Rate limit      : {CMD_INTERVAL}s  |  Stale timeout: {STALE_TIMEOUT}s")

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
