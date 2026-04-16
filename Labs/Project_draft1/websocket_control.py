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
MODE_ACTIVATE_THRESH = 0.70   # finger flex ≥ this activates a new mode
MODE_RELEASE_THRESH  = 0.30   # current-mode finger must drop below this to release
                               # The gap [0.40, 0.60] is the per-finger hysteresis band.
MODE_DEBOUNCE_TIME   = 0.20   # seconds a candidate mode must be stable before switching
MODE_SWITCH_COOLDOWN = 0.40   # seconds to suppress commands after any mode switch

# Priority order (highest → lowest): thumb > ring > middle > index.
# When multiple left-hand fingers are active, the highest-priority one wins.
# The CURRENT mode's finger uses RELEASE_THRESH; all others use ACTIVATE_THRESH.
MODE_PRIORITY = [
    ("thumbflex", "gripper"),
    ("ring",      "wrist"),
    ("middle",    "arm"),
    ("index",     "base"),
]

# =============================================================================
# RIGHT-HAND JOYSTICK NEUTRAL  (fixed, not learned from startup)
# 0.5 = mid-range of the normalized [0, 1] sensor output.
# Displacement above neutral → positive step; below → negative step.
# Adjust any axis here if your glove's resting posture is off-center.
# =============================================================================
RIGHT_INDEX_NEUTRAL  = 0.5
RIGHT_MIDDLE_NEUTRAL = 0.5
RIGHT_RING_NEUTRAL   = 0.5

# =============================================================================
# JOYSTICK PARAMETERS  (right hand)
# =============================================================================
AXIS_DEADBAND  = 0.15   # displacement from neutral below which no command fires
AXIS_SMOOTHING = 0.25   # EMA alpha for right-hand finger values (0=frozen, 1=raw)
JOYSTICK_RANGE = 0.40   # displacement that produces full-scale step

# Per-axis max step sizes (physical units per command burst)
BASE_ROT_MAX_STEP    = 0.16   # radians — rotate_mobile_base
BASE_TRANS_MAX_STEP  = 0.12   # meters  — translate_mobile_base
LIFT_MAX_STEP        = 0.05   # meters  — joint_lift
ARM_EXT_MAX_STEP     = 0.05   # meters  — wrist_extension
WRIST_PITCH_MAX_STEP = 0.15   # radians — joint_wrist_pitch
WRIST_YAW_MAX_STEP   = 0.15   # radians — joint_wrist_yaw
WRIST_ROLL_MAX_STEP  = 0.15   # radians — joint_wrist_roll
GRIPPER_MAX_STEP     = 0.06   #         — gripper_aperture

# Safe absolute position clamps for incremental joints
LIFT_RANGE        = (0.20, 1.10)
ARM_EXT_RANGE     = (0.00, 0.50)
WRIST_PITCH_RANGE = (-0.90,  0.40)
WRIST_YAW_RANGE   = (-1.40,  1.40)
WRIST_ROLL_RANGE  = (-1.57,  1.57)
GRIPPER_RANGE     = ( 0.00,  0.60)

# =============================================================================
# STOW POSE
# Seeds _last_commanded so the very first incremental command starts from the
# correct position and produces no jump.  These values must match what
# stow_the_robot() leaves the robot in; update here if the startup sequence
# changes.
# =============================================================================
STOW_POSE = {
    "joint_lift":        0.20,   # meters  — stow height
    "wrist_extension":   0.00,   # meters  — fully retracted
    # Wrist joints: use physical neutral (0.0), NOT the range hard-stops.
    # These must be close to the actual robot position after stow_the_robot()
    # so the first incremental wrist command does not produce a large jump.
    # Verify against your specific Stretch configuration and update if needed.
    "joint_wrist_pitch":  0.00,  # radians — neutral horizontal  (range -0.90 to 0.40)
    "joint_wrist_yaw":    0.00,  # radians — centered            (range -1.40 to 1.40)
    "joint_wrist_roll":   0.00,  # radians — neutral roll        (range -1.57 to 1.57)
    "gripper_aperture":   0.00,  #         — closed
}


# =============================================================================
# CONTROL NODE
# =============================================================================

class WebsocketStretchControl(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        # Active mode; None = no mode / no motion commanded
        self._mode: str | None = None
        self._mode_candidate: str | None = None
        self._mode_candidate_since: float = 0.0
        self._mode_switch_time: float = time.monotonic()

        # Right-hand neutral — fixed at the configured constants, not learned from startup.
        # EMA is seeded at neutral so the first displacement reads zero.
        self._neutral: dict[str, float] = {
            "r_index":  RIGHT_INDEX_NEUTRAL,
            "r_middle": RIGHT_MIDDLE_NEUTRAL,
            "r_ring":   RIGHT_RING_NEUTRAL,
        }
        # EMA state for right-hand finger values (keys match _neutral above).
        # Updated on EVERY valid packet so values are always current when dispatch fires.
        self._smoothed: dict[str, float] = dict(self._neutral)

        # Last position commanded per Stretch joint — drives incremental control.
        # Pre-seeded from STOW_POSE; never relies on arbitrary fallback defaults.
        self._last_commanded: dict[str, float] = dict(STOW_POSE)

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
    # Right-hand EMA: advance every packet regardless of active mode
    # ------------------------------------------------------------------

    def _update_right_smoothed(self, right: dict) -> None:
        """
        Run EMA on right-hand fingers on every valid packet.
        Keeping this separate from the dispatch path means smoothed values are
        already converged when a mode activates — no stale-state jump on the
        first command burst after a quiet period or a mode switch.
        """
        for finger, key in (("index", "r_index"), ("middle", "r_middle"), ("ring", "r_ring")):
            raw = right.get(finger)
            if raw is not None:
                self._smooth(key, float(raw), AXIS_SMOOTHING)

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------

    def _select_mode(self, left: dict) -> str | None:
        """
        Return the highest-priority active mode based on left-hand flexion.
        The currently active mode's finger uses MODE_RELEASE_THRESH; every
        other finger uses MODE_ACTIVATE_THRESH.  This means:
          - the active mode stays latched until its own finger is released, and
          - a higher-priority finger must be clearly held (≥ ACTIVATE) to override,
            after which MODE_DEBOUNCE_TIME further filters accidental contacts.
        """
        for finger, mode_name in MODE_PRIORITY:
            val = float(left.get(finger, 0.0))
            thresh = MODE_RELEASE_THRESH if self._mode == mode_name else MODE_ACTIVATE_THRESH
            if val >= thresh:
                return mode_name
        return None

    def _update_mode(self, left: dict) -> None:
        """Debounce mode transitions; log on every confirmed change."""
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
    # Joystick step helper  (read-only: EMA already advanced by _update_right_smoothed)
    # ------------------------------------------------------------------

    def _joystick_step(self, key: str, max_step: float, label: str) -> float:
        """
        Compute a signed incremental step from a pre-smoothed right-hand value.
          key      : "r_index" / "r_middle" / "r_ring"
          max_step : maximum per-command increment at full deflection
          label    : mode tag used in debug log lines

        Logs displacement and outcome (deadband suppressed or scaled step) at DEBUG.
        Returns 0.0 when displacement is within AXIS_DEADBAND.
        """
        neutral  = self._neutral.get(key, 0.5)
        smoothed = self._smoothed.get(key, neutral)
        disp     = smoothed - neutral

        if abs(disp) < AXIS_DEADBAND:
            self.get_logger().debug(
                f"[{label}] {key} disp={disp:+.3f} → deadband (|d|<{AXIS_DEADBAND})"
            )
            return 0.0

        usable = JOYSTICK_RANGE - AXIS_DEADBAND
        scale  = min(1.0, (abs(disp) - AXIS_DEADBAND) / usable) if usable > 0 else 0.0
        step   = (1.0 if disp > 0 else -1.0) * scale * max_step

        self.get_logger().debug(
            f"[{label}] {key} disp={disp:+.3f} scale={scale:.2f} step={step:+.5f}"
        )
        return step

    # ------------------------------------------------------------------
    # Dispatch guard
    # ------------------------------------------------------------------

    def _can_dispatch(self, label: str) -> bool:
        now = time.monotonic()
        if now - self._last_cmd_time < CMD_INTERVAL:
            self.get_logger().debug(f"[{label}] Rate limited — skipping.")
            return False
        if now - self._mode_switch_time < MODE_SWITCH_COOLDOWN:
            self.get_logger().debug(f"[{label}] Mode-switch cooldown — skipping.")
            return False
        return True

    # ------------------------------------------------------------------
    # Command dispatch — BASE MODE
    # right index  → rotate_mobile_base   (relative increment)
    # right middle → translate_mobile_base (relative increment)
    # ------------------------------------------------------------------

    def _dispatch_base_mode(self) -> None:
        if not self._can_dispatch("BASE"):
            return
        now = time.monotonic()
        sent = False

        rot = self._joystick_step("r_index", BASE_ROT_MAX_STEP, "BASE")
        if rot != 0.0:
            self.get_logger().info(f"[BASE] rotate={rot:+.4f}rad")
            self.move_to_pose({"rotate_mobile_base": rot}, blocking=False)
            sent = True

        trans = self._joystick_step("r_middle", BASE_TRANS_MAX_STEP, "BASE")
        if trans != 0.0:
            self.get_logger().info(f"[BASE] translate={trans:+.4f}m")
            self.move_to_pose({"translate_mobile_base": trans}, blocking=False)
            sent = True

        if sent:
            self._last_cmd_time = now

    # ------------------------------------------------------------------
    # Command dispatch — ARM MODE
    # right index  → joint_lift      (joystick incremental)
    # right middle → wrist_extension (joystick incremental)
    # ------------------------------------------------------------------

    def _dispatch_arm_mode(self) -> None:
        if not self._can_dispatch("ARM"):
            return
        now = time.monotonic()

        lift_step = self._joystick_step("r_index",  LIFT_MAX_STEP,    "ARM")
        ext_step  = self._joystick_step("r_middle", ARM_EXT_MAX_STEP, "ARM")

        pose: dict = {}
        if lift_step != 0.0:
            last = self._last_commanded["joint_lift"]
            new  = max(LIFT_RANGE[0], min(LIFT_RANGE[1], last + lift_step))
            if abs(new - last) > 0.001:
                pose["joint_lift"] = new

        if ext_step != 0.0:
            last = self._last_commanded["wrist_extension"]
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
    # right index  → joint_wrist_pitch (joystick incremental)
    # right middle → joint_wrist_yaw   (joystick incremental)
    # right ring   → joint_wrist_roll  (joystick incremental)
    # ------------------------------------------------------------------

    def _dispatch_wrist_mode(self) -> None:
        if not self._can_dispatch("WRIST"):
            return
        now = time.monotonic()

        pitch_step = self._joystick_step("r_index",  WRIST_PITCH_MAX_STEP, "WRIST")
        yaw_step   = self._joystick_step("r_middle", WRIST_YAW_MAX_STEP,   "WRIST")
        roll_step  = self._joystick_step("r_ring",   WRIST_ROLL_MAX_STEP,  "WRIST")

        pose: dict = {}

        if pitch_step != 0.0:
            last = self._last_commanded["joint_wrist_pitch"]
            new  = max(WRIST_PITCH_RANGE[0], min(WRIST_PITCH_RANGE[1], last + pitch_step))
            if abs(new - last) > 0.001:
                self.get_logger().info(
                    f"[WRIST] pitch  smoothed={self._smoothed.get('r_index', 0.5):.3f}"
                    f"  disp={self._smoothed.get('r_index', 0.5) - self._neutral['r_index']:+.3f}"
                    f"  step={pitch_step:+.4f}  target={new:.4f}"
                )
                pose["joint_wrist_pitch"] = new

        if yaw_step != 0.0:
            last = self._last_commanded["joint_wrist_yaw"]
            new  = max(WRIST_YAW_RANGE[0], min(WRIST_YAW_RANGE[1], last + yaw_step))
            if abs(new - last) > 0.001:
                self.get_logger().info(
                    f"[WRIST] yaw    smoothed={self._smoothed.get('r_middle', 0.5):.3f}"
                    f"  disp={self._smoothed.get('r_middle', 0.5) - self._neutral['r_middle']:+.3f}"
                    f"  step={yaw_step:+.4f}  target={new:.4f}"
                )
                pose["joint_wrist_yaw"] = new

        if roll_step != 0.0:
            last = self._last_commanded["joint_wrist_roll"]
            new  = max(WRIST_ROLL_RANGE[0], min(WRIST_ROLL_RANGE[1], last + roll_step))
            if abs(new - last) > 0.001:
                self.get_logger().info(
                    f"[WRIST] roll   smoothed={self._smoothed.get('r_ring', 0.5):.3f}"
                    f"  disp={self._smoothed.get('r_ring', 0.5) - self._neutral['r_ring']:+.3f}"
                    f"  step={roll_step:+.4f}  target={new:.4f}"
                )
                pose["joint_wrist_roll"] = new

        if pose:
            self.move_to_pose(pose, blocking=False)
            self._last_commanded.update(pose)
            self._last_cmd_time = now

    # ------------------------------------------------------------------
    # Command dispatch — GRIPPER MODE
    # right index → gripper_aperture (joystick incremental)
    # ------------------------------------------------------------------

    def _dispatch_gripper_mode(self) -> None:
        if not self._can_dispatch("GRIPPER"):
            return
        now = time.monotonic()

        step = self._joystick_step("r_index", GRIPPER_MAX_STEP, "GRIPPER")
        if step != 0.0:
            last = self._last_commanded["gripper_aperture"]
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

                # Phase 1: advance right-hand EMA so smoothed values are always current
                self._update_right_smoothed(right)

                # Phase 2: resolve active mode from left hand
                self._update_mode(left)

                self.get_logger().debug(
                    f"mode={self._mode or 'NONE'}  "
                    f"L[idx={left.get('index', 0):.2f} mid={left.get('middle', 0):.2f} "
                    f"ring={left.get('ring', 0):.2f} thumb={left.get('thumbflex', 0):.2f}]  "
                    f"R[idx={right.get('index', 0):.2f} mid={right.get('middle', 0):.2f} "
                    f"ring={right.get('ring', 0):.2f}]"
                )

                # Phase 3: dispatch — only when a mode is active
                if self._mode == "base":
                    self._dispatch_base_mode()
                elif self._mode == "arm":
                    self._dispatch_arm_mode()
                elif self._mode == "wrist":
                    self._dispatch_wrist_mode()
                elif self._mode == "gripper":
                    self._dispatch_gripper_mode()
                # self._mode is None → no motion, _last_commanded never changes

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
    print(f"  Joystick neutral    : index={RIGHT_INDEX_NEUTRAL}  middle={RIGHT_MIDDLE_NEUTRAL}  ring={RIGHT_RING_NEUTRAL}")
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
