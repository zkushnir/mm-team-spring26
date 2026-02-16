#!/usr/bin/env python3
"""
Inverse Kinematics control for Stretch 3 via ROS 2.

This script replaces the stretch_body Python API with ROS 2 interfaces,
while keeping the same ikpy-based IK solver from the original script.

PREREQUISITES:
  1. Install dependencies (on the Stretch):
       pip3 install ikpy urchin --break-system-packages
  2. Launch the Stretch driver in a separate terminal:
       ros2 launch stretch_core stretch_driver.launch.py mode:=position
  3. Then run this script:
       python3 ik_ros2.py

ARCHITECTURE OVERVIEW:
  - Original script used `stretch_body.robot.Robot()` to directly talk to hardware.
  - This script uses ROS 2 topics and actions instead:
      * Reads joint positions from `/stretch/joint_states` topic (sensor_msgs/JointState)
      * Sends motion commands via HelloNode's `move_to_pose()` method, which internally
        uses the `/stretch_controller/follow_joint_trajectory` action server.
  - The URDF cleanup and ikpy chain construction are IDENTICAL to the original.

KEY JOINT NAME MAPPING (stretch_body → ROS 2):
  ┌──────────────────────────┬──────────────────────────────────────────────┐
  │ Original (stretch_body)  │ ROS 2 (joint_states topic / move_to_pose)   │
  ├──────────────────────────┼──────────────────────────────────────────────┤
  │ robot.base.rotate_by()   │ move_to_pose({'rotate_mobile_base': x})     │
  │ robot.base.translate_by()│ move_to_pose({'translate_mobile_base': x})  │
  │ robot.lift.status['pos'] │ joint_state['joint_lift']                   │
  │ robot.arm.status['pos']  │ joint_state['wrist_extension']  (total arm) │
  │ robot.end_of_arm 'yaw'   │ joint_state['joint_wrist_yaw']              │
  │ robot.end_of_arm 'pitch' │ joint_state['joint_wrist_pitch']            │
  │ robot.end_of_arm 'roll'  │ joint_state['joint_wrist_roll']             │
  └──────────────────────────┴──────────────────────────────────────────────┘

  VIRTUAL BASE CHAIN (added to URDF for IK):
    base_link → [revolute Z: base rotation] → link_base_rotation
             → [prismatic X: base translation] → link_base_translation
             → [fixed: mast] → link_mast → [prismatic: lift] → ...

  EXECUTION ORDER: The robot ROTATES first, then TRANSLATES + moves arm/wrist.
  This is safer and matches diff-drive kinematics (turn to face, then drive).
"""

import numpy as np
import ikpy.chain
import ikpy.utils.geometry
import urchin as urdfpy
import importlib.resources as importlib_resources
import time
import rclpy
from hello_helpers.hello_misc import HelloNode


# =============================================================================
# TARGET ORIENTATION — default gripper orientation for all moves
# =============================================================================
# Individual moves can override this. See the `moves` list in main().
DEFAULT_ORIENTATION = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)


# =============================================================================
# URDF CLEANUP — identical to original, builds a clean kinematic chain
# =============================================================================
def build_ik_chain():
    """
    Load the Stretch URDF, strip unnecessary links/joints (cameras, wheels,
    ArUco markers, head, etc.), add a virtual base translation joint, and
    return an ikpy Chain.

    This function is UNCHANGED from the original script — the IK math doesn't
    care whether we're using stretch_body or ROS 2 underneath.
    """
    pkg_path = str(importlib_resources.files('stretch_urdf'))
    urdf_file_path = pkg_path + '/SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf'

    original_urdf = urdfpy.URDF.load(urdf_file_path)
    modified_urdf = original_urdf.copy()

    # --- Remove links we don't need for arm IK ---
    names_of_links_to_remove = [
        'link_right_wheel', 'link_left_wheel', 'caster_link',
        'link_head', 'link_head_pan', 'link_head_tilt',
        'link_aruco_right_base', 'link_aruco_left_base', 'link_aruco_shoulder',
        'link_aruco_top_wrist', 'link_aruco_inner_wrist',
        'camera_bottom_screw_frame', 'camera_link', 'camera_depth_frame',
        'camera_depth_optical_frame', 'camera_infra1_frame',
        'camera_infra1_optical_frame', 'camera_infra2_frame',
        'camera_infra2_optical_frame', 'camera_color_frame',
        'camera_color_optical_frame', 'camera_accel_frame',
        'camera_accel_optical_frame', 'camera_gyro_frame',
        'camera_gyro_optical_frame',
        'gripper_camera_bottom_screw_frame', 'gripper_camera_link',
        'gripper_camera_depth_frame', 'gripper_camera_depth_optical_frame',
        'gripper_camera_infra1_frame', 'gripper_camera_infra1_optical_frame',
        'gripper_camera_infra2_frame', 'gripper_camera_infra2_optical_frame',
        'gripper_camera_color_frame', 'gripper_camera_color_optical_frame',
        'laser', 'base_imu', 'respeaker_base',
        'link_wrist_quick_connect',
        'link_gripper_finger_right', 'link_gripper_fingertip_right',
        'link_aruco_fingertip_right',
        'link_gripper_finger_left', 'link_gripper_fingertip_left',
        'link_aruco_fingertip_left',
        'link_aruco_d405', 'link_head_nav_cam',
    ]
    links_to_remove = [l for l in modified_urdf._links if l.name in names_of_links_to_remove]
    for lr in links_to_remove:
        modified_urdf._links.remove(lr)

    # --- Remove corresponding joints ---
    names_of_joints_to_remove = [
        'joint_right_wheel', 'joint_left_wheel', 'caster_joint',
        'joint_head', 'joint_head_pan', 'joint_head_tilt',
        'joint_aruco_right_base', 'joint_aruco_left_base', 'joint_aruco_shoulder',
        'joint_aruco_top_wrist', 'joint_aruco_inner_wrist',
        'camera_joint', 'camera_link_joint', 'camera_depth_joint',
        'camera_depth_optical_joint', 'camera_infra1_joint',
        'camera_infra1_optical_joint', 'camera_infra2_joint',
        'camera_infra2_optical_joint', 'camera_color_joint',
        'camera_color_optical_joint', 'camera_accel_joint',
        'camera_accel_optical_joint', 'camera_gyro_joint',
        'camera_gyro_optical_joint',
        'gripper_camera_joint', 'gripper_camera_link_joint',
        'gripper_camera_depth_joint', 'gripper_camera_depth_optical_joint',
        'gripper_camera_infra1_joint', 'gripper_camera_infra1_optical_joint',
        'gripper_camera_infra2_joint', 'gripper_camera_infra2_optical_joint',
        'gripper_camera_color_joint', 'gripper_camera_color_optical_joint',
        'joint_laser', 'joint_base_imu', 'joint_respeaker',
        'joint_wrist_quick_connect',
        'joint_gripper_finger_right', 'joint_gripper_fingertip_right',
        'joint_aruco_fingertip_right',
        'joint_gripper_finger_left', 'joint_gripper_fingertip_left',
        'joint_aruco_fingertip_left',
        'joint_aruco_d405', 'joint_head_nav_cam',
    ]
    joints_to_remove = [j for j in modified_urdf._joints if j.name in names_of_joints_to_remove]
    for jr in joints_to_remove:
        modified_urdf._joints.remove(jr)

    # --- Add virtual base joints: ROTATION first, then TRANSLATION ---
    # The kinematic chain from base becomes:
    #   base_link → [rotation about Z] → link_base_rotation
    #            → [translation along X] → link_base_translation
    #            → [mast (fixed)] → link_mast → ...
    #
    # WHY THIS ORDER MATTERS:
    #   The robot rotates in place first (rotation about its own Z axis),
    #   then drives forward/backward (translation along its new X axis).
    #   This matches how a diff-drive base actually moves — you turn to face
    #   the target, then drive toward it. If we did translate-then-rotate,
    #   the IK solver would be working in a frame that doesn't match reality.

    # 1. Base rotation: revolute joint about Z axis
    #    Limits: ±π/2 (±90°) prevents the IK solver from finding solutions
    #    where the robot rotates more than a quarter turn. This avoids
    #    unintuitive solutions like "rotate 137° and drive backwards."
    joint_base_rotation = urdfpy.Joint(
        name='joint_base_rotation',
        parent='base_link',
        child='link_base_rotation',
        joint_type='revolute',
        axis=np.array([0.0, 0.0, 1.0]),  # rotate about Z (yaw)
        origin=np.eye(4, dtype=np.float64),
        limit=urdfpy.JointLimit(effort=100.0, velocity=1.0,
                                lower=-np.pi/2, upper=np.pi/2),
    )
    modified_urdf._joints.append(joint_base_rotation)

    link_base_rotation = urdfpy.Link(
        name='link_base_rotation',
        inertial=None, visuals=None, collisions=None,
    )
    modified_urdf._links.append(link_base_rotation)

    # 2. Base translation: prismatic joint along X (after rotation)
    #    Limits: 0 to 1.0m — FORWARD ONLY. Setting lower=0 prevents the
    #    IK solver from driving the robot backwards. If you need backwards
    #    motion, change lower to a negative value (e.g., -0.5).
    joint_base_translation = urdfpy.Joint(
        name='joint_base_translation',
        parent='link_base_rotation',       # child of rotation link
        child='link_base_translation',
        joint_type='prismatic',
        axis=np.array([1.0, 0.0, 0.0]),   # translate along X
        origin=np.eye(4, dtype=np.float64),
        limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=0.0, upper=1.0),
    )
    modified_urdf._joints.append(joint_base_translation)

    link_base_translation = urdfpy.Link(
        name='link_base_translation',
        inertial=None, visuals=None, collisions=None,
    )
    modified_urdf._links.append(link_base_translation)

    # Re-parent the mast to hang off the virtual base translation link
    for j in modified_urdf._joints:
        if j.name == 'joint_mast':
            j.parent = 'link_base_translation'

    # Save and load as ikpy chain
    new_urdf_path = '/tmp/iktutorial/stretch.urdf'
    import os
    os.makedirs('/tmp/iktutorial', exist_ok=True)
    modified_urdf.save(new_urdf_path)

    # Build the chain WITH an active_links_mask so ikpy only solves for
    # joints that can actually move. Without this, ikpy assigns values to
    # fixed joints (like joint_mast, joint_arm_l4) which produces solutions
    # that look valid mathematically but can't be executed on the real robot.
    #
    # Index:  [0]     [1]    [2]    [3]    [4]   [5]    [6]   [7]   [8]   [9]   [10]  [11]   [12]  [13]  [14]   [15]
    # Joint:  base_lk rot    trans  mast   lift  arm_l4 l3    l2    l1    l0    yaw   yaw_bt pitch roll  grip   grasp
    # Type:   fixed   rev    prism  fixed  prism fixed  prism prism prism prism rev   fixed  rev   rev   fixed  fixed
    # Active: False   True   True   False  True  False  True  True  True  True  True  False  True  True  False  False
    active_links_mask = [
        False,  # [0]  Base link (fixed origin)
        True,   # [1]  joint_base_rotation (revolute)
        True,   # [2]  joint_base_translation (prismatic)
        False,  # [3]  joint_mast (fixed)
        True,   # [4]  joint_lift (prismatic)
        False,  # [5]  joint_arm_l4 (fixed)
        True,   # [6]  joint_arm_l3 (prismatic)
        True,   # [7]  joint_arm_l2 (prismatic)
        True,   # [8]  joint_arm_l1 (prismatic)
        True,   # [9]  joint_arm_l0 (prismatic)
        True,   # [10] joint_wrist_yaw (revolute)
        False,  # [11] joint_wrist_yaw_bottom (fixed)
        True,   # [12] joint_wrist_pitch (revolute)
        True,   # [13] joint_wrist_roll (revolute)
        False,  # [14] joint_gripper_s3_body (fixed)
        False,  # [15] joint_grasp_center (fixed)
    ]

    chain = ikpy.chain.Chain.from_urdf_file(
        new_urdf_path,
        active_links_mask=active_links_mask,
    )

    print("\n=== IK Chain Links ===")
    for i, link in enumerate(chain.links):
        active = "ACTIVE" if active_links_mask[i] else "fixed"
        print(f"  [{i:2d}] {link.name:40s} type={link.joint_type:10s} {active}")
    print()

    return chain


# =============================================================================
# ROS 2 NODE — this replaces all stretch_body calls
# =============================================================================
class StretchIKNode(HelloNode):
    """
    A ROS 2 node that:
      1. Subscribes to /stretch/joint_states (handled by HelloNode parent class)
      2. Runs ikpy inverse kinematics
      3. Commands the robot via move_to_pose() (handled by HelloNode parent class)

    HelloNode gives us for free:
      - self.joint_state  → latest JointState message (updated by subscriber)
      - self.move_to_pose(dict, blocking=True) → sends FollowJointTrajectory goals
      - self.trajectory_client → the action client (if you need lower-level control)
    """

    def __init__(self):
        HelloNode.__init__(self)
        # Initialize the ROS 2 node. HelloNode.main() sets up:
        #   - The /stretch/joint_states subscriber
        #   - The /stretch_controller/follow_joint_trajectory action client
        #   - TF2 buffer (for transforms, if needed later)
        HelloNode.main(self, 'stretch_ik_node', 'stretch_ik_node',
                        wait_for_first_pointcloud=False)

    def wait_for_joint_states(self):
        """Block until we've received at least one JointState message."""
        while not self.joint_state.position:
            self.get_logger().info('Waiting for /stretch/joint_states...')
            time.sleep(0.1)
        self.get_logger().info('Joint states received!')

    def get_joint_position(self, joint_name):
        """
        Look up a joint's current position from the latest JointState message.
        
        The JointState message has parallel arrays:
          joint_state.name     = ['joint_arm_l0', 'joint_lift', 'joint_wrist_yaw', ...]
          joint_state.position = [0.0,             0.6,          1.57,              ...]
        So we find the index of the name and return the position at that index.
        """
        js = self.joint_state
        if joint_name in js.name:
            return js.position[js.name.index(joint_name)]
        # Special case: wrist_extension may not appear directly in some firmware
        # versions. Compute it from the 4 arm segments if needed.
        if joint_name == 'wrist_extension':
            total = 0.0
            for seg in ['joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3']:
                if seg in js.name:
                    total += js.position[js.name.index(seg)]
            return total
        raise KeyError(f"Joint '{joint_name}' not found in joint_states: {js.name}")

    def get_current_configuration(self, chain):
        """
        Read current joint positions from ROS 2 and map them into the ikpy
        chain's configuration vector.

        WHAT CHANGED FROM ORIGINAL:
          Before: robot.lift.status['pos'], robot.arm.status['pos'], etc.
          Now:    self.get_joint_position('joint_lift'), etc.

        The ikpy chain has 16 elements (including fixed joints). The mapping is:
          [0]  base_link origin        → always 0.0 (fixed)
          [1]  joint_base_rotation     → 0.0 (virtual base rotation, starts at 0)
          [2]  joint_base_translation  → 0.0 (virtual base translation, starts at 0)
          [3]  joint_mast              → 0.0 (fixed)
          [4]  joint_lift              → lift position
          [5]  joint_arm_l4            → 0.0 (fixed)
          [6]  joint_arm_l3            → arm_extension / 4
          [7]  joint_arm_l2            → arm_extension / 4
          [8]  joint_arm_l1            → arm_extension / 4
          [9]  joint_arm_l0            → arm_extension / 4
          [10] joint_wrist_yaw         → wrist yaw
          [11] joint_wrist_yaw_bottom  → 0.0 (fixed)
          [12] joint_wrist_pitch       → wrist pitch
          [13] joint_wrist_roll        → wrist roll
          [14] joint_gripper_s3_body   → 0.0 (fixed)
          [15] joint_grasp_center      → 0.0 (fixed)
        """
        def bound_range(joint_name, value):
            """Clamp a value to the ikpy chain's joint limits."""
            names = [l.name for l in chain.links]
            index = names.index(joint_name)
            bounds = chain.links[index].bounds
            return min(max(value, bounds[0]), bounds[1])

        # Read from ROS 2 topic instead of stretch_body
        q_base_rot = 0.0   # virtual — always starts at 0
        q_base_trans = 0.0  # virtual — always starts at 0
        q_lift = bound_range('joint_lift', self.get_joint_position('joint_lift'))

        # Total arm extension, divided by 4 for the ikpy chain's 4 prismatic segments
        total_arm = self.get_joint_position('wrist_extension')
        q_arml = bound_range('joint_arm_l0', total_arm / 4.0)

        q_yaw = bound_range('joint_wrist_yaw', self.get_joint_position('joint_wrist_yaw'))
        q_pitch = bound_range('joint_wrist_pitch', self.get_joint_position('joint_wrist_pitch'))
        q_roll = bound_range('joint_wrist_roll', self.get_joint_position('joint_wrist_roll'))

        return [0.0, q_base_rot, q_base_trans, 0.0, q_lift, 0.0,
                q_arml, q_arml, q_arml, q_arml,
                q_yaw, 0.0, q_pitch, q_roll, 0.0, 0.0]

    def move_to_configuration(self, q):
        """
        Take an ikpy solution vector and command the robot via ROS 2.

        EXECUTION ORDER:
          1. Rotate the base first  (so the arm faces the target)
          2. Then translate + move all arm/wrist joints simultaneously

        WHY ROTATE FIRST?
          Stretch is a diff-drive robot. If you translate and rotate at the
          same time, the arm sweeps through an arc which might collide with
          things. Rotating first to face the target, then driving forward,
          is safer and matches how you'd intuitively position the robot.

        UPDATED INDICES (16-element vector with rotation added):
          [0]  fixed (base_link origin)
          [1]  joint_base_rotation     ← NEW: revolute about Z
          [2]  joint_base_translation  ← was index [1]
          [3]  fixed (joint_mast)
          [4]  joint_lift              ← was index [3]
          [5]  fixed (joint_arm_l4)
          [6-9]  arm segments          ← were indices [5-8]
          [10] joint_wrist_yaw         ← was index [9]
          [11] fixed (wrist_yaw_bottom)
          [12] joint_wrist_pitch       ← was index [11]
          [13] joint_wrist_roll        ← was index [12]
          [14-15] fixed (gripper)
        """
        q_base_rot = q[1]   # Base rotation (relative, radians)
        q_base_trans = q[2]  # Base translation (relative, meters)
        q_lift = q[4]        # Lift height (absolute, meters)
        q_arm = q[6] + q[7] + q[8] + q[9]  # Sum 4 segments → total extension
        q_yaw = q[10]        # Wrist yaw (radians)
        q_pitch = q[12]      # Wrist pitch (radians)
        q_roll = q[13]       # Wrist roll (radians)

        # --- Step 1: Rotate the base ---
        if abs(q_base_rot) > 0.001:  # skip if negligible rotation
            self.get_logger().info(f'Step 1: Rotating base by {q_base_rot:.3f} rad '
                                   f'({np.degrees(q_base_rot):.1f} deg)')
            self.move_to_pose({'rotate_mobile_base': q_base_rot}, blocking=True)
        else:
            self.get_logger().info('Step 1: No base rotation needed')

        # --- Step 2: Translate + move arm/wrist simultaneously ---
        self.get_logger().info(
            f'Step 2: Commanding translate={q_base_trans:.3f}m, lift={q_lift:.3f}m, '
            f'arm={q_arm:.3f}m, yaw={q_yaw:.2f}, pitch={q_pitch:.2f}, roll={q_roll:.2f}'
        )
        pose = {
            'translate_mobile_base': q_base_trans,  # relative base translation
            'joint_lift': q_lift,                    # absolute lift position
            'wrist_extension': q_arm,                # absolute arm extension
            'joint_wrist_yaw': q_yaw,                # absolute wrist yaw
            'joint_wrist_pitch': q_pitch,            # absolute wrist pitch
            'joint_wrist_roll': q_roll,              # absolute wrist roll
        }
        self.move_to_pose(pose, blocking=True)

    def move_to_grasp_goal(self, chain, target_point, target_orientation):
        """
        Run IK and move the robot. Returns the solution or None if IK failed.

        This is essentially unchanged from the original — the ikpy math is the same,
        we just swapped how we read current state and send commands.
        """
        q_init = self.get_current_configuration(chain)
        self.get_logger().info(f'Current config: {[f"{v:.3f}" for v in q_init]}')

        q_soln = chain.inverse_kinematics(
            target_point,
            target_orientation,
            orientation_mode='all',
            initial_position=q_init,
        )
        self.get_logger().info(f'IK solution: {[f"{v:.3f}" for v in q_soln]}')

        # Validate: check if the FK of the solution is close to the target
        fk_result = chain.forward_kinematics(q_soln)
        achieved_pos = fk_result[:3, 3]
        err = np.linalg.norm(achieved_pos - target_point)
        self.get_logger().info(f'Position error: {err:.4f} m')

        if not np.isclose(err, 0.0, atol=1e-2):
            self.get_logger().error(
                f'IK did not find a valid solution (error={err:.4f}m > 0.01m). '
                f'Target: {target_point}, Achieved: {achieved_pos.tolist()}'
            )
            return None

        self.move_to_configuration(q_soln)
        return q_soln

    def get_current_grasp_pose(self, chain):
        """Get the current end-effector pose via forward kinematics."""
        q = self.get_current_configuration(chain)
        return chain.forward_kinematics(q)


# =============================================================================
# MAIN
# =============================================================================
def main():
    # NOTE: We do NOT call rclpy.init() here because HelloNode.main()
    # (called inside StretchIKNode.__init__) already calls it internally.
    # Calling it twice causes: RuntimeError: Context.init() must only be called once

    # Build the IK chain (URDF cleanup — no ROS needed for this step)
    chain = build_ik_chain()

    # Create our ROS 2 node
    node = StretchIKNode()

    try:
        # Wait until we have joint state data from the robot
        node.wait_for_joint_states()

        # --- Stow the robot first so we start from a known configuration ---
        # This retracts the arm and lowers the lift, giving us a clean starting
        # point that's far from the target so you can see the robot move.
        # --- Safety lift: raise the arm before stowing to avoid collisions ---
        # If the lift is too low and the arm is extended, the stow command tries
        # to retract the arm which can hit the ground or objects. Raising the
        # lift first gives clearance.
        node.get_logger().info('Raising lift for safe stow...')
        node.move_to_pose({'joint_lift': 0.5}, blocking=True)

        node.get_logger().info('Stowing robot to start from known configuration...')
        from std_srvs.srv import Trigger
        future = node.stow_the_robot_service.call_async(Trigger.Request())
        # Wait for stow to complete
        time.sleep(6.0)
        node.get_logger().info('Stow complete. Starting IK...')

        # Print current end-effector pose (should be stowed position)
        current_pose = node.get_current_grasp_pose(chain)
        node.get_logger().info(f'Current EE pose (stowed):\n{current_pose}')

        # =================================================================
        # MULTI-TARGET SEQUENCE
        # =================================================================
        # Each move is defined as a relative offset [dx, dy, dz] from the
        # CURRENT end-effector position at the time of that move.
        #
        # The orientation can change per move too. We define each move as
        # a tuple: (name, [dx, dy, dz], rpy_orientation)
        #
        # Remember the coordinate frame:
        #   +X = forward, -X = backward
        #   +Y = left,    -Y = right     (arm extends to the left)
        #   +Z = up,      -Z = down
        #
        # And the joint limits constrain:
        #   rotation: ±90° per move
        #   translation: forward only (0 to 1m)

        moves = [
            # 
            ("Zigzag RIGHT",
             [0.5, 0, 0.2],
             ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)),

            ("Zigzag LEFT",
             [0.2, -0.5, 0.1],
             ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)),

            # 
            ("Reach UP",
             [-0.5, -0.3, 0.2],
             ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)),

            # 
            ("Reach FORWARD LOW",
             [0.4, -0.3, -0.1],
             ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)),

             ("Zigzag RIGHT",
             [0.5, -0.2, 0.2],
             ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)),

            ("Zigzag LEFT",
             [0.1, 0.5, 0.1],
             ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)),

            # 
            ("Reach UP",
             [-0.4, -0.3, 0.2],
             ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)),

            # 
            ("Reach FORWARD LOW",
             [0.4, -0.3, -0.3],
             ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)),
        ]

        for i, (name, offset, orientation) in enumerate(moves):
            node.get_logger().info(f'\n{"="*50}')
            node.get_logger().info(f'Move {i+1}/{len(moves)}: {name}')
            node.get_logger().info(f'Offset: {offset}')
            node.get_logger().info(f'{"="*50}')

            # Get current EE position (updates each move since robot moved)
            current_pose = node.get_current_grasp_pose(chain)
            current_pos = current_pose[:3, 3]
            target_point = (current_pos + np.array(offset)).tolist()

            node.get_logger().info(
                f'Current EE: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]'
            )
            node.get_logger().info(
                f'Target:     [{target_point[0]:.3f}, {target_point[1]:.3f}, {target_point[2]:.3f}]'
            )

            q_soln = node.move_to_grasp_goal(chain, target_point, orientation)

            if q_soln is not None:
                time.sleep(1.0)  # Let robot settle before next move
                final_pose = node.get_current_grasp_pose(chain)
                final_pos = final_pose[:3, 3]
                node.get_logger().info(
                    f'Move {i+1} complete. EE at: '
                    f'[{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]'
                )
            else:
                node.get_logger().warn(
                    f'Move {i+1} ({name}) failed IK — skipping to next move'
                )
                continue

        node.get_logger().info('\nAll moves complete!')

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # HelloNode may have already shut down the context


if __name__ == '__main__':
    main()