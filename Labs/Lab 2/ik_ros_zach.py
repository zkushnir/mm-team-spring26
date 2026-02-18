import numpy as np
import ikpy.chain
import ikpy.utils.geometry
import urchin as urdfpy
import importlib.resources as importlib_resources
import time
import rclpy
from hello_helpers.hello_misc import HelloNode


TARGET_OFFSET = [0.5, 0 , 0.06]  
TARGET_ORIENTATION = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)


def build_ik_chain():
   
    pkg_path = str(importlib_resources.files('stretch_urdf'))
    urdf_file_path = pkg_path + '/SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf'

    original_urdf = urdfpy.URDF.load(urdf_file_path)
    modified_urdf = original_urdf.copy()

    # Remove links we don't need for arm IK
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

    # Remove corresponding joints
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

    #add rotation
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


# ROS 2 NODE 
class StretchIKNode(HelloNode):
    
    def __init__(self):
        HelloNode.__init__(self)
        HelloNode.main(self, 'stretch_ik_node', 'stretch_ik_node',
                        wait_for_first_pointcloud=False)

    def wait_for_joint_states(self):
        while not self.joint_state.position:
            self.get_logger().info('Waiting for /stretch/joint_states...')
            time.sleep(0.1)
        self.get_logger().info('Joint states received!')

    def get_joint_position(self, joint_name):
        
        js = self.joint_state
        if joint_name in js.name:
            return js.position[js.name.index(joint_name)]
        if joint_name == 'wrist_extension':
            total = 0.0
            for seg in ['joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3']:
                if seg in js.name:
                    total += js.position[js.name.index(seg)]
            return total
        raise KeyError(f"Joint '{joint_name}' not found in joint_states: {js.name}")

    def get_current_configuration(self, chain):
        
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
        
        q_base_rot = q[1]   # Base rotation (relative, radians)
        q_base_trans = q[2]  # Base translation (relative, meters)
        q_lift = q[4]        # Lift height (absolute, meters)
        q_arm = q[6] + q[7] + q[8] + q[9]  # Sum 4 segments → total extension
        q_yaw = q[10]        # Wrist yaw (radians)
        q_pitch = q[12]      # Wrist pitch (radians)
        q_roll = q[13]       # Wrist roll (radians)

        # Step 1: Rotate the base 
        if abs(q_base_rot) > 0.001:  # skip if negligible rotation
            self.get_logger().info(f'Step 1: Rotating base by {q_base_rot:.3f} rad '
                                   f'({np.degrees(q_base_rot):.1f} deg)')
            self.move_to_pose({'rotate_mobile_base': q_base_rot}, blocking=True)
        else:
            self.get_logger().info('Step 1: No base rotation needed')

        # Step 2: Translate + move arm/wrist simultaneously
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


def main():
    
    chain = build_ik_chain()

    node = StretchIKNode()

    try:
        node.wait_for_joint_states()

    
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

        # Compute absolute target from relative offset 
        current_pos = current_pose[:3, 3]
        target_point = (current_pos + np.array(TARGET_OFFSET)).tolist()

        node.get_logger().info(
            f'Current EE position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]'
        )
        node.get_logger().info(
            f'Offset: {TARGET_OFFSET}'
        )
        node.get_logger().info(
            f'Absolute target: [{target_point[0]:.3f}, {target_point[1]:.3f}, {target_point[2]:.3f}]'
        )

        # Run IK and move to the target
        node.get_logger().info(
            f'Moving to target: pos={[f"{v:.3f}" for v in target_point]}, '
            f'orientation=rpy(0, 0, {-np.pi/2:.2f})'
        )
        q_soln = node.move_to_grasp_goal(chain, target_point, TARGET_ORIENTATION)

        if q_soln is not None:
            # Print final pose after moving
            time.sleep(1.0)  # Let the robot settle
            final_pose = node.get_current_grasp_pose(chain)
            node.get_logger().info(f'Final EE pose:\n{final_pose}')

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  


if __name__ == '__main__':
    main()