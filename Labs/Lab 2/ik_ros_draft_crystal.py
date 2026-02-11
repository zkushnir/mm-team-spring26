import ikpy.urdf.utils
import urchin as urdfpy
import numpy as np
import ikpy.chain
import stretch_body.robot
import importlib.resources as importlib_resources

import rclpy
from rclpy.node import Node

import hello_helpers.hello_misc as hm
#temp = hm.HelloNode.quick_create('temp')

# NOTE before running: `python3 -m pip install --upgrade ikpy graphviz urchin networkx`

target_point = [-0.043, -0.441, 0.654]
target_orientation = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi/2) # [roll, pitch, yaw]

class IKNode(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)
        
    def main(self):
        hm.HelloNode.main(self, 'ik_ros', 'ik_ros', wait_for_first_pointcloud=False)


        pkg_path = str(importlib_resources.files('stretch_urdf'))
        urdf_file_path = pkg_path + '/SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf'

        # Remove unnecessary links/joints
        ### Get rid of joints that don't move, use RViz for reference
        original_urdf = urdfpy.URDF.load(urdf_file_path)
        modified_urdf = original_urdf.copy()

        names_of_links_to_remove = ['link_right_wheel', 'link_left_wheel', 'caster_link', 'link_head', 'link_head_pan', 'link_head_tilt', 'link_aruco_right_base', 'link_aruco_left_base', 'link_aruco_shoulder', 'link_aruco_top_wrist', 'link_aruco_inner_wrist', 'camera_bottom_screw_frame', 'camera_link', 'camera_depth_frame', 'camera_depth_optical_frame', 'camera_infra1_frame', 'camera_infra1_optical_frame', 'camera_infra2_frame', 'camera_infra2_optical_frame', 'camera_color_frame', 'camera_color_optical_frame', 'camera_accel_frame', 'camera_accel_optical_frame', 'camera_gyro_frame', 'camera_gyro_optical_frame', 'gripper_camera_bottom_screw_frame', 'gripper_camera_link', 'gripper_camera_depth_frame', 'gripper_camera_depth_optical_frame', 'gripper_camera_infra1_frame', 'gripper_camera_infra1_optical_frame', 'gripper_camera_infra2_frame', 'gripper_camera_infra2_optical_frame', 'gripper_camera_color_frame', 'gripper_camera_color_optical_frame', 'laser', 'base_imu', 'respeaker_base', 'link_wrist_quick_connect', 'link_gripper_finger_right', 'link_gripper_fingertip_right', 'link_aruco_fingertip_right', 'link_gripper_finger_left', 'link_gripper_fingertip_left', 'link_aruco_fingertip_left', 'link_aruco_d405', 'link_head_nav_cam']
        # links_kept = ['base_link', 'link_mast', 'link_lift', 'link_arm_l4', 'link_arm_l3', 'link_arm_l2', 'link_arm_l1', 'link_arm_l0', 'link_wrist_yaw', 'link_wrist_yaw_bottom', 'link_wrist_pitch', 'link_wrist_roll', 'link_gripper_s3_body', 'link_grasp_center']
        links_to_remove = [l for l in modified_urdf._links if l.name in names_of_links_to_remove]
        for lr in links_to_remove:
            modified_urdf._links.remove(lr)
        names_of_joints_to_remove = ['joint_right_wheel', 'joint_left_wheel', 'caster_joint', 'joint_head', 'joint_head_pan', 'joint_head_tilt', 'joint_aruco_right_base', 'joint_aruco_left_base', 'joint_aruco_shoulder', 'joint_aruco_top_wrist', 'joint_aruco_inner_wrist', 'camera_joint', 'camera_link_joint', 'camera_depth_joint', 'camera_depth_optical_joint', 'camera_infra1_joint', 'camera_infra1_optical_joint', 'camera_infra2_joint', 'camera_infra2_optical_joint', 'camera_color_joint', 'camera_color_optical_joint', 'camera_accel_joint', 'camera_accel_optical_joint', 'camera_gyro_joint', 'camera_gyro_optical_joint', 'gripper_camera_joint', 'gripper_camera_link_joint', 'gripper_camera_depth_joint', 'gripper_camera_depth_optical_joint', 'gripper_camera_infra1_joint', 'gripper_camera_infra1_optical_joint', 'gripper_camera_infra2_joint', 'gripper_camera_infra2_optical_joint', 'gripper_camera_color_joint', 'gripper_camera_color_optical_joint', 'joint_laser', 'joint_base_imu', 'joint_respeaker', 'joint_wrist_quick_connect', 'joint_gripper_finger_right', 'joint_gripper_fingertip_right', 'joint_aruco_fingertip_right', 'joint_gripper_finger_left', 'joint_gripper_fingertip_left', 'joint_aruco_fingertip_left', 'joint_aruco_d405', 'joint_head_nav_cam'] 
        # joints_kept = ['joint_mast', 'joint_lift', 'joint_arm_l4', 'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 'joint_wrist_yaw', 'joint_wrist_yaw_bottom', 'joint_wrist_pitch', 'joint_wrist_roll', 'joint_gripper_s3_body', 'joint_grasp_center']
        joints_to_remove = [l for l in modified_urdf._joints if l.name in names_of_joints_to_remove]
        for jr in joints_to_remove:
            modified_urdf._joints.remove(jr)

        # Add virtual base joint
        # JOINT YOU ADD IS CLOSE TO Q_BASE
        # somewhere between mast and base??
        joint_base_rotation = urdfpy.Joint(name='joint_base_rotation',
                                            parent='base_link',
                                            child='link_base_rotation',
                                            joint_type='revolute', # Modified
                                            axis=np.array([0.0, 0.0, 1.0]), # Modified
                                            origin=np.eye(4, dtype=np.float64),
                                            limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=-1.0, upper=1.0))
        modified_urdf._joints.append(joint_base_rotation)
        link_base_rotation = urdfpy.Link(name='link_base_rotation',
                                            inertial=None,
                                            visuals=None,
                                            collisions=None)
        modified_urdf._links.append(link_base_rotation)

        joint_base_translation = urdfpy.Joint(name='joint_base_translation',
                                            parent='link_base_rotation', # MODIFIED!!
                                            child='link_base_translation',
                                            joint_type='prismatic',
                                            axis=np.array([1.0, 0.0, 0.0]),
                                            origin=np.eye(4, dtype=np.float64),
                                            limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=-1.0, upper=1.0))
        modified_urdf._joints.append(joint_base_translation)
        link_base_translation = urdfpy.Link(name='link_base_translation',
                                            inertial=None,
                                            visuals=None,
                                            collisions=None)
        modified_urdf._links.append(link_base_translation)

        # amend the chain
        #MAST LINK IS CURRENTLY CONNECTED to BASE LINK BUT IS NOT SUPPOSED TO BE
        # base_link -> base_rotation -> base_translation -> mast_link
        for j in modified_urdf._joints:
            if j.name == 'joint_mast':
                j.parent = 'link_base_translation' #was link_base_rotation

        new_urdf_path = "/tmp/iktutorial/stretch.urdf"
        modified_urdf.save(new_urdf_path)

        self.chain = ikpy.chain.Chain.from_urdf_file(new_urdf_path)
        
        # GET RID OF THIS API STUFF AND SWITCH TO ROS
        # want to add translation and rotation in in between base and mast link
        # ROS 2 uses only move_to_pose ?
        print("\n=== IK Chain Structure ===")
        for i, link in enumerate(self.chain.links):
            print(f"Index {i}: {link.name} (Type: {link.joint_type})")
        print("=========================\n")

        self.move_to_grasp_goal(target_point, target_orientation)

        # random motions - refine better for submission
        # get current position of end effector with Lecture 2 code with 1 liner
        # use that as target
        self.move_to_grasp_goal([-0.043, -0.88, 0.654],
                                ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi/2))

        self.stow_the_robot()
        rclpy.shutdown()

    def move_to_grasp_goal(self, target_point, target_orientation):

        q_soln = self.chain.inverse_kinematics(
            target_point,
            target_orientation,
            orientation_mode='all'
        )

        base_rot = q_soln[1]
        base_trans = q_soln[2]

        pose_cmd = {
            'joint_lift': q_soln[4],
            'joint_arm': q_soln[6] + q_soln[7] + q_soln[8] + q_soln[9],
            'joint_wrist_yaw': q_soln[10],
            'joint_wrist_pitch': q_soln[12],
            'joint_wrist_roll': q_soln[13]
        }

        # Only send ONE mobile base command
        if abs(base_trans) > 1e-4:
            pose_cmd['translate_mobile_base'] = base_trans
        elif abs(base_rot) > 1e-4:
            pose_cmd['rotate_mobile_base'] = base_rot

        self.move_to_pose(pose_cmd, blocking=True)


node = IKNode()
node.main()