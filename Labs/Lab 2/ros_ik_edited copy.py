#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
import urchin as urdfpy
import numpy as np
import ikpy.chain
import importlib.resources as importlib_resources
import threading
from rclpy.duration import Duration

class StretchIKNode(Node):
    def __init__(self):
        super().__init__('stretch_ik_node')
        self._action_client = ActionClient(self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory')
        self.joint_states = None
        self.joint_state_lock = threading.Lock()
        self.create_subscription(JointState, '/joint_states', lambda msg: setattr(self, 'joint_states', msg), 10)
        
        while self.joint_states is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        self.setup_ik_chain()
    
    def get_joint_position(self, joint_name):
        with self.joint_state_lock:
            try:
                idx = self.joint_states.name.index(joint_name)
                return self.joint_states.position[idx]
            except:
                return 0.0
    
    def setup_ik_chain(self):
        pkg_path = str(importlib_resources.files('stretch_urdf'))
        urdf = urdfpy.URDF.load(f'{pkg_path}/SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf').copy()
        
        # Remove non-essential links and joints
        remove_prefixes = ['link_right_wheel', 'link_left_wheel', 'caster_', 'link_head', 'link_aruco', 'camera_', 'gripper_camera_', 'laser', 'base_imu', 'respeaker', 'link_wrist_quick_connect', 'link_gripper_finger']
        urdf._links = [l for l in urdf._links if not any(l.name.startswith(p) for p in remove_prefixes)]
        urdf._joints = [j for j in urdf._joints if not any(j.name.startswith(p.replace('link_', 'joint_')) for p in remove_prefixes)]
        
        # Add virtual base joints
        for name, axis in [('translation', [1, 0, 0]), ('rotation', [0, 0, 1])]:
            urdf._joints.append(urdfpy.Joint(f'joint_base_{name}', 'base_link', f'link_base_{name}', 'prismatic',
                                            np.array(axis, dtype=float), np.eye(4), urdfpy.JointLimit(100, 1, -1, 1)))
            urdf._links.append(urdfpy.Link(f'link_base_{name}', None, None, None))
        
        for j in urdf._joints:
            if j.name == 'joint_mast':
                j.parent = 'link_base_rotation'
        
        urdf.save("/tmp/iktutorial/stretch.urdf")
        self.chain = ikpy.chain.Chain.from_urdf_file("/tmp/iktutorial/stretch.urdf")
    
    def get_current_configuration(self):
        def bound(name, value):
            idx = [l.name for l in self.chain.links].index(name)
            b = self.chain.links[idx].bounds
            return np.clip(value, b[0], b[1])
        
        q_arml = bound('joint_arm_l0', self.get_joint_position('joint_arm_l0') / 4.0)
        return [0, 0, 0, bound('joint_lift', self.get_joint_position('joint_lift')), 0, 
                q_arml, q_arml, q_arml, q_arml,
                bound('joint_wrist_yaw', self.get_joint_position('joint_wrist_yaw')), 0,
                bound('joint_wrist_pitch', self.get_joint_position('joint_wrist_pitch')),
                bound('joint_wrist_roll', self.get_joint_position('joint_wrist_roll')), 0, 0]
    
    def move_to_configuration(self, q):
        joint_names = ['joint_lift', 'joint_arm_l0', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll']
        joint_positions = [q[3], sum(q[5:9]), q[9], q[11], q[12]]
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        point = JointTrajectoryPoint(positions=joint_positions, time_from_start=Duration(seconds=5.0).to_msg())
        goal.trajectory.points = [point]
        
        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result().accepted:
            result_future = future.result().get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            return True
        return False
    
    def move_to_grasp_goal(self, target_point, target_orientation):
        q_soln = self.chain.inverse_kinematics(target_point, target_orientation, 
                                               orientation_mode='all', initial_position=self.get_current_configuration())
        
        err = np.linalg.norm(self.chain.forward_kinematics(q_soln)[:3, 3] - target_point)
        if not np.isclose(err, 0.0, atol=1e-2):
            self.get_logger().error(f"IK failed. Error: {err}")
            return None
        
        self.move_to_configuration(q_soln)
        return q_soln
    
    def get_current_grasp_pose(self):
        return self.chain.forward_kinematics(self.get_current_configuration())


def main(args=None):
    rclpy.init(args=args)
    node = StretchIKNode()
    
    target_point = [0.5, -0.441, 0.3]
    target_orientation = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi/2)
    
    q_soln = node.move_to_grasp_goal(target_point, target_orientation)
    if q_soln is not None:
        node.get_logger().info(f'Current pose:\n{node.get_current_grasp_pose()}')
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
