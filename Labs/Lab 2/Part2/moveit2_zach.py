import rclpy, time
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from moveit.core.robot_state import RobotState
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import FollowJointTrajectory
from hello_helpers.hello_misc import HelloNode
import moveit2_utils

# Make sure to run:
# ros2 launch stretch_core stretch_driver.launch.py

class MoveMe(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)
        self.main('move_me', 'move_me', wait_for_first_pointcloud=False)

        self._static_tf = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.rotation.w = 1.0
        self._static_tf.sendTransform(t)
        time.sleep(0.5)  # Let TF propagate before MoveIt starts up

        self.stow_the_robot()

        planning_group = 'mobile_base_arm'
        moveit, moveit_plan, planning_params = moveit2_utils.setup_moveit(planning_group)

        # Snapshot stow positions.  Clamp arm segments away from the 0.0 minimum
        stow_lift = self.get_joint_pos('joint_lift')
        stow_arm  = [max(1e-3, self.get_joint_pos(f'joint_arm_l{i}')) for i in range(3, -1, -1)]
        stow_wrist = [
            self.get_joint_pos('joint_wrist_yaw'),
            self.get_joint_pos('joint_wrist_pitch'),
            self.get_joint_pos('joint_wrist_roll'),
        ]

        arm_each  = 0.4 / 4.0          # 0.1 m per segment when arm is fully extended
        wrist_rot = np.deg2rad(15.0)

        # Absolute waypoints.  Base values are (x, y, theta) relative to the
        # robot's starting position.
        # +x = out the front (flat side), +y = robot's left, theta = yaw (rad).
        abs_poses = [
            dict(name='pose0', x=0.0,   y=0.0,   theta=0.0,
                 lift=stow_lift, arm=stow_arm, wrist=stow_wrist),

            dict(name='pose1', x=0.2, y=0.2, theta=0.0,
                 lift=0.50,      arm=stow_arm, wrist=stow_wrist),

            # From pose1 
            dict(name='pose2', x=0.2,  y=0.0,  theta=-np.pi/2,
                 lift=0.50,      arm=[arm_each]*4, wrist=stow_wrist),

            # From pose2 
            dict(name='pose3', x=0.2,  y=-0.2,  theta=np.pi/2,
                 lift=0.50,      arm=[arm_each]*4,
                 wrist=[stow_wrist[0]+wrist_rot,
                        stow_wrist[1]+wrist_rot,
                        stow_wrist[2]+wrist_rot]),

            # From pose3 
            dict(name='pose4', x=.2, y=-.4,  theta=2*np.pi,
                 lift=stow_lift, arm=stow_arm, wrist=stow_wrist),
        ]

        # Track the robot's absolute position 
        prev_x, prev_y, prev_theta = 0.0, 0.0, 0.0

        for seg_idx in range(len(abs_poses) - 1):
            goal = abs_poses[seg_idx + 1]
            self.get_logger().info(
                f"--- Segment {seg_idx}: {abs_poses[seg_idx]['name']} -> {goal['name']} ---")

        
            dx     = goal['x']     - prev_x
            dy     = goal['y']     - prev_y
            dtheta = (goal['theta'] - prev_theta + np.pi) % (2 * np.pi) - np.pi

            goal_state = RobotState(moveit.get_robot_model())
            goal_state.set_joint_group_positions(planning_group, [
                dx, dy, dtheta,
                goal['lift'],
                goal['arm'][0], goal['arm'][1], goal['arm'][2], goal['arm'][3],
                goal['wrist'][0], goal['wrist'][1], goal['wrist'][2],
            ])

           
            moveit_plan.set_start_state_to_current_state()
            moveit_plan.set_goal_state(robot_state=goal_state)

            plan = moveit_plan.plan(parameters=planning_params)
            if plan is None or getattr(plan, 'trajectory', None) is None:
                self.get_logger().error('Planning failed. Aborting.')
                break

            self.execute_plan(plan)

            # Advance the tracked absolute position 
            prev_x, prev_y, prev_theta = goal['x'], goal['y'], goal['theta']
            time.sleep(0.5)

    def execute_plan(self, plan):
        # NOTE: You don't need to edit this function
        processor = moveit2_utils.TrajectoryProcessor()
        segments = processor.process_trajectory(plan, self.joint_state)

        for i, goal_traj in enumerate(segments):
            self.get_logger().info(
                f"Executing segment {i+1}/{len(segments)} (Mode: {self._detect_mode(goal_traj)})"
            )

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = goal_traj

            future = self.trajectory_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.get_logger().error(f"Segment {i+1} rejected!")
                break

            res_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, res_future)
            res = res_future.result()

            if res.result.error_code != res.result.SUCCESSFUL:
                self.get_logger().error(f"Segment {i+1} failed! Code: {res.result.error_code}")
                break

    def get_joint_pos(self, joint_name):
        return self.joint_state.position[self.joint_state.name.index(joint_name)]

    def _detect_mode(self, traj):
        if 'translate_mobile_base' in traj.joint_names: return 'TRANSLATE'
        if 'rotate_mobile_base' in traj.joint_names: return 'ROTATE'
        return 'ARM_ONLY'

if __name__ == '__main__':
    MoveMe()
