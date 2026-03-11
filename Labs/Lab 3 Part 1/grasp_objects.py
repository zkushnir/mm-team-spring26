import rclpy
import rclpy.duration
import rclpy.time
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import FollowJointTrajectory
from hello_helpers.hello_misc import HelloNode
import threading
import tf2_ros
from tf2_geometry_msgs import TransformStamped
import tf2_geometry_msgs
from sensor_msgs.msg import JointState
import ik_ros_utils as ik
import ikpy
import ikpy.utils.geometry

# Make sure to run:
#   ros2 launch stretch_core stretch_driver.launch.py

class IKTargetFollowing(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        self.delta = 0.03 # meters (3 cm max step toward goal per timestep)
        self.target_frame = 'base_link'
        self.gripper_frame = 'link_grasp_center'
        self.tf_buffer = None
        self.tf_listener = None
        self.joint_states_lock = threading.Lock()

        # Guard: joint_state may not be populated yet when first goal arrives
        self.joint_state = None
    
    def joint_states_callback(self, msg):
        # unpacks joint state messages for what works with/is expected by ikpy
        with self.joint_states_lock:
            joint_states = msg
        # extract information needed for ik_solver
        joint_names = [
            'joint_lift', 'joint_arm_l0', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll'
        ]
        self.joint_state = {}
        for joint_name in joint_names:
            i = joint_states.name.index(joint_name)
            self.joint_state[joint_name] = joint_states.position[i]

    def get_goal_pose_in_base_frame(self, goal_msg):
        # TODO: ------------- start --------------
        # fill with your response
        #   transform the goal pose to the base frame

        # Addressed "Lookup would require extrapolation into the past" error by zeroing
        # the timestamp so TF uses latest available transform
        goal_msg.header.stamp = rclpy.time.Time().to_msg()
        goal_transformed = self.tf_buffer.transform(goal_msg, self.target_frame, timeout=rclpy.duration.Duration(seconds=0.1))

        # TODO: -------------- end ---------------

        return goal_transformed

    def get_gripper_pose_in_base_frame(self):
        # TODO: ------------- start --------------
        # fill with your response
        #   transform the gripper pose to the base frame

        gripper_transformed = self.tf_buffer.lookup_transform(self.target_frame,self.gripper_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))

        # TODO: -------------- end ---------------

        return gripper_transformed

    def goal_callback(self, goal_msg):
        # self.get_logger().info(f'Received goal pose: {goal_msg.pose}')

        # skip if joint states haven't been received yet
        if self.joint_state is None:
            print("Waiting for joint states")
            return

        try:
            goal_transformed    = self.get_goal_pose_in_base_frame(goal_msg)
            gripper_transformed = self.get_gripper_pose_in_base_frame()
            goal_pos    = ik.get_xyz_from_msg(goal_transformed)
            gripper_pos = ik.get_xyz_from_msg(gripper_transformed)
        except Exception as e:
            print(f"Error getting transforms: {e}")
            return

        waypoint_pos, waypoint_orient = self.compute_waypoint_to_goal(goal_pos, gripper_pos)

        # TODO: ------------- start --------------
        # fill with your response
        #   use the same functions you used for IK in Lab 2, now in `ik_ros_utils.py`, 
        #   to move the robot to the transformed goal point.

        # Part 2: lazy-init grasp state machine variables
        if not hasattr(self, '_grasp_state'):
            self._grasp_state = 'approaching'
            self._goal_reached_count = 0

        q_soln = None
        distance_to_goal = np.linalg.norm(goal_pos - gripper_pos)
        print(f"[{self._grasp_state}] Distance to goal: {distance_to_goal*100:.1f} cm")

        # Skip if already grasped and retracted
        if self._grasp_state == 'done':
            return

        if self._grasp_state == 'approaching':
            # Check if gripper is close enough to grasp
            if distance_to_goal < 0.04:  # 4 cm threshold
                self._goal_reached_count += 1
                print(f"Close to goal ({self._goal_reached_count}/3)")
                if self._goal_reached_count >= 3:
                    # --- GRASP ---
                    self._grasp_state = 'grasping'
                    print("========== GRASPING ===========")
                    self.move_to_pose({'gripper_aperture': -0.2}, blocking=True)
                    import time
                    time.sleep(1.0)
                    # --- RETRACT ---
                    self._grasp_state = 'retracting'
                    print("========== RETRACTING ===========")
                    self.move_to_pose({'joint_lift': 0.9}, blocking=True)
                    self.move_to_pose({'wrist_extension': 0.0}, blocking=True)
                    self._grasp_state = 'done'
                    print("========== GRASP COMPLETE ===========")
                    return
            else:
                self._goal_reached_count = 0  # reset if we drifted away

            # Normal IK approach toward the waypoint
            q_init = ik.get_current_configuration(self.joint_state)

            if q_init is None:
                print("Could not get current configuration, skipping IK solve")
                return

            q_soln = ik.chain.inverse_kinematics(
                waypoint_pos,
                initial_position=q_init
            )

            err = np.linalg.norm(ik.chain.forward_kinematics(q_soln)[:3, 3] - waypoint_pos)
            if not np.isclose(err, 0.0, atol=1e-2):
                print(f"IK position error too large ({err*100:.1f} cm), skipping")
                q_soln = None

        # TODO: -------------- end ---------------

        # NOTE: if you find that the robot's base is moving too much, its likely that the ik solver is
        # struggling to find solutions without the base doing most of the work to achieve the waypoint pose.
        # you can adjust the `self.delta` variable to be smaller so that the displacements are smaller, and
        #   there is a valid solution without excessive base movement
        # you can also set your own triggers manually (keep delta large but use an if/else on move_to_pose()
        #   so the base only moves above a certain distance threshold
        # you can also try adjusting joint limits of the base trans/rot in `ik_ros_utils.py` to be much smaller
        # one or some combination of these should help!

        ik.print_q(q_soln)
        if q_soln is not None:
            ik.move_to_configuration(self, q_soln)

    def compute_waypoint_to_goal(self, goal_pos, gripper_pos):

        # TODO: ------------- start --------------
        # fill with your response
        #   find the distance between the published goal position and the gripper position
        #   if its above some threshold (delta), consider the goal to be too far (since we're trying to track the object
        #   at least 2Hz) to reach before the next goal is published
        #   in this case, find a waypoint toward the goal position that is delta away from the gripper position (make some progress towards the goal)
        #   otherwise, the goal is close and we can move there directly

        difference = goal_pos - gripper_pos
        distance = np.linalg.norm(difference)

        if distance > self.delta:
            # waypoint to goal position 
            direction    = difference / distance
            waypoint_pos = gripper_pos + self.delta * direction
        else:
            # close, go directly to the goal
            waypoint_pos = goal_pos

        # TODO: -------------- end ---------------

        # use an zero rotation for the waypoint (its a point so we don't need to worry about orientation)
        waypoint_orient = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, 0.0) # [roll, pitch, yaw]

        return waypoint_pos, waypoint_orient


    def move_to_ready_pose(self):
        # TODO: minor - uncomment the correct ready pose for part 1 or 2!
        #   part 1: 
        # self.move_to_pose(ik.READY_POSE_P1, blocking=True)
        #   part 2: READY_POSE_P2
        self.move_to_pose(ik.READY_POSE_P2, blocking=True)

    def main(self):
        HelloNode.main(self, 'follow_target', 'follow_target', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()
        self.joint_states_subscriber = self.create_subscription(
            JointState,
            '/stretch/joint_states',
            callback=self.joint_states_callback,
            qos_profile=1
        )

        self.stow_the_robot()
        self.move_to_ready_pose()
        print("At Ready Pose")

        # TODO: ------------- start --------------
        # fill with your response
        #   create a tf2 buffer and listener
        #   create a subscriber to the goal pose published by your object detector

        # buffer & listener for transforms
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to the goal pose published by object_detector.py
        self.goal_subscriber = self.create_subscription(PoseStamped, '/object_detector/goal_pose', self.goal_callback, qos_profile=10, callback_group=self.callback_group)

        # TODO: -------------- end ---------------


if __name__ == '__main__':
    target_follower = IKTargetFollowing()
    target_follower.main()
    target_follower.new_thread.join()