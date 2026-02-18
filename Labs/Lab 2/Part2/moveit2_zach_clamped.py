import rclpy, time
import numpy as np
from moveit.core.robot_state import RobotState
from control_msgs.action import FollowJointTrajectory
from hello_helpers.hello_misc import HelloNode
import moveit2_utils

# Make sure to run:
#   ros2 launch stretch_core stretch_driver.launch.py
#
# This script plans with MoveIt (mobile base + arm) and then executes the resulting
# trajectory by sending FollowJointTrajectory goals to the Stretch controller.

class MoveMe(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)
        self.main('move_me', 'move_me', wait_for_first_pointcloud=False)

        # Put the robot into a known configuration first
        self.stow_the_robot()

        planning_group = 'mobile_base_arm'
        moveit, moveit_plan, planning_params = moveit2_utils.setup_moveit(planning_group)

        # ---------------------------------------------------------------------
        # How the planning_group joint vector is ordered (per the assignment):
        #   [x, y, theta, lift, arm_l3, arm_l2, arm_l1, arm_l0, wrist_yaw, wrist_pitch, wrist_roll]
        #
        # Base frame convention in this assignment:
        #   +x : out the front of the robot (the flat side of the base)
        #   +y : to the robot's left (opposite the direction the arm faces)
        #   theta : yaw rotation about +z (radians)
        # ---------------------------------------------------------------------

        # Snapshot the "stow" joint positions so we can return to them at pose 4.
        stow = {
            "lift": _nz(self.get_joint_pos("joint_lift")),
            "arm_l3": _nz(self.get_joint_pos("joint_arm_l3")),
            "arm_l2": _nz(self.get_joint_pos("joint_arm_l2")),
            "arm_l1": _nz(self.get_joint_pos("joint_arm_l1")),
            "arm_l0": _nz(self.get_joint_pos("joint_arm_l0")),
            "wrist_yaw": self.get_joint_pos("joint_wrist_yaw"),
            "wrist_pitch": self.get_joint_pos("joint_wrist_pitch"),
            "wrist_roll": self.get_joint_pos("joint_wrist_roll"),
        }


        # Clamp prismatic joints to be within bounds.
        # On some Stretch configs, joint_states can report slightly negative values for prismatic joints,
        # which MoveIt treats as out-of-bounds and OMPL refuses to plan ("invalid start state").
        def _nz(v: float) -> float:
            try:
                v = float(v)
            except Exception:
                return 0.0
            return v if v > 0.0 else 0.0

        arm_extend = 0.4  # meters total extension target at pose 2
        arm_each = arm_extend / 4.0
        wrist_rot = np.deg2rad(45.0)

        # Base waypoints from the diagram (meters / radians), relative to the start (pose 0).
        # NOTE: y is "robot-left" (downward in the diagram), so "up" in the diagram is negative y.
        poses = [
            # pose 0: start (also pose 4)
            dict(name="pose0", x=0.0,  y=0.0,  theta=0.0,
                 lift=stow["lift"],
                 arm=[stow["arm_l3"], stow["arm_l2"], stow["arm_l1"], stow["arm_l0"]],
                 wrist=[stow["wrist_yaw"], stow["wrist_pitch"], stow["wrist_roll"]]),

            # pose 1: base to (-20cm, -20cm) and lift to 0.5m
            dict(name="pose1", x=-0.20, y=-0.20, theta=0.0,
                 lift=0.50,
                 arm=[stow["arm_l3"], stow["arm_l2"], stow["arm_l1"], stow["arm_l0"]],
                 wrist=[stow["wrist_yaw"], stow["wrist_pitch"], stow["wrist_roll"]]),

            # pose 2: base to (+40cm, -20cm), rotate to +90deg, extend arm to 0.4m total
            dict(name="pose2", x=0.40, y=-0.20, theta=np.pi/2.0,
                 lift=0.50,
                 arm=[arm_each, arm_each, arm_each, arm_each],
                 wrist=[stow["wrist_yaw"], stow["wrist_pitch"], stow["wrist_roll"]]),

            # pose 3: base to (+20cm, +20cm), rotate to 180deg, rotate wrist +45deg on yaw/pitch/roll
            dict(name="pose3", x=0.20, y=0.20, theta=np.pi,
                 lift=0.50,
                 arm=[arm_each, arm_each, arm_each, arm_each],
                 wrist=[stow["wrist_yaw"] + wrist_rot,
                        stow["wrist_pitch"] + wrist_rot,
                        stow["wrist_roll"] + wrist_rot]),

            # pose 4: return to start and stow all arm joints
            dict(name="pose4", x=0.0,  y=0.0,  theta=0.0,
                 lift=stow["lift"],
                 arm=[stow["arm_l3"], stow["arm_l2"], stow["arm_l1"], stow["arm_l0"]],
                 wrist=[stow["wrist_yaw"], stow["wrist_pitch"], stow["wrist_roll"]]),
        ]

        # Plan + execute each segment (0->1, 1->2, 2->3, 3->4).
        for seg_idx in range(len(poses) - 1):
            start_pose = poses[seg_idx]
            goal_pose  = poses[seg_idx + 1]
            self.get_logger().info(f"--- Planning segment {seg_idx}: {start_pose['name']} -> {goal_pose['name']} ---")

            goal_state = RobotState(moveit.get_robot_model())
            goal_state.set_joint_group_positions(
                planning_group,
                [
                    goal_pose["x"], goal_pose["y"], goal_pose["theta"],
                    _nz(goal_pose["lift"]),
                    _nz(goal_pose["arm"][0]), _nz(goal_pose["arm"][1]), _nz(goal_pose["arm"][2]), _nz(goal_pose["arm"][3]),
                    goal_pose["wrist"][0], goal_pose["wrist"][1], goal_pose["wrist"][2],
                ],
            )

            # IMPORTANT: On some setups, TF for odom->base_link may not be available when this script starts.
            # If we call set_start_state_to_current_state() in that case, MoveIt gets an invalid (NaN) base start state
            # and OMPL will refuse to plan ("Skipping invalid start state (invalid bounds)").
            #
            # To make planning robust, we explicitly set the start state to our previous pose (start_pose).
            start_state = RobotState(moveit.get_robot_model())
            start_state.set_joint_group_positions(
                planning_group,
                [
                    start_pose["x"], start_pose["y"], start_pose["theta"],
                    _nz(start_pose["lift"]),
                    _nz(start_pose["arm"][0]), _nz(start_pose["arm"][1]), _nz(start_pose["arm"][2]), _nz(start_pose["arm"][3]),
                    start_pose["wrist"][0], start_pose["wrist"][1], start_pose["wrist"][2],
                ],
            )

            if hasattr(moveit_plan, "set_start_state"):
                try:
                    moveit_plan.set_start_state(robot_state=start_state)
                except TypeError:
                    moveit_plan.set_start_state(start_state)
            else:
                # Fallback (may fail if TF is missing)
                moveit_plan.set_start_state_to_current_state()

            moveit_plan.set_goal_state(robot_state=goal_state)

            plan = moveit_plan.plan(parameters=planning_params)
            if plan is None or getattr(plan, "trajectory", None) is None:
                self.get_logger().error("Planning failed (plan is None). Check base bounds / TF frames. Aborting.")
                break

            self.execute_plan(plan)

            # Small pause so it's easy to observe segment boundaries
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
    try:
        MoveMe()
    finally:
        # Avoid MoveIt / rclpy teardown issues causing abort on exit
        try:
            rclpy.shutdown()
        except Exception:
            pass
