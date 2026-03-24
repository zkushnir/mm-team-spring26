#! /usr/bin/env python3

# Adapted from the simple commander demo examples on
# https://github.com/ros-planning/navigation2/blob/galactic/nav2_simple_commander/nav2_simple_commander/demo_security.py

from copy import deepcopy
import threading

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from stretch_nav2.robot_navigator import BasicNavigator, TaskResult

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
from cv_bridge import CvBridge

"""
Goals: navigate to 4 poses
record video from head camera while navigating
collect poses from RViz using '2D Pose Estimate' tool:
    ros2 launch stretch_nav2 navigation.launch.py map:=${HELLO_FLEET_PATH}/maps/<map_name>.yaml
then read x, y, and quaternion (z, w) from base_footprint TF branch.
"""

# Output video file
VIDEO_OUTPUT   = 'maker_space_tour.avi'
VIDEO_FPS      = 10
HEAD_CAM_TOPIC = '/camera/color/image_raw'


class CameraRecorder(Node):
    #subscribe to camera topic to get video file

    def __init__(self):
        super().__init__('camera_recorder')
        self.bridge = CvBridge()
        self.writer = None
        self.recording = False
        self.subscription = self.create_subscription(Image,HEAD_CAM_TOPIC,self._image_callback,10)

    def start_recording(self, filepath, fps=VIDEO_FPS):
        self.filepath = filepath
        self.fps = fps
        self.recording = True
        self.get_logger().info(f'Camera recorder will write to {filepath}')

    def stop_recording(self):
        self.recording = False
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info('Video saved')

    def _image_callback(self, msg):
        if not self.recording:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (w, h))
        self.writer.write(frame)


def main():
    rclpy.init()

    navigator = BasicNavigator()
    recorder  = CameraRecorder()

    # Spin the camera recorder node in a background thread so it keeps
    # receiving frames while the main thread blocks on navigation calls.
    recorder_thread = threading.Thread(
        target=rclpy.spin, args=(recorder,), daemon=True)
    recorder_thread.start()

    # Poses: [x, y, quat_z, quat_w]
    # Quaternion from yaw: qz = sin(yaw/2), qw = cos(yaw/2), qx = qy = 0.
    # ------------------------------------
    # adjustable initial pose - use same one as simple_commander for now
    initial_pose_data = [0.0,    0.0,    0.0,    1.0] 

    nav_route = [
        [2.50,   1.80,   0.707,  0.707],   # Pose 1 — facing +y
        [2.30,  -2.10,   0.0,    1.0  ],   # Pose 2 — facing +x
        [-1.60, -2.40,  -0.707,  0.707],   # Pose 3 — facing -y
        [-2.00,  1.50,   1.0,    0.0  ],   # Pose 4 — facing -x
    ]

    # Set the initial pose
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = initial_pose_data[0]
    initial_pose.pose.position.y = initial_pose_data[1]
    initial_pose.pose.orientation.z = initial_pose_data[2]
    initial_pose.pose.orientation.w = initial_pose_data[3]
    navigator.setInitialPose(initial_pose)

    navigator.waitUntilNav2Active()

    # Start camera recording
    recorder.start_recording(VIDEO_OUTPUT)

    # Navigate to each pose in sequence
    for idx, pt in enumerate(nav_route):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = navigator.get_clock().now().to_msg()
        pose.pose.position.x = pt[0]
        pose.pose.position.y = pt[1]
        pose.pose.orientation.z = pt[2]
        pose.pose.orientation.w = pt[3]

        navigator.get_logger().info(
            f'Navigating to pose {idx + 1}/{len(nav_route)}: '
            f'x={pt[0]}, y={pt[1]}, qz={pt[2]}, qw={pt[3]}')

        navigator.goToPose(pose)

        i = 0
        while not navigator.isTaskComplete():
            i += 1
            feedback = navigator.getFeedback()
            if feedback and i % 5 == 0:
                navigator.get_logger().info(
                    f'Distance remaining to pose {idx + 1}: '
                    f'{feedback.distance_remaining:.2f} m')

        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            navigator.get_logger().info(f'Reached pose {idx + 1}!')
        elif result == TaskResult.CANCELED:
            navigator.get_logger().info('Navigation canceled. Stopping tour.')
            break
        elif result == TaskResult.FAILED:
            navigator.get_logger().info(f'Failed to reach pose {idx + 1}. Continuing...')

    # Stop recording and clean up
    recorder.stop_recording()
    navigator.get_logger().info('Tour complete, video saved to ' + VIDEO_OUTPUT)

    rclpy.shutdown()


if __name__ == '__main__':
    main()