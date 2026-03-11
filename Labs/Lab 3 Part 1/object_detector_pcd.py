import cv2
import yaml
import rclpy
import os.path as osp
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import detection_utils
import message_filters
import numpy as np


# Don't forget to start the camera before starting this node!
# Part 1: using in-gripper camera
#    ros2 launch stretch_core d405_basic.launch.py
# Part 2: using head camera
#    ros2 launch stretch_core d435i_low_resolution.launch.py
#
# ros2 run rviz2 rviz2 -d `ros2 pkg prefix --share stretch_calibration`/rviz/stretch_simple_test.rviz


class YOLOEObjectDetector(Node):
    def __init__(self, obj_queries):
        super().__init__('yoloe_object_detector')
        self.visualize = True

        # ----------- Camera Streaming Setup -----------

        # subscribe to the robot's color and aligned depth camera image topics from the gripper camera
        # using message_filters, instead of self.create_subscription() to allow us
        #   to synchronize the two camera streams can use a single callback that triggers when both come in
        # TODO: ------------- start --------------
        # leave as is for part 1, 
        # change for part 2 to use the head camera

        # Part 2: use the head camera (D435i) instead of the gripper camera (D405)
        # The D435i launch (d435i_basic.launch.py) uses camera_name='camera', so topics are under /camera/
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.color_cam_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info')
        # TODO: -------------- end ---------------
        self.latest_color = None
        self.latest_depth = None
        self.latest_color_cam_info = None

        # Use ApproximateTimeSynchronizer and register a callback function that runs within some time tolerance of when both images are received
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.color_cam_info_sub],
            queue_size=10,
            slop=0.01  # 10ms tolerance
        )
        self.synchronizer.registerCallback(self.image_callback)

        # bridge to convert ROS2 image messages to OpenCV images
        self.bridge = CvBridge()

        # -----------------------------------------------------

        # ----------- YOLO-E Object Detection SetuP -----------

        # Load the YOLOE model, which should already saved to common models directory on the robot
        #   we use yolo-e-v26-small for its high performance and low latency on limited compute
        model_path = '/home/hello-robot/models'
        model_name = 'yoloe-26s-seg.pt'
        self.model = YOLO(osp.join(model_path, model_name))

        # pass prompt for the object/s you want to detect
        self.obj_queries = obj_queries
        self.model.set_classes(self.obj_queries)

        # Run the detector and goals at a fixed frequency to reduce latency introduced by the detector
        #   and give the robot time to move between poses
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.publish_goals_callback)
        self.goal_pub = self.create_publisher(PoseStamped, '/object_detector/goal_pose', 10)
        self.goal_pose_msg = None

        # -----------------------------------------------------

    def image_callback(self, color_msg, depth_msg, color_cam_info_msg):
        # convert the color and depth ROS2 image messages to OpenCV images
        # TODO: ------------- start --------------
        # in part 1,fill with your response
        #   you may need to nest things in a try, except in case frames are missing
        #.  if you are unpacking frames correctly, you should see the live color and depth output
        #   plotted in a cv2 window by detection_utils.visualize_detection_masks()
        # in part 2, you may need to make changes to the code to handle the head camera orientation


        # Part 2: the head camera (D435i) image may be rotated relative to the gripper camera.
        # We rotate the images 90 degrees clockwise so the scene appears upright for detection.
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Rotate 90° clockwise to correct head camera orientation if needed
            # (comment these two lines out if the image already looks upright)
            # color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
            # depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)

            self.latest_color = color
            self.latest_depth = depth
            self.latest_color_cam_info = color_cam_info_msg

        except CvBridgeError as e:
            self.get_logger().warn(f'CvBridge error in image_callback: {e}')
            return

        # TODO: -------------- end ---------------


    def publish_goals_callback(self):
        # Guard: skip if we haven't received frames yet
        if self.latest_color is None or self.latest_depth is None:
            print("Waiting for camera frames...")
            return

        # run object detection on the RGB image
        # TODO: ------------- start --------------
        # fill with your response
        #   pass the color frame to YOLO-E, parse the results using detection_utils.parse_results()

        results = self.model(self.latest_color, verbose=False)
        detections = detection_utils.parse_results(results)
        if len(detections) == 0:
            detections = None

        # TODO: -------------- end ---------------

        # create visualizations from the detections
        if self.visualize:
            detection_utils.visualize_detections_masks(
                # TODO: minor - change the part= arg when you edit your code for part 2! 
                #   adjusts the color scaling of the depth image display to match the camera range
                part=2, detections=detections, rgb_image=self.latest_color, depth_image=self.latest_depth)

        # get the goal pose and publish it, if it exists
        self.get_goal_pose(detections)

        if self.goal_pose_msg is None:
            print("OBJECT NOT DETECTED, no pose to publish")
            return
        else:
            self.goal_pub.publish(self.goal_pose_msg)
            print()
            print("---------- Published Goal Pose ----------")




    def get_goal_pose(self, detections, target_idx=0):
        if detections is None or len(detections) == 0:
            self.goal_pose_msg = None
            return None

        # TODO: ------------- start --------------
        # in part 1, fill with your response
        #   find the depth at the centroid and project it to 3D using detection_utils.pixel_to_3d()
        #   convert that pose to a PoseStamped msg using detection_utils.get_pose_msg()
        #   save that message to self.goal_pose_msg
        # in part 2, edit the code you wrote for part 1 to now project all points in the mask to 3D,
        #   then get the centroid of the resulting pointcloud to use as the goal pose (instead of the 2D centroid in part 1)

        target_detection = detections[target_idx]
        mask = target_detection["mask"]  # Nx2 array of [x, y] pixel coords

        # --- Part 2: project all mask points to 3D, compute pointcloud centroid ---
        camera_mat = np.reshape(self.latest_color_cam_info.k, (3, 3))
        f_x = camera_mat[0, 0]
        c_x = camera_mat[0, 2]
        f_y = camera_mat[1, 1]
        c_y = camera_mat[1, 2]

        points_3d = []
        for pt in mask:
            px, py = int(pt[0]), int(pt[1])
            # bounds check
            if py < 0 or py >= self.latest_depth.shape[0] or px < 0 or px >= self.latest_depth.shape[1]:
                continue
            z_depth = float(self.latest_depth[py, px])
            if z_depth == 0.0:
                continue  # skip invalid depth
            z_m = z_depth / 1000.0  # convert mm to meters
            x_m = ((px - c_x) * z_m) / f_x
            y_m = ((py - c_y) * z_m) / f_y
            points_3d.append([x_m, y_m, z_m])

        if len(points_3d) == 0:
            print("No valid 3D points from mask, skipping goal pose update")
            self.goal_pose_msg = None
            return None

        points_3d = np.array(points_3d)
        centroid_3d = points_3d.mean(axis=0)
        print(f"3D centroid: {centroid_3d * 100} cm  ({len(points_3d)} valid mask points)")

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.latest_color_cam_info.header.frame_id
        msg.pose.position.x = float(centroid_3d[0])
        msg.pose.position.y = float(centroid_3d[1])
        msg.pose.position.z = float(centroid_3d[2])
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_pose_msg = msg

        # TODO: -------------- end ---------------


if __name__ == '__main__':
    rclpy.init()

    # load in the full list of object queries from the yaml file, as well as a target (if specified)
    with open('object_queries.yaml', 'r') as file:
        config = yaml.safe_load(file)
        obj_queries = config['queries']

    yolo_object_detector = YOLOEObjectDetector(obj_queries)
    rclpy.spin(yolo_object_detector)
    yolo_object_detector.destroy_node()
    rclpy.shutdown()
