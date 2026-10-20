# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division

import copy
import socket
import threading

import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from tf2_geometry_msgs import do_transform_pose

from x_mobility_navigator.inference_client import InferenceClient

IMAGE_TOPIC_NAME = '/front_stereo_camera/left/image_raw'
ODOM_TOPIC_NAME = '/chassis/odom'
CMD_TOPIC_NAME = '/cmd_vel'
ROUTE_TOPIC_NAME = '/route'
GOAL_TOPIC_NAME = '/move_base_simple/goal'
PATH_TOPIC_NAME = '/x_mobility_path'
MAPLESS_FLAG = 'is_mapless'

NUM_ROUTE_POINTS = 20
# Route vector with 4 values representing start and end positions
ROUTE_VECTOR_SIZE = 4
ROBOT_FRAME = 'base_link'


# Upsample the points between start and goal.
def upsample_points(start, goal, max_segment_length):
    x1, y1 = start
    x2, y2 = goal

    # Calculate the Euclidean distance between the two points
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Handle the case where the start and goal are too close (or identical)
    if distance <= max_segment_length:
        return [start, goal]

    # Determine the number of segments based on the maximum segment length
    num_segments = max(1, int(np.ceil(distance / max_segment_length)))

    # Generate the interpolated points
    interpolated_points = [(x1 + (i / num_segments) * (x2 - x1),
                            y1 + (i / num_segments) * (y2 - y1))
                           for i in range(num_segments + 1)]

    return interpolated_points


class XMobilityNavigator(object):
    '''Native Python 2.7 ROS Melodic node with separate GPU inference.
    '''
    def __init__(self):
        # Parameters
        self.inference_client = InferenceClient(
            rospy.get_param('~inference_socket', '/tmp/x-mobility/inference.sock'),
            rospy.get_param('~inference_timeout', 5.0))
        self.is_mapless = rospy.get_param('~' + MAPLESS_FLAG, True)
        self.robot_frame = rospy.get_param('~robot_frame', ROBOT_FRAME)
        self.inference_rate = rospy.get_param('~inference_rate', 5.0)
        if not np.isfinite(self.inference_rate) or self.inference_rate <= 0:
            raise ValueError('inference_rate must be a positive finite number')

        # rospy subscriber callbacks run on other threads. Initialize all shared
        # state before subscribing, then serialize updates with inference.
        self._state_lock = threading.RLock()
        self.action = np.zeros(6, dtype=np.float32)
        self.path = np.zeros(10, dtype=np.float32)
        self.history = np.zeros((1, 1024), dtype=np.float32)
        self.sample = np.zeros((1, 512), dtype=np.float32)
        self.camera_image = None
        self.route_vectors = None
        self.goal = None
        self.ego_speed = None
        self.cv_bridge = CvBridge()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher
        self.cmd_publisher = rospy.Publisher(CMD_TOPIC_NAME, Twist, queue_size=1)
        self.path_publisher = rospy.Publisher(PATH_TOPIC_NAME, Path, queue_size=1)

        # Subscriber
        self.image_subscriber = rospy.Subscriber(IMAGE_TOPIC_NAME,
                                                Image,
                                                self.image_callback,
                                                queue_size=1,
                                                buff_size=2**24)
        self.odom_subscriber = rospy.Subscriber(
            ODOM_TOPIC_NAME, Odometry, self.odom_callback, queue_size=1)
        self.route_subscriber = rospy.Subscriber(
            ROUTE_TOPIC_NAME, Path, self.route_callback, queue_size=1)
        self.goal_subscriber = rospy.Subscriber(
            GOAL_TOPIC_NAME, PoseStamped, self.goal_callback, queue_size=1)

    def run(self):
        # Callbacks update inputs under a lock; only this loop uses the socket.
        rate = rospy.Rate(self.inference_rate, reset=True)
        while not rospy.is_shutdown():
            self.inference()
            rate.sleep()

    def image_callback(self, image_msg):
        try:
            image = self.process_image_msg(image_msg)
        except CvBridgeError as ex:
            rospy.logwarn_throttle(5.0, 'Could not convert camera image: %s' % ex)
            return
        with self._state_lock:
            self.camera_image = image

    def odom_callback(self, odom_msg):
        with self._state_lock:
            self.ego_speed = np.array(odom_msg.twist.twist.linear.x,
                                     dtype=np.float32)

    def goal_callback(self, goal_msg):
        with self._state_lock:
            self.goal = goal_msg
            # Reset history state
            self.history = np.zeros((1, 1024), dtype=np.float32)
            self.sample = np.zeros((1, 512), dtype=np.float32)

    def route_callback(self, route_msg):
        if self.is_mapless:
            return
        with self._state_lock:
            self._update_route(route_msg)

    def _update_route(self, route_msg):
        self.route_vectors = None
        # Get transform between route and robot.
        try:
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame, route_msg.header.frame_id, rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn_throttle(
                5.0,
                'Could not transform %s to %s: %s' %
                (route_msg.header.frame_id, self.robot_frame, ex)
            )
            return

        route_poses = route_msg.poses
        num_poses = min(len(route_poses), NUM_ROUTE_POINTS)
        # Return if route is empty.
        if num_poses == 0:
            return
        # Select the first NUM_ROUTE_POINTS and append the last route point as needed.
        indices = [idx for idx in range(num_poses)]
        indices.extend([num_poses - 1] * (NUM_ROUTE_POINTS - len(indices)))
        # Extract the x and y position in robot frame.
        selected_route_positions = []
        for idx in indices:
            transformed_pose = do_transform_pose(route_poses[idx], transform)
            selected_route_positions.append(
                [
                    transformed_pose.pose.position.x,
                    transformed_pose.pose.position.y
                ])
        self.route_vectors = np.zeros(
            (NUM_ROUTE_POINTS - 1, ROUTE_VECTOR_SIZE), np.float32)
        for idx in range(NUM_ROUTE_POINTS - 1):
            self.route_vectors[idx] = np.concatenate(
                (selected_route_positions[idx],
                 selected_route_positions[idx + 1]),
                axis=0)

    def compose_mapless_route(self):
        self.route_vectors = None
        if self.goal is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame, self.goal.header.frame_id, rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn_throttle(
                5.0,
                'Could not transform %s to %s: %s' %
                (self.goal.header.frame_id, self.robot_frame, ex)
            )
            return
        goal_in_robot_frame = do_transform_pose(self.goal, transform)
        route_poses = upsample_points(
            [0.0, 0.0],
            [
                goal_in_robot_frame.pose.position.x,
                goal_in_robot_frame.pose.position.y
            ], 1.0)
        num_poses = min(len(route_poses), NUM_ROUTE_POINTS)
        # Return if route is empty.
        if num_poses == 0:
            return
        # Select the first NUM_ROUTE_POINTS and append the last route point as needed.
        indices = [idx for idx in range(num_poses)]
        indices.extend([num_poses - 1] * (NUM_ROUTE_POINTS - len(indices)))
        # Extract the x and y position in robot frame.
        selected_route_positions = []
        for idx in indices:
            selected_route_positions.append(route_poses[idx])
        self.route_vectors = np.zeros(
            (NUM_ROUTE_POINTS - 1, ROUTE_VECTOR_SIZE), np.float32)
        for idx in range(NUM_ROUTE_POINTS - 1):
            self.route_vectors[idx] = np.concatenate(
                (selected_route_positions[idx],
                 selected_route_positions[idx + 1]),
                axis=0)

    def inference(self):
        with self._state_lock:
            self._inference()

    def _inference(self):
        # Compose a simple route in mapless mode.
        if self.is_mapless:
            self.compose_mapless_route()

        # Sanity checks of the inputs.
        # TODO: Sync the msgs.
        if self.camera_image is None or self.route_vectors is None or self.ego_speed is None:
            rospy.loginfo_throttle(
                5.0, 'Waiting for image, odometry and route/goal.')
            return
        try:
            self._trt_inference()
        except (socket.error, IOError, OSError, ValueError, KeyError, TypeError) as ex:
            self.inference_client.close()
            self.action.fill(0)
            self.publish_action()
            rospy.logerr_throttle(5.0, 'Inference service unavailable: %s' % ex)
            return
        self.publish_action()
        self.publish_path()

    def _trt_inference(self):
        outputs = self.inference_client.infer({
            'image': self.camera_image,
            'route': self.route_vectors,
            'speed': self.ego_speed,
            'action_input': self.action,
            'history_input': self.history,
            'sample_input': self.sample,
        })
        expected = {'action_output': self.action.shape,
                    'path_output': self.path.shape,
                    'history_output': self.history.shape,
                    'sample_output': self.sample.shape}
        if set(outputs) != set(expected):
            raise ValueError('Unexpected inference output names')
        values = {}
        for name, shape in expected.items():
            value = outputs[name].reshape(shape)
            if not np.all(np.isfinite(value)):
                raise ValueError('Non-finite inference output: ' + name)
            values[name] = value
        self.action = values['action_output']
        self.path = values['path_output']
        self.history = values['history_output']
        self.sample = values['sample_output']

    def publish_action(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(self.action[0])
        cmd_vel.angular.z = float(self.action[5])
        self.cmd_publisher.publish(cmd_vel)

    def publish_path(self):
        path = Path()
        path.header.frame_id = self.robot_frame
        path.header.stamp = rospy.Time.now()
        for idx in range(len(self.route_vectors)):
            path_pose = PoseStamped()
            path_pose.header = path.header
            path_pose.pose.position.x = float(self.route_vectors[idx][0])
            path_pose.pose.position.y = float(self.route_vectors[idx][1])
            path_pose.pose.orientation.w = 1.0
            path.poses.append(path_pose)
        self.path_publisher.publish(path)

    def process_image_msg(self, image_msg):
        # Melodic's Python cv_bridge assumes packed rows. Strip any padding
        # before asking it to convert the camera encoding to RGB.
        dtype, channels = self.cv_bridge.encoding_to_dtype_with_channels(
            image_msg.encoding)
        packed_step = image_msg.width * channels * np.dtype(dtype).itemsize
        if (image_msg.width <= 0 or image_msg.height <= 0
                or image_msg.step < packed_step
                or len(image_msg.data) != image_msg.height * image_msg.step):
            raise CvBridgeError('Invalid image dimensions, step or data length')
        if image_msg.step != packed_step:
            rows = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
                image_msg.height, image_msg.step)
            image_msg = copy.copy(image_msg)
            image_msg.data = rows[:, :packed_step].copy().tobytes()
            image_msg.step = packed_step
        image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(image)


def main():
    rospy.init_node('x_mobility_navigator')
    x_mobility_navigator = XMobilityNavigator()
    try:
        x_mobility_navigator.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        x_mobility_navigator.inference_client.close()


if __name__ == '__main__':
    main()
