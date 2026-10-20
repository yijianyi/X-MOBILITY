"""ROS 1 migration regressions; ROS and GPU dependencies are test doubles."""

import importlib.util
import sys
import threading
import unittest
from pathlib import Path as FilePath
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

SOURCE_ROOT = FilePath(__file__).resolve().parents[1] / 'x_mobility_navigator/src'
sys.path.insert(0, str(SOURCE_ROOT))


def vector():
    return SimpleNamespace(x=0.0, y=0.0, z=0.0)


def header():
    return SimpleNamespace(frame_id='', stamp=None)


class PoseStamped:
    def __init__(self):
        self.header = header()
        self.pose = SimpleNamespace(
            position=vector(),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0))


class Path:
    def __init__(self):
        self.header = header()
        self.poses = []


class Twist:
    def __init__(self):
        self.linear = vector()
        self.angular = vector()


class Time:
    def __init__(self, secs=0):
        self.secs = secs

    @staticmethod
    def now():
        return Time(123)


class TransformError(Exception):
    pass


class CvBridgeError(Exception):
    pass


def make_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


def transform_pose(pose, transform):
    # Melodic's do_transform_pose consumes and returns PoseStamped, not Pose.
    if not isinstance(pose, PoseStamped):
        raise TypeError('ROS 1 TF requires PoseStamped')
    result = PoseStamped()
    result.pose.position.x = pose.pose.position.x + transform.x
    result.pose.position.y = pose.pose.position.y + transform.y
    return result


class NavigatorTests(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.ros = make_module(
            'rospy', get_param=Mock(side_effect=lambda key, default: self.params.get(key, default)),
            Publisher=Mock(side_effect=lambda *args, **kwargs: Mock()),
            Subscriber=Mock(), Time=Time, Rate=Mock(),
            is_shutdown=Mock(), init_node=Mock(),
            loginfo=Mock(), loginfo_throttle=Mock(), logwarn_throttle=Mock(),
            logerr_throttle=Mock(),
            ROSInterruptException=TransformError)
        self.tf_buffer = Mock()
        self.tf_buffer.lookup_transform.return_value = SimpleNamespace(x=2, y=-1)
        self.bridge = Mock()
        self.bridge.encoding_to_dtype_with_channels.return_value = (np.uint8, 3)
        modules = {
            'rospy': self.ros,
            'tf2_ros': make_module(
                'tf2_ros', Buffer=Mock(return_value=self.tf_buffer),
                TransformListener=Mock(), LookupException=TransformError,
                ConnectivityException=TransformError,
                ExtrapolationException=TransformError),
            'tf2_geometry_msgs': make_module('tf2_geometry_msgs',
                                             do_transform_pose=transform_pose),
            'cv_bridge': make_module('cv_bridge', CvBridge=Mock(return_value=self.bridge),
                                     CvBridgeError=CvBridgeError),
        }
        for name, attributes in (
                ('geometry_msgs', dict(Twist=Twist, PoseStamped=PoseStamped)),
                ('nav_msgs', dict(Odometry=object, Path=Path)),
                ('sensor_msgs', dict(Image=object))):
            modules[name] = make_module(name)
            modules[name + '.msg'] = make_module(name + '.msg', **attributes)
        source = FilePath(__file__).resolve().parents[1] / (
            'x_mobility_navigator/src/x_mobility_navigator/x_mobility_navigator.py')
        spec = importlib.util.spec_from_file_location('navigator_under_test', source)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', modules):
            spec.loader.exec_module(self.module)

    def navigator(self, **params):
        self.params.update({'~' + name: value for name, value in params.items()})
        return self.module.XMobilityNavigator()

    def test_private_parameters_and_ros1_topics(self):
        node = self.navigator(inference_socket='/run/nav.sock', robot_frame='robot')
        self.assertEqual(node.inference_client.socket_path, '/run/nav.sock')
        self.assertEqual(node.robot_frame, 'robot')
        self.assertEqual(self.ros.Subscriber.call_args_list[-1].args[:2],
                         ('/move_base_simple/goal', PoseStamped))
        self.assertEqual(self.ros.Publisher.call_args_list[0].args[:2],
                         ('/cmd_vel', Twist))

    def test_invalid_inference_rates_are_rejected(self):
        for rate in (0.0, -1.0, float('inf'), float('nan')):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                self.navigator(inference_rate=rate)

    def test_mapless_goal_uses_stamped_pose_and_latest_ros_time(self):
        node = self.navigator(robot_frame='robot')
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1
        node.goal_callback(goal)
        node.compose_mapless_route()
        self.assertEqual(node.route_vectors.shape, (19, 4))
        self.assertEqual(node.route_vectors.dtype, np.float32)
        np.testing.assert_allclose(node.route_vectors[-1], [3, -1, 3, -1])
        args = self.tf_buffer.lookup_transform.call_args.args
        self.assertEqual(args[:2], ('robot', 'map'))
        self.assertIsInstance(args[2], Time)
        self.assertEqual(args[2].secs, 0)

    def test_external_route_transforms_stamped_poses_and_pads(self):
        node = self.navigator(is_mapless=False)
        route = Path()
        route.header.frame_id = 'map'
        route.poses = [PoseStamped(), PoseStamped()]
        route.poses[1].pose.position.x = 4
        node.route_callback(route)
        np.testing.assert_allclose(node.route_vectors[0], [2, -1, 6, -1])
        np.testing.assert_allclose(node.route_vectors[-1], [6, -1, 6, -1])
        route.poses = []
        node.route_callback(route)
        self.assertIsNone(node.route_vectors)

    def test_mapless_mode_ignores_external_routes(self):
        node = self.navigator()
        node.route_callback(Path())
        self.tf_buffer.lookup_transform.assert_not_called()
        self.assertIsNone(node.route_vectors)

    def test_tf_failure_discards_old_route(self):
        node = self.navigator()
        node.goal = PoseStamped()
        node.route_vectors = np.ones((19, 4))
        self.tf_buffer.lookup_transform.side_effect = TransformError('no transform')
        node.compose_mapless_route()
        self.assertIsNone(node.route_vectors)
        self.ros.logwarn_throttle.assert_called_once()

    def test_image_conversion_requests_rgb_and_normalizes_chw(self):
        node = self.navigator()
        image = SimpleNamespace(data=bytes([255, 0, 128, 0, 255, 64]),
                                width=2, height=1, step=6, encoding='rgb8')
        self.bridge.imgmsg_to_cv2.return_value = np.frombuffer(
            image.data, dtype=np.uint8).reshape(1, 2, 3)
        node.image_callback(image)
        self.bridge.imgmsg_to_cv2.assert_called_once_with(image, desired_encoding='rgb8')
        self.assertEqual(node.camera_image.shape, (3, 1, 2))
        self.assertEqual(node.camera_image.dtype, np.float32)
        self.assertTrue(node.camera_image.flags.c_contiguous)
        np.testing.assert_allclose(node.camera_image[:, 0, 0], [1, 0, 128 / 255])

    def test_bad_image_is_logged_without_replacing_last_frame(self):
        node = self.navigator()
        last_frame = np.zeros((3, 1, 1), dtype=np.float32)
        node.camera_image = last_frame
        self.bridge.encoding_to_dtype_with_channels.side_effect = CvBridgeError(
            'invalid encoding')
        node.image_callback(SimpleNamespace(encoding='invalid'))
        self.assertIs(node.camera_image, last_frame)
        self.ros.logwarn_throttle.assert_called_once()

    def test_padded_ros1_image_rows_are_packed_before_conversion(self):
        node = self.navigator()
        image = SimpleNamespace(data=bytes([255, 0, 0, 99, 0, 255, 0, 99]),
                                width=1, height=2, step=4, encoding='rgb8')
        self.bridge.imgmsg_to_cv2.side_effect = lambda msg, **kwargs: np.frombuffer(
            msg.data, dtype=np.uint8).reshape(2, 1, 3)
        node.image_callback(image)
        packed = self.bridge.imgmsg_to_cv2.call_args.args[0]
        self.assertEqual(packed.step, 3)
        self.assertEqual(packed.data, bytes([255, 0, 0, 0, 255, 0]))
        self.assertEqual(image.step, 4)
        np.testing.assert_allclose(node.camera_image[:, :, 0], [[1, 0], [0, 1], [0, 0]])

    def test_invalid_image_layout_is_rejected(self):
        node = self.navigator()
        for width, height, step, data in (
                (0, 1, 0, b''), (1, 0, 3, b''),
                (2, 1, 3, b'abc'), (1, 2, 3, b'abc')):
            with self.subTest(width=width, height=height, step=step):
                image = SimpleNamespace(width=width, height=height, step=step,
                                        data=data, encoding='rgb8')
                node.image_callback(image)
                self.assertIsNone(node.camera_image)
        self.bridge.imgmsg_to_cv2.assert_not_called()

    def test_run_keeps_inference_on_calling_thread(self):
        node = self.navigator(inference_rate=7.5)
        threads = []
        node.inference = lambda: threads.append(threading.get_ident())
        self.ros.is_shutdown.side_effect = [False, False, True]
        node.run()
        self.assertEqual(threads, [threading.get_ident()] * 2)
        self.ros.Rate.assert_called_once_with(7.5, reset=True)
        self.assertEqual(self.ros.Rate.return_value.sleep.call_count, 2)

    def test_route_interpolation_keeps_fractional_positions(self):
        points = self.module.upsample_points([0, 0], [3, 0], 1.0)
        self.assertEqual(points, [(0, 0), (1, 0), (2, 0), (3, 0)])

    def test_goal_reset_waits_for_inference_lock(self):
        node = self.navigator()
        node.history[:] = 1
        node.sample[:] = 1
        started = threading.Event()
        completed = threading.Event()

        def update_goal():
            started.set()
            node.goal_callback(PoseStamped())
            completed.set()

        with node._state_lock:  # pylint: disable=protected-access
            worker = threading.Thread(target=update_goal)
            worker.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(completed.wait(0.05))
            self.assertTrue(np.all(node.history == 1))
        worker.join(timeout=1)
        self.assertTrue(completed.is_set())
        self.assertFalse(np.any(node.history))
        self.assertFalse(np.any(node.sample))

    def test_missing_inputs_skip_gpu_and_publication(self):
        node = self.navigator()
        node._trt_inference = Mock()  # pylint: disable=protected-access
        node.inference()
        node._trt_inference.assert_not_called()  # pylint: disable=protected-access
        node.cmd_publisher.publish.assert_not_called()

    def test_messages_use_ros1_timestamps_and_configured_frame(self):
        node = self.navigator(robot_frame='robot')
        node.action[:] = [0.75, 0, 0, 0, 0, -0.5]
        node.route_vectors = np.zeros((19, 4), dtype=np.float32)
        node.publish_action()
        node.publish_path()
        command = node.cmd_publisher.publish.call_args.args[0]
        path = node.path_publisher.publish.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (0.75, -0.5))
        self.assertEqual(path.header.frame_id, 'robot')
        self.assertIsInstance(path.header.stamp, Time)
        self.assertEqual(path.header.stamp.secs, 123)
        self.assertEqual(len(path.poses), 19)
        self.assertEqual(path.poses[0].pose.orientation.w, 1)

    def test_successful_inference_updates_recurrent_state_and_publishes(self):
        node = self.navigator(is_mapless=False)
        node.camera_image = np.zeros((3, 1, 2), dtype=np.float32)
        node.route_vectors = np.zeros((19, 4), dtype=np.float32)
        node.ego_speed = np.array(0.5, dtype=np.float32)
        node.inference_client = Mock()
        node.inference_client.infer.return_value = {
            'action_output': np.full((1, 6), 7, dtype=np.float32),
            'path_output': np.full((1, 10), 8, dtype=np.float32),
            'history_output': np.full((1, 1024), 9, dtype=np.float32),
            'sample_output': np.full((1, 512), 10, dtype=np.float32),
        }
        node.inference()
        request = node.inference_client.infer.call_args.args[0]
        self.assertEqual(set(request), {'image', 'route', 'speed', 'action_input',
                                        'history_input', 'sample_input'})
        np.testing.assert_array_equal(node.history, np.full((1, 1024), 9))
        np.testing.assert_array_equal(node.sample, np.full((1, 512), 10))
        self.assertEqual(node.cmd_publisher.publish.call_count, 1)
        self.assertEqual(node.path_publisher.publish.call_count, 1)

    def test_failed_service_publishes_zero_and_reconnects_next_cycle(self):
        node = self.navigator(is_mapless=False)
        node.action.fill(0.5)
        node.camera_image = np.zeros((3, 1, 2), dtype=np.float32)
        node.route_vectors = np.zeros((19, 4), dtype=np.float32)
        node.ego_speed = np.array(0.5, dtype=np.float32)
        node.inference_client = Mock()
        node.inference_client.infer.side_effect = IOError('connection lost')
        node.inference()
        node.inference_client.close.assert_called_once()
        command = node.cmd_publisher.publish.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (0.0, 0.0))
        node.path_publisher.publish.assert_not_called()
        self.ros.logerr_throttle.assert_called_once()

    def test_wrong_output_shape_does_not_commit_partial_state(self):
        node = self.navigator(is_mapless=False)
        node.camera_image = np.zeros((3, 1, 2), dtype=np.float32)
        node.route_vectors = np.zeros((19, 4), dtype=np.float32)
        node.ego_speed = np.array(0.5, dtype=np.float32)
        node.inference_client = Mock()
        node.inference_client.infer.return_value = {
            'action_output': np.full(6, 7, dtype=np.float32),
            'path_output': np.full(10, 8, dtype=np.float32),
            'history_output': np.full(1, 9, dtype=np.float32),
            'sample_output': np.full(512, 10, dtype=np.float32),
        }
        node.inference()
        self.assertFalse(np.any(node.action))
        self.assertFalse(np.any(node.history))
        node.path_publisher.publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
