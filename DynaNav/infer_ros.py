#!/usr/bin/env python3
"""TIC-VLA ROS 2 image subscriber; replay bags separately with ros2 bag play.

Keep beside infer_video_async.py. --bag selects topics only; it does not start
playback. --inspect-bag requires only Python's standard library. Outputs are
under /ticvla by default; velocity is a Float32MultiArray [v, omega]. No robot
command topic is published. Without odometry, model state is zero, as in the
video script and supplied reference. Overlay is schematic, not calibrated.

From the repository root, in a Python environment containing both ROS 2 and
TIC-VLA dependencies (rclpy/cv_bridge must match the interpreter ABI):
  python DynaNav/infer_ros.py --bag cross_gait_scenario_1_final.bag_0.db3
In a second sourced ROS terminal, after the node reports Model ready:
  ros2 bag play cross_gait_scenario_1_final.bag_0.db3 --storage sqlite3 \
    --topics /argus/ar0234_front_left/image_raw --rate 0.25
The node uses the latest frame, so warmup/slow inference can skip observations.
Ctrl-C stops the node after playback; it can also receive a live camera topic.
"""
from __future__ import annotations

import argparse
from collections import deque
import math
from pathlib import Path
import sqlite3
import tempfile
import time


def bag_topics(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        files = sorted(list(path.glob('*.db3')) + list(path.glob('*.bd3')))
        if not files:
            raise ValueError(f'No SQLite .db3/.bd3 files in {path}')
        path = files[0]
    if not path.is_file():
        raise FileNotFoundError(f'Bag not found: {path}')
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        return db.execute('SELECT name, type FROM topics ORDER BY name').fetchall()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', default=None, help='SQLite ROS 2 bag/file for topic discovery; replay separately')
    parser.add_argument('--inspect-bag', action='store_true')
    parser.add_argument('--image-topic')
    parser.add_argument('--image-type', choices=('raw', 'compressed'))
    parser.add_argument('--checkpoint', default=str(Path(__file__).resolve().parents[1] / 'checkpoints/TIC-VLA-model.ckpt'))
    parser.add_argument('--base-model', default='OpenGVLab/InternVL3-1B')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--robot-type', default='legged robot', help='Robot description supplied to the VLM')
    parser.add_argument('-i', '--instruction', default='Move forward safely and avoid obstacles.')
    parser.add_argument('--rate', type=float, default=2.0, help='Maximum inference calls per wall-clock second')
    parser.add_argument('-s', '--scale', type=float, default=1.0)
    parser.add_argument('--v-max', type=float, default=1.5)
    parser.add_argument('--w-max', type=float, default=1.2)
    parser.add_argument('--frame-id', default='base_link', help='Robot frame in which predicted waypoints are expressed')
    parser.add_argument('--prefix', default='/ticvla')
    parser.add_argument('--show', action='store_true')
    args, ros_args = parser.parse_known_args()
    if ros_args and ros_args[0] != '--ros-args':
        parser.error(f'Unrecognized arguments: {ros_args}')
    if not all(math.isfinite(v) and v > 0 for v in (args.rate, args.scale, args.v_max, args.w_max)):
        parser.error('rate, scale, v-max and w-max must be finite and positive')
    if args.inspect_bag and not args.bag:
        parser.error('--inspect-bag requires --bag')
    if args.bag:
        topics = bag_topics(args.bag)
        if args.inspect_bag:
            for name, kind in topics:
                print(f'{name}\t{kind}')
            return args, ros_args
        candidates = [(n, t) for n, t in topics if t in ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage')]
        if args.image_topic:
            candidates = [(n, t) for n, t in candidates if n == args.image_topic]
        if len(candidates) != 1:
            parser.error(f'Select --image-topic from bag camera topics: {candidates or topics}')
        args.image_topic, kind = candidates[0]
        actual = 'compressed' if kind.endswith('/CompressedImage') else 'raw'
        if args.image_type and args.image_type != actual:
            parser.error('--image-type conflicts with bag metadata')
        args.image_type = actual
    return args, ros_args


def main():
    args, ros_args = parse_args()
    if args.inspect_bag:
        return 0
    # Lazy imports keep bag inspection available outside a ROS/model environment.
    import cv2
    import numpy as np
    import torch
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CompressedImage
    from std_msgs.msg import Empty, Float32MultiArray
    from nav_msgs.msg import Path as PathMsg
    from geometry_msgs.msg import PoseStamped
    from infer_video_async import load_ticvla, WaypointController, draw_waypoints_simple, save_temp_frame
    from threading import Lock

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; activate the GPU environment/driver.')

    class TICVLANode(Node):
        def __init__(self, model, temp_dir):
            super().__init__('ticvla_inference')
            self.model, self.temp_dir = model, Path(temp_dir)
            self.bridge = CvBridge()
            self.lock = Lock()
            self.latest = None
            self.last_image_received = time.monotonic()
            self.last_image_warning = self.last_image_received
            self.sequence = self.processed = 0
            self.last_stamp = None
            self.history = deque(maxlen=4)
            self.starts = deque(maxlen=2)
            self.controller = WaypointController(v_max=args.v_max, w_max=args.w_max)
            self.started = False
            self.received_first_image = False
            self.started_first_inference = False
            self.subscription = None
            prefix = '/' + args.prefix.strip('/')
            self.path_pub = self.create_publisher(PathMsg, prefix + '/path', 10)
            self.waypoint_pub = self.create_publisher(Float32MultiArray, prefix + '/waypoint', 10)
            self.actions_pub = self.create_publisher(Float32MultiArray, prefix + '/sampled_actions', 10)
            self.velocity_pub = self.create_publisher(Float32MultiArray, prefix + '/velocity', 10)
            self.overlay_pub = self.create_publisher(Image, prefix + '/overlay', 10)
            self.started_pub = self.create_publisher(Empty, '/started', 10)
            # Live operation uses system time unless use_sim_time is enabled.
            self.timer = self.create_timer(1 / args.rate, self.infer)
            self.discovery = self.create_timer(1.0, self.discover)
            if args.image_topic and args.image_type:
                self.subscribe(args.image_topic, args.image_type)
            self.get_logger().info('Model ready. Waiting for live camera images or bag playback.')

        def subscribe(self, topic, kind):
            self.compressed = kind == 'compressed'
            self.subscription = self.create_subscription(
                CompressedImage if self.compressed else Image, topic, self.image_callback,
                qos_profile_sensor_data)
            self.get_logger().info(f'Camera: {topic} ({kind})')

        def discover(self):
            if self.subscription is not None:
                return
            cameras = [(n, t) for n, types in self.get_topic_names_and_types() for t in types
                       if t in ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage')
                       and n != self.overlay_pub.topic_name
                       and (not args.image_topic or n == args.image_topic)]
            if len(cameras) == 1:
                n, t = cameras[0]
                self.subscribe(n, 'compressed' if t.endswith('/CompressedImage') else 'raw')
            elif len(cameras) > 1:
                self.get_logger().warning(f'Multiple cameras: {cameras}; restart with --image-topic.')

        def image_callback(self, msg):
            try:
                if self.compressed:
                    frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
                else:
                    frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                with self.lock:
                    self.sequence += 1
                    self.latest = (self.sequence, frame.copy(), msg.header)
                    self.last_image_received = time.monotonic()
                if not self.received_first_image:
                    self.received_first_image = True
                    self.get_logger().info('Received first camera image.')
            except Exception as exc:
                self.get_logger().error(f'Image conversion failed: {exc}')

        def infer(self):
            with self.lock:
                observation = self.latest
                image_age = time.monotonic() - self.last_image_received
            if observation is None or observation[0] == self.processed:
                now = time.monotonic()
                if image_age >= 5 and now - self.last_image_warning >= 5:
                    self.last_image_warning = now
                    topic = self.subscription.topic_name if self.subscription else args.image_topic
                    self.get_logger().warning(
                        f'No new camera image for {image_age:.1f}s on {topic}. '
                        'Topic discovery does not guarantee image delivery. '
                        'For RealSense, try --image-topic /camera/camera/color/image_raw/compressed '
                        '--image-type compressed.')
                return
            sequence, frame, header = observation
            self.processed = sequence
            if not self.started_first_inference:
                self.started_first_inference = True
                self.get_logger().info('Starting first TIC-VLA inference.')
            stamp = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
            if self.last_stamp is not None and stamp < self.last_stamp:
                # Bag loop/seek: discard cached model state and controller history.
                self.model.reset_episode_state()
                self.history.clear()
                self.starts.clear()
                self.controller.reset()
            self.last_stamp = stamp
            file = self.temp_dir / f'{sequence:012d}.jpg'
            save_temp_frame(frame, file)
            self.history.append(str(file))
            images = [self.history[0]] * (4 - len(self.history)) + list(self.history)
            delay = (stamp - self.starts[0]) / 1e9 if self.starts else 0.0
            start = time.perf_counter()
            with torch.inference_mode():
                response, tensor, generation_stamp, available, _ = self.model.predict_async(
                    image_paths=images, delayed_image_paths=list(images),
                    instruction=args.instruction, robot_state=torch.zeros(6),
                    current_step=stamp, time_delay=max(0.0, delay), robot_type=args.robot_type)
            if generation_stamp is not None:
                self.starts.append(generation_stamp)
            trajectory = tensor.detach().float().cpu().numpy()
            if trajectory.ndim != 3 or trajectory.shape[0] != 1 or trajectory.shape[1] < 1 or trajectory.shape[2] not in (2, 3):
                raise ValueError(f'Unexpected trajectory shape: {trajectory.shape}')
            xy = trajectory[0, :, :2] * args.scale
            if not np.isfinite(xy).all():
                raise ValueError('Non-finite predicted waypoints')
            v, omega = self.controller(xy)
            elapsed = time.perf_counter() - start
            path = PathMsg()
            path.header.stamp = header.stamp
            path.header.frame_id = args.frame_id
            for x, y in xy:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x, pose.pose.position.y = float(x), float(y)
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            self.path_pub.publish(path)
            # Explicit output contracts: final [x,y], flattened Nx2 path, [v,omega].
            for publisher, values in ((self.waypoint_pub, xy[-1]),
                                      (self.actions_pub, xy.flatten()),
                                      (self.velocity_pub, [v, omega])):
                msg = Float32MultiArray()
                msg.data = [float(x) for x in values]
                publisher.publish(msg)
            overlay = draw_waypoints_simple(frame, xy)
            for y, text in ((40, f'v: {v:.2f} m/s'), (75, f'omega: {omega:.2f} rad/s'),
                            (110, f'inference: {elapsed:.2f}s')):
                cv2.putText(overlay, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
            msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            msg.header = header
            self.overlay_pub.publish(msg)
            if not self.started:
                self.started_pub.publish(Empty())
                self.started = True
            if args.show:
                cv2.imshow('TIC-VLA ROS', overlay)
                cv2.waitKey(1)
            self.get_logger().info(f'{elapsed:.3f}s, delay={delay:.3f}s, cache={available}, v={v:.3f}, omega={omega:.3f}')
            if response:
                self.get_logger().info(str(response))

    rclpy.init(args=ros_args)
    model = node = executor = None
    with tempfile.TemporaryDirectory(prefix='ticvla-ros-') as temp_dir:
        try:
            model = load_ticvla(args.checkpoint, args.base_model, args.device)
            node = TICVLANode(model, temp_dir)
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            # Stop inference callbacks before shutting down the model's VLM worker;
            # retain temporary image files until both have finished reading them.
            try:
                if executor is not None:
                    executor.shutdown()
                if model is not None:
                    model.cleanup()
            finally:
                if node is not None:
                    node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                if args.show:
                    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
