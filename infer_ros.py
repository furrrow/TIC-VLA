#!/usr/bin/env python3
"""ROS 2 TIC-VLA async waypoint inference.

This is the ROS companion to ``infer_video_async.py``. It loads the async
DynaNav TIC-VLA model, subscribes to the robot camera and odometry topics from
``configs/robot.yaml`` in the same style as ``omnivla_ros.py``, and publishes
the waypoint/path topics consumed by the external PD controller.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
from pathlib import Path
import sqlite3
import tempfile
import time
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Empty, Float32MultiArray

from infer_once import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CHECKPOINT,
    DEFAULT_HISTORY_LEN,
    DEFAULT_INSTRUCTION,
    _ensure_custom_utils_on_path,
    configure_runtime_cache,
)
from infer_video_async import load_ticvla, save_frame


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "robot.yaml"


def bag_topics(path: str) -> list[tuple[str, str]]:
    bag_path = Path(path).expanduser().resolve()
    if bag_path.is_dir():
        files = sorted(list(bag_path.glob("*.db3")) + list(bag_path.glob("*.bd3")))
        if not files:
            raise ValueError(f"No SQLite .db3/.bd3 files in {bag_path}")
        bag_path = files[0]
    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag not found: {bag_path}")
    with sqlite3.connect(bag_path.as_uri() + "?mode=ro", uri=True) as db:
        return db.execute("SELECT name, type FROM topics ORDER BY name").fetchall()


def load_robot_config(config_path: Path, robot: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if robot not in config:
        raise KeyError(f"Robot {robot!r} is not present in {config_path}")
    return config, config[robot]


def resolve_config_path(value: str | None, config_path: Path) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    candidates = [config_path.parent / path, Path(__file__).resolve().parent / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--robot", default="husky", help="Robot key in configs/robot.yaml")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Robot YAML config path")
    parser.add_argument("--bag", default=None, help="SQLite ROS 2 bag/file for topic discovery; replay separately")
    parser.add_argument("--inspect-bag", action="store_true")
    parser.add_argument("--image-topic", help="Override camera topic from robot config")
    parser.add_argument("--image-type", choices=("raw", "compressed"))
    parser.add_argument("--odom-topic", help="Override odometry topic from robot config")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robot-type", default="legged robot", help="Robot description supplied to the VLM")
    parser.add_argument("-i", "--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--rate", type=float, default=None, help="Inference timer rate. Defaults to config frame_rate")
    parser.add_argument("--history-len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument(
        "--history-interval-seconds",
        type=float,
        default=3.0,
        help="Seconds between VLM context frames, matching async video's -9/-6/-3/current pattern.",
    )
    parser.add_argument("--metric-waypoint-spacing", type=float, default=1.0)
    parser.add_argument("--waypoint-idx", type=int, default=None, help="Path index published to waypoint_topic")
    parser.add_argument("--frame-id", default="base_link", help="Frame for predicted relative policy path")
    parser.add_argument("--camera-matrix", default=None, help="Calibration JSON for projected overlay")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--quiet-response", action="store_true", default=True)
    parser.add_argument("--print-response", action="store_false", dest="quiet_response")
    args, ros_args = parser.parse_known_args()

    if ros_args and ros_args[0] != "--ros-args":
        parser.error(f"Unrecognized arguments: {ros_args}")
    if args.inspect_bag and not args.bag:
        parser.error("--inspect-bag requires --bag")
    if args.history_len < 1:
        parser.error("--history-len must be >= 1")
    if not math.isfinite(args.history_interval_seconds) or args.history_interval_seconds <= 0:
        parser.error("--history-interval-seconds must be finite and > 0")
    if args.metric_waypoint_spacing <= 0:
        parser.error("--metric-waypoint-spacing must be > 0")
    if args.rate is not None and (not math.isfinite(args.rate) or args.rate <= 0):
        parser.error("--rate must be finite and > 0")

    if args.bag:
        topics = bag_topics(args.bag)
        if args.inspect_bag:
            for name, kind in topics:
                print(f"{name}\t{kind}")
            return args, ros_args
        candidates = [
            (name, kind)
            for name, kind in topics
            if kind in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
        ]
        if args.image_topic:
            candidates = [(name, kind) for name, kind in candidates if name == args.image_topic]
        if len(candidates) != 1:
            parser.error(f"Select --image-topic from bag camera topics: {candidates or topics}")
        args.image_topic, kind = candidates[0]
        actual = "compressed" if kind.endswith("/CompressedImage") else "raw"
        if args.image_type and args.image_type != actual:
            parser.error("--image-type conflicts with bag metadata")
        args.image_type = actual
    return args, ros_args


def stamp_to_ns(stamp: Any) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def sample_history_paths(
    history: deque[tuple[int, str]],
    history_len: int,
    interval_ns: int,
) -> list[str]:
    if not history:
        return []
    newest_stamp = history[-1][0]
    selected: list[str] = []
    seen: set[str] = set()
    offsets = [interval_ns * i for i in range(history_len - 1, -1, -1)]
    for offset in offsets:
        target_stamp = newest_stamp - offset
        if offset and target_stamp < history[0][0]:
            continue
        best_path = history[-1][1]
        for frame_stamp, path in reversed(history):
            if frame_stamp <= target_stamp:
                best_path = path
                break
        if best_path not in seen:
            selected.append(best_path)
            seen.add(best_path)
    return selected


def draw_waypoints_simple(frame_bgr: np.ndarray, path_xy: np.ndarray) -> np.ndarray:
    overlay = frame_bgr.copy()
    height, width = overlay.shape[:2]
    origin = np.array([width // 2, int(height * 0.82)], dtype=np.float32)
    px_per_m = min(width, height) * 0.12
    points = []
    for x, y in path_xy:
        point = origin + np.array([float(y), -float(x)], dtype=np.float32) * px_per_m
        points.append((int(point[0]), int(point[1])))
    for idx, point in enumerate(points):
        cv2.circle(overlay, point, 5, (0, 255, 0), -1)
        if idx:
            cv2.line(overlay, points[idx - 1], point, (0, 255, 0), 2)
    return overlay


class TICVLAROSNode(Node):
    def __init__(
        self,
        args: argparse.Namespace,
        config: dict[str, Any],
        robot_config: dict[str, Any],
        model: Any,
        temp_dir: str,
    ) -> None:
        super().__init__("ticvla_node")
        self.args = args
        self.model = model
        self.temp_dir = Path(temp_dir)
        self.bridge = CvBridge()
        self.lock = Lock()
        self.obs_img_bgr: np.ndarray | None = None
        self.obs_header: Any | None = None
        self.obs_sequence = 0
        self.processed_sequence = 0
        self.last_image_stamp_ns: int | None = None
        self.image_history: deque[tuple[int, str]] = deque()
        self.vlm_generation_refs: deque[dict[str, Any]] = deque(maxlen=2)
        self.current_pos = np.zeros(3, dtype=np.float64)
        self.current_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.robot_velocity_base = np.zeros(3, dtype=np.float64)
        self.robot_angular_velocity_base = np.zeros(3, dtype=np.float64)
        self.have_odom = False
        self.started_sent = False
        self.inference_count = 0
        self.inference_start_time = time.perf_counter()
        self.path_frame_id = args.frame_id

        self.rate = float(args.rate or config.get("frame_rate", 10))
        self.waypoint_idx = int(
            args.waypoint_idx
            if args.waypoint_idx is not None
            else config.get("waypoint_idx", 2)
        )
        self.history_interval_ns = int(args.history_interval_seconds * 1_000_000_000)
        self.history_window_ns = self.history_interval_ns * max(1, args.history_len)

        image_topic = args.image_topic or robot_config["image_topic"]
        odom_topic = args.odom_topic or robot_config["odom_topic"]
        self.compressed_img_topic = args.image_type == "compressed" if args.image_type else "compressed" in image_topic
        policy_path_topic = robot_config["policy_path_topic"]
        waypoint_topic = robot_config["waypoint_topic"]
        sampled_actions_topic = robot_config["sampled_actions_topic"]
        overlay_topic = robot_config["overlay_topic"]

        self.cam_matrix = None
        self.T_cam_from_base = None
        camera_matrix = args.camera_matrix or config.get("cam_matrix")
        camera_matrix_path = resolve_config_path(camera_matrix, Path(args.config).expanduser().resolve())
        if camera_matrix_path and Path(camera_matrix_path).is_file():
            _ensure_custom_utils_on_path()
            from custom_utils.io_utils import load_calibration

            self.cam_matrix, _, T_base_from_cam = load_calibration(camera_matrix_path)
            self.T_cam_from_base = np.linalg.inv(T_base_from_cam)
        elif camera_matrix:
            self.get_logger().warning(f"Calibration file not found; using schematic overlay: {camera_matrix_path}")

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.input_group = MutuallyExclusiveCallbackGroup()
        self.inference_group = MutuallyExclusiveCallbackGroup()

        msg_type = CompressedImage if self.compressed_img_topic else Image
        self.image_sub = self.create_subscription(
            msg_type,
            image_topic,
            self.img_callback_obs,
            qos_profile=reliable_qos,
            callback_group=self.input_group,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback_obs,
            qos_profile=reliable_qos,
            callback_group=self.input_group,
        )
        self.waypoint_pub = self.create_publisher(Float32MultiArray, waypoint_topic, qos_profile=reliable_qos)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, sampled_actions_topic, qos_profile=best_effort_qos)
        self.trajectory_visual_pub = self.create_publisher(Image, overlay_topic, qos_profile=reliable_qos)
        self.pub_path = self.create_publisher(PathMsg, policy_path_topic, qos_profile=reliable_qos)
        self.pub_started = self.create_publisher(Empty, "/started", 10)

        wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(
            1.0 / self.rate,
            self.run_inference_loop,
            callback_group=self.inference_group,
            clock=wall_clock,
        )

        self.get_logger().info(f"Using robot config for: {args.robot}")
        self.get_logger().info(f"IMAGE_TOPIC: {image_topic} compressed_img_topic: {self.compressed_img_topic}")
        self.get_logger().info(f"ODOM_TOPIC: {odom_topic}")
        self.get_logger().info(f"WAYPOINT_TOPIC: {waypoint_topic}")
        self.get_logger().info("Model ready. Waiting for image observations and odometry...")

    def img_callback_obs(self, msg: Image | CompressedImage) -> None:
        try:
            if self.compressed_img_topic:
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.obs_img_bgr = frame.copy()
                self.obs_header = msg.header
                self.obs_sequence += 1
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")

    def odom_callback_obs(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self.lock:
            self.current_pos[:] = [p.x, p.y, p.z]
            self.current_quat_wxyz[:] = [q.w, q.x, q.y, q.z]
            self.robot_velocity_base[:] = [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ]
            self.robot_angular_velocity_base[:] = [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ]
            self.have_odom = True

    def make_robot_pose(self, current_pos: np.ndarray, current_quat_wxyz: np.ndarray) -> dict[str, Any]:
        return {
            "position": current_pos.tolist(),
            "quaternion": current_quat_wxyz.tolist(),
            "rotation_matrix": quat_wxyz_to_matrix(current_quat_wxyz).tolist(),
        }

    def make_robot_state(
        self,
        stamp_ns: int,
        current_pos: np.ndarray,
        robot_velocity_base: np.ndarray,
        robot_angular_velocity_base: np.ndarray,
    ) -> tuple[torch.Tensor, float]:
        dx, dy, delay_time = 0.0, 0.0, 0.0
        if self.vlm_generation_refs:
            ref = self.vlm_generation_refs[0]
            ref_pos = np.asarray(ref["position"], dtype=np.float64)
            ref_quat = np.asarray(ref["quaternion"], dtype=np.float64)
            delta_world = current_pos - ref_pos
            delta_base = quat_wxyz_to_matrix(ref_quat).T @ delta_world
            dx = float(delta_base[0])
            dy = float(delta_base[1])
            delay_time = max(0.0, float(stamp_ns - ref["stamp_ns"]) / 1e9)

        robot_state = torch.tensor(
            [
                float(robot_velocity_base[0]),
                float(robot_velocity_base[1]),
                float(robot_velocity_base[2]),
                float(robot_angular_velocity_base[2]),
                dx,
                dy,
            ],
            dtype=torch.float32,
        )
        return robot_state, delay_time

    def to_path_msg(self, path_xy: np.ndarray, stamp: Any) -> PathMsg:
        msg = PathMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = self.path_frame_id
        for x, y in path_xy:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def overlay_image(self, frame_bgr: np.ndarray, path_xy: np.ndarray) -> np.ndarray:
        if self.cam_matrix is not None and self.T_cam_from_base is not None:
            _ensure_custom_utils_on_path()
            from custom_utils.io_utils import overlay_path

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb = overlay_path(path_xy, frame_rgb, self.cam_matrix, self.T_cam_from_base)
            if overlay_rgb is not None:
                return cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        return draw_waypoints_simple(frame_bgr, path_xy)

    def run_inference_loop(self) -> None:
        with self.lock:
            if self.obs_img_bgr is None:
                return
            if not self.have_odom:
                self.get_logger().info("waiting on odom")
                return
            if self.obs_sequence == self.processed_sequence:
                return

            sequence = self.obs_sequence
            frame_bgr = self.obs_img_bgr.copy()
            header = self.obs_header
            self.processed_sequence = sequence
            current_pos = self.current_pos.copy()
            current_quat_wxyz = self.current_quat_wxyz.copy()
            robot_velocity_base = self.robot_velocity_base.copy()
            robot_angular_velocity_base = self.robot_angular_velocity_base.copy()

        if header is None:
            return
        stamp_ns = stamp_to_ns(header.stamp)
        if self.last_image_stamp_ns is not None and stamp_ns < self.last_image_stamp_ns:
            self.model.reset_episode_state()
            self.image_history.clear()
            self.vlm_generation_refs.clear()
        self.last_image_stamp_ns = stamp_ns

        frame_path = self.temp_dir / f"frame_{sequence:012d}.jpg"
        save_frame(frame_bgr, frame_path)
        self.image_history.append((stamp_ns, str(frame_path)))
        while self.image_history and stamp_ns - self.image_history[0][0] > self.history_window_ns:
            self.image_history.popleft()
        image_paths = sample_history_paths(
            self.image_history,
            self.args.history_len,
            self.history_interval_ns,
        )
        if not image_paths:
            return

        robot_state, delay_time = self.make_robot_state(
            stamp_ns,
            current_pos,
            robot_velocity_base,
            robot_angular_velocity_base,
        )
        robot_pose = self.make_robot_pose(current_pos, current_quat_wxyz)
        t0 = time.perf_counter()
        with torch.inference_mode():
            response, waypoint_tensor, generation_step, cache_available, generation_pose = self.model.predict_async(
                image_paths=image_paths,
                delayed_image_paths=image_paths,
                instruction=self.args.instruction,
                robot_state=robot_state,
                current_step=sequence,
                current_robot_pose=robot_pose,
                time_delay=delay_time,
                robot_type=self.args.robot_type,
            )
        inference_seconds = time.perf_counter() - t0

        if generation_step is not None and generation_pose is not None:
            pos = np.asarray(generation_pose.get("position", current_pos), dtype=np.float64)
            quat = np.asarray(generation_pose.get("quaternion", current_quat_wxyz), dtype=np.float64)
            self.vlm_generation_refs.append(
                {
                    "step": generation_step,
                    "stamp_ns": stamp_ns,
                    "position": pos.copy(),
                    "quaternion": quat.copy(),
                }
            )

        waypoints = waypoint_tensor.detach().float().cpu().numpy()
        if waypoints.ndim != 3 or waypoints.shape[0] != 1 or waypoints.shape[2] < 2:
            raise ValueError(f"Expected waypoints shaped (1,T,2+), got {waypoints.shape}")
        path_xy = waypoints[0, :, :2] * self.args.metric_waypoint_spacing
        if not np.isfinite(path_xy).all():
            raise ValueError("Model returned non-finite waypoints.")

        chosen_idx = min(max(self.waypoint_idx, 0), len(path_xy) - 1)
        chosen_waypoint = path_xy[chosen_idx]
        self.pub_path.publish(self.to_path_msg(path_xy, header.stamp))

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.flatten().astype(float).tolist()
        self.waypoint_pub.publish(waypoint_msg)

        sampled_actions_msg = Float32MultiArray()
        sampled_actions_msg.data = path_xy.flatten().astype(float).tolist()
        self.sampled_actions_pub.publish(sampled_actions_msg)

        overlay = self.overlay_image(frame_bgr, path_xy)
        cv2.putText(
            overlay,
            f"async inference {1.0 / max(inference_seconds, 1e-6):.2f} Hz",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        out_msg.header = header
        self.trajectory_visual_pub.publish(out_msg)

        if not self.started_sent:
            self.started_sent = True
            self.pub_started.publish(Empty())
            self.get_logger().info("Published /started (once).")

        if self.args.show:
            cv2.imshow("TIC-VLA ROS", overlay)
            cv2.waitKey(1)

        self.inference_count += 1
        elapsed = time.perf_counter() - self.inference_start_time
        if elapsed >= 1.0:
            inference_rate = self.inference_count / elapsed
            self.get_logger().info(
                f"Inference rate: {inference_rate:.2f} Hz | "
                f"last={inference_seconds:.2f}s delay={delay_time:.2f}s "
                f"cache={cache_available} waypoint[{chosen_idx}]={chosen_waypoint.tolist()}"
            )
            self.inference_count = 0
            self.inference_start_time = time.perf_counter()
        if response and not self.args.quiet_response:
            self.get_logger().info(str(response))


def main() -> int:
    args, ros_args = parse_args()
    if args.inspect_bag:
        return 0

    configure_runtime_cache()
    _ensure_custom_utils_on_path()
    config_path = Path(args.config).expanduser().resolve()
    config, robot_config = load_robot_config(config_path, args.robot)

    checkpoint_path = Path(args.checkpoint).expanduser()
    base_model_path = Path(args.base_model).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path.resolve()}")
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_model_path.resolve()}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")

    rclpy.init(args=ros_args)
    model = None
    node = None
    executor = None
    with tempfile.TemporaryDirectory(prefix="ticvla-ros-") as temp_dir:
        try:
            model = load_ticvla(str(checkpoint_path), str(base_model_path), args.device)
            node = TICVLAROSNode(args, config, robot_config, model, temp_dir)
            executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(node)
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
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


if __name__ == "__main__":
    raise SystemExit(main())
