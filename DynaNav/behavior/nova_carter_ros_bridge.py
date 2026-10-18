"""Bridge Nova Carter's camera, odometry, and wheel commands to ROS 2."""

import cv2
import json
import os
import time
from pathlib import Path
import numpy as np
import carb
import omni.usd
import omni.replicator.core as rep
import yaml
from omni.kit.scripting import BehaviorScript
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.core.utils.rotations import quat_to_rot_matrix

import rclpy
from builtin_interfaces.msg import Time
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import benchmark_runtime as runtime
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


IMAGE_TOPIC = "/isaacsim/camera/front/image_raw/compressed"
ODOM_TOPIC = "/isaacsim/odom"
CMD_VEL_TOPIC = os.environ.get("TICVLA_CMD_VEL_TOPIC", "/isaacsim/cmd_vel")
EPISODE_TOPIC = "/isaacsim/episode"
COMMAND_TIMEOUT_S = 0.5
CAMERA_PATH = "/World/Robots/Nova_Carter/chassis_link/front_hawk/left/camera_left"
BASE_LINK_PATH = "/World/Robots/Nova_Carter/chassis_link"
DEFAULT_ROBOT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "robot.yaml"


def simulation_time_msg(current_time: float) -> Time:
    """Convert Isaac simulation seconds to a normalized ROS timestamp."""
    total_ns = max(0, round(float(current_time) * 1_000_000_000))
    return Time(
        sec=total_ns // 1_000_000_000,
        nanosec=total_ns % 1_000_000_000,
    )


def load_command_limits():
    config_path = Path(os.environ.get("TICVLA_ROBOT_CONFIG", DEFAULT_ROBOT_CONFIG)).expanduser()
    with config_path.open("r", encoding="utf-8") as config_file:
        robot_config = yaml.safe_load(config_file)["sim_nova"]
    max_v = float(robot_config["max_v"])
    max_w = float(robot_config["max_w"])
    if not np.isfinite([max_v, max_w]).all() or max_v < 0 or max_w < 0:
        raise ValueError(f"Invalid sim_nova command limits in {config_path}: max_v={max_v}, max_w={max_w}")
    return max_v, max_w


class NovaCarterROSBridge(BehaviorScript):
    """Isaac-thread publisher for one benchmark child and one robot."""

    def on_init(self):
        self._benchmark_context = runtime.get(str(self.prim_path))
        self._benchmark_context.ros_cleanup = self._close
        self._robot = WheeledRobot(
            prim_path=str(self.prim_path),
            name="nova_carter_ros_bridge",
            wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
            create_robot=False,
        )
        self._node = None
        self._controller = DifferentialController(
            name="nova_carter_ros_differential_controller", wheel_radius=0.152, wheel_base=0.413
        )
        self._command = (0.0, 0.0)
        self._command_time = None
        self._owns_rclpy = False
        self._render_product = None
        self._annotator = None
        self._closed = False
        self._max_linear_speed, self._max_angular_speed = load_command_limits()

    def on_play(self):
        self._benchmark_context.ros_cleanup = self._close
        self._command = (0.0, 0.0)
        self._command_time = None
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        if self._node is None:
            self._node = rclpy.create_node("ticvla_simulation")
            self._image_pub = self._node.create_publisher(CompressedImage, IMAGE_TOPIC, 10)
            self._odom_pub = self._node.create_publisher(Odometry, ODOM_TOPIC, 10)
            self._clock_pub = self._node.create_publisher(Clock, "/clock", 10)
            self._cmd_sub = self._node.create_subscription(Twist, CMD_VEL_TOPIC, self._on_command, 10)
            episode_qos = QoSProfile(depth=1)
            episode_qos.reliability = ReliabilityPolicy.RELIABLE
            episode_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self._episode_pub = self._node.create_publisher(String, EPISODE_TOPIC, episode_qos)
        if not getattr(self._robot, "_initialized", False):
            self._robot.initialize()
        if self._annotator is None:
            self._render_product = rep.create.render_product(CAMERA_PATH, resolution=(1920, 1080))
            self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._annotator.attach([self._render_product])
        self._closed = False
        episode = self._benchmark_context
        message = String()
        message.data = json.dumps({
            "episode_id": f"{episode.run_id}:{episode.episode_name}:{os.getpid()}",
            "run_id": episode.run_id,
            "episode_name": episode.episode_name,
            "instruction": episode.instruction,
            "robot_type": "nova_carter",
            "state": "started",
        })
        self._episode_pub.publish(message)

    def _on_command(self, msg: Twist):
        self._command = (
            float(np.clip(msg.linear.x, -self._max_linear_speed, self._max_linear_speed)),
            float(np.clip(msg.angular.z, -self._max_angular_speed, self._max_angular_speed)),
        )
        self._command_time = time.monotonic()

    def on_update(self, current_time: float, delta_time: float):
        if self._node is not None:
            rclpy.spin_once(self._node, timeout_sec=0.0)
        if self._closed or self._node is None:
            return
        stamp = simulation_time_msg(current_time)
        clock = Clock()
        clock.clock = stamp
        self._clock_pub.publish(clock)
        if not self._robot.is_valid():
            return
        command = self._command if self._command_time is not None and time.monotonic() - self._command_time < COMMAND_TIMEOUT_S else (0.0, 0.0)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        base_prim = stage.GetPrimAtPath(BASE_LINK_PATH)
        if not base_prim or not base_prim.IsValid():
            return
        transform = omni.usd.get_world_transform_matrix(base_prim)
        position = transform.ExtractTranslation()
        action = self._controller.forward(command=np.asarray(command, dtype=np.float32))
        self._robot.apply_wheel_actions(action)

        # Both messages describe this update and share its simulation timestamp.
        rotation = transform.ExtractRotation().GetQuat()
        q_wxyz = np.array([rotation.GetReal(), *rotation.GetImaginary()], dtype=np.float64)
        world_from_base = quat_to_rot_matrix(q_wxyz)
        linear_world = self._robot.get_linear_velocity()
        angular_world = self._robot.get_angular_velocity()
        linear_base = world_from_base.T @ np.asarray(
            linear_world if linear_world is not None else np.zeros(3), dtype=np.float64
        )
        angular_base = world_from_base.T @ np.asarray(
            angular_world if angular_world is not None else np.zeros(3), dtype=np.float64
        )

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, position)
        odom.pose.pose.orientation.w = float(q_wxyz[0])
        odom.pose.pose.orientation.x = float(q_wxyz[1])
        odom.pose.pose.orientation.y = float(q_wxyz[2])
        odom.pose.pose.orientation.z = float(q_wxyz[3])
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, linear_base)
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = map(float, angular_base)
        self._odom_pub.publish(odom)

        # Replicator rgb is RGB or RGBA. OpenCV's JPEG encoder expects BGR.
        if self._annotator is None:
            return
        frame = self._annotator.get_data()
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            carb.log_warn(f"Unexpected front camera frame shape: {frame.shape}")
            return
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR if frame.shape[2] == 4 else cv2.COLOR_RGB2BGR)
        encoded, jpeg = cv2.imencode(".jpg", bgr)
        if not encoded:
            carb.log_warn("Failed to JPEG-encode front camera frame")
            return
        image = CompressedImage()
        image.header.stamp = stamp
        image.header.frame_id = "front_camera"
        image.format = "jpeg"
        image.data = jpeg.tobytes()
        self._image_pub.publish(image)

    def on_stop(self):
        self._close()

    def on_destroy(self):
        self._close()

    def _close(self):
        if self._closed:
            return
        self._closed = True
        if self._benchmark_context.ros_cleanup == self._close:
            self._benchmark_context.ros_cleanup = None
        self._command = (0.0, 0.0)
        self._command_time = None
        try:
            if self._robot.is_valid():
                action = self._controller.forward(command=np.zeros(2, dtype=np.float32))
                self._robot.apply_wheel_actions(action)
        except Exception as exc:
            carb.log_warn(f"Failed to stop Nova Carter wheels: {exc}")
        if self._annotator is not None:
            try:
                self._annotator.detach()
            except Exception as exc:
                carb.log_warn(f"Failed to detach front camera annotator: {exc}")
            self._annotator = None
        self._render_product = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
            self._owns_rclpy = False
