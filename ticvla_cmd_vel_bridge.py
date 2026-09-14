#!/usr/bin/env python3
"""Bridge TIC-VLA ``[v, omega]`` predictions to a ROS 2 ``Twist`` command.

The bridge publishes at a fixed rate and sends zero velocity whenever TIC-VLA
has not produced a fresh prediction within ``--timeout`` seconds. Non-zero
commands require ``--enable`` so launching the node alone cannot move a robot.

Example, after DynaNav/infer_ros.py is publishing /ticvla/velocity:
  python DynaNav/ticvla_cmd_vel_bridge.py --enable \\
    --output-topic /a200_0648/cmd_vel --v-max 0.20 --w-max 0.40
"""
from __future__ import annotations

import argparse
import math
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-topic', default='/ticvla/velocity',
                        help='Float32MultiArray with [forward_mps, yaw_rate_radps]')
    parser.add_argument('--output-topic', default='/cmd_vel',
                        help='geometry_msgs/Twist command topic')
    parser.add_argument('--rate', type=float, default=10.0,
                        help='Twist publish rate in Hz')
    parser.add_argument('--timeout', type=float, default=0.75,
                        help='Stop if no fresh TIC-VLA prediction arrives within this many seconds')
    parser.add_argument('--v-max', type=float, default=0.20,
                        help='Absolute forward-speed limit in m/s')
    parser.add_argument('--w-max', type=float, default=0.40,
                        help='Absolute yaw-rate limit in rad/s')
    parser.add_argument('--enable', action='store_true',
                        help='Permit non-zero Twist output; omit for observation-only mode')
    args, ros_args = parser.parse_known_args()
    if ros_args and ros_args[0] != '--ros-args':
        parser.error(f'Unrecognized arguments: {ros_args}')
    if not all(math.isfinite(value) and value > 0
               for value in (args.rate, args.timeout, args.v_max, args.w_max)):
        parser.error('rate, timeout, v-max, and w-max must be finite and positive')
    return args, ros_args


def main():
    args, ros_args = parse_args()
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray

    class Bridge(Node):
        def __init__(self):
            super().__init__('ticvla_cmd_vel_bridge')
            self.latest = None
            self.last_received = None
            self.last_status = None
            self.publisher = self.create_publisher(Twist, args.output_topic, 10)
            self.subscription = self.create_subscription(
                Float32MultiArray, args.input_topic, self.velocity_callback, 10)
            self.timer = self.create_timer(1.0 / args.rate, self.publish_command)
            mode = 'ENABLED' if args.enable else 'observation-only'
            self.get_logger().info(
                f'{mode}: {args.input_topic} -> {args.output_topic}; '
                f'limits v={args.v_max:.2f} m/s, omega={args.w_max:.2f} rad/s; '
                f'timeout={args.timeout:.2f}s')

        def velocity_callback(self, message):
            if len(message.data) != 2 or not all(math.isfinite(value) for value in message.data):
                self.get_logger().warning('Ignoring invalid TIC-VLA velocity; expected finite [v, omega].')
                return
            self.latest = (float(message.data[0]), float(message.data[1]))
            self.last_received = time.monotonic()

        def publish_command(self):
            command = Twist()
            now = time.monotonic()
            fresh = self.last_received is not None and now - self.last_received <= args.timeout
            if args.enable and fresh:
                v, omega = self.latest
                command.linear.x = max(-args.v_max, min(args.v_max, v))
                command.angular.z = max(-args.w_max, min(args.w_max, omega))
                status = f'commanding v={command.linear.x:.3f}, omega={command.angular.z:.3f}'
            elif args.enable:
                age = float('inf') if self.last_received is None else now - self.last_received
                status = f'stopped: stale TIC-VLA input (age={age:.2f}s)'
            else:
                status = 'observation-only: publishing zero Twist'
            self.publisher.publish(command)
            if status != self.last_status:
                self.last_status = status
                self.get_logger().info(status)

        def stop(self):
            self.publisher.publish(Twist())

    rclpy.init(args=ros_args)
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
