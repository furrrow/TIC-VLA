#!/usr/bin/env python3
"""Passive ROS command/clock recorder. Run only for an explicitly approved diagnostic."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args, ros_args = parser.parse_known_args()
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry, Path as PathMsg
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String, Float32MultiArray
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    stream = args.output.open('x', encoding='utf-8', buffering=1)
    counts = Counter()
    sequence = 0
    episode = None
    latest_clock_ns = None
    rclpy.init(args=ros_args)
    node = rclpy.create_node('command_timing_observer')
    def write(event, **fields):
        nonlocal sequence
        sequence += 1
        counts[event] += 1
        row = dict(event=event, sequence=sequence, host=socket.gethostname(), pid=os.getpid(),
                   monotonic_ns=time.monotonic_ns(), system_time_ns=time.time_ns(),
                   ros_now_ns=node.get_clock().now().nanoseconds,
                   latest_clock_ns=latest_clock_ns, episode_id=episode, **fields)
        stream.write(json.dumps(row, allow_nan=False)+'\n')
    def stamp(header):
        return header.stamp.sec*1_000_000_000+header.stamp.nanosec
    def on_clock(msg):
        nonlocal latest_clock_ns
        latest_clock_ns = msg.clock.sec*1_000_000_000+msg.clock.nanosec
        write('clock', clock_ns=latest_clock_ns)
    def on_episode(msg):
        nonlocal episode
        value = json.loads(msg.data)
        episode = value.get('episode_id')
        write('episode', metadata=value)
    def on_odom(msg):
        p = msg.pose.pose.position
        v = msg.twist.twist
        write('odom', stamp_ns=stamp(msg.header), position=[p.x,p.y,p.z],
              linear_x=v.linear.x, angular_z=v.angular.z)
    qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    episode_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
    subscriptions = [
        node.create_subscription(Clock,'/clock',on_clock,qos),
        node.create_subscription(String,'/isaacsim/episode',on_episode,episode_qos),
        node.create_subscription(Odometry,'/isaacsim/odom',on_odom,qos),
        node.create_subscription(Twist,'/isaacsim/cmd_vel',
            lambda m:write('cmd_vel',linear_x=m.linear.x,angular_z=m.angular.z),qos),
        node.create_subscription(Float32MultiArray,'/sim_nova/waypoint',
            lambda m:write('policy_waypoint',data=list(m.data)),qos),
        node.create_subscription(Float32MultiArray,'/sim_nova/steered_waypoint',
            lambda m:write('steered_waypoint',data=list(m.data)),qos),
        node.create_subscription(PathMsg,'/sim_nova/policy_path',
            lambda m:write('policy_path',stamp_ns=stamp(m.header),poses=len(m.poses)),qos),
    ]
    write('observer_start', boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          note='Receipt observations; BEST_EFFORT can lose messages; unstamped Twist has no sender ID.')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            write('observer_stop',counts=dict(counts))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        stream.close()


if __name__ == '__main__':
    main()
