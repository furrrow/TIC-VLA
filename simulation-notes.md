# Simulation Migration Plan

This plan moves TIC-VLA simulation from an Isaac Sim in-process behavior script to a two-process ROS 2 architecture:

```text
machine A, now or later:
  Isaac Sim process
    publishes observations over ROS 2
    subscribes to velocity commands over ROS 2

machine B, now or later:
  infer_ros.py process
    subscribes to observations over ROS 2
    runs TIC-VLA inference
    publishes (v, w) as geometry_msgs/Twist over ROS 2
```

The immediate goal is to run Isaac Sim and inference as separate local processes. The later goal is to run them on separate machines without changing the policy/simulator contract.

## Current State

The benchmark path is still tightly coupled:

```text
DynaNav/run_benchmark.sh
  -> Isaac Sim Python
  -> DynaNav/benchmark.py
  -> per-episode SimulationApp child process
  -> DynaNav/behavior/nova_carter_test_ticvla.py
  -> TICVLA.predict_async()
  -> predicted waypoints
  -> pure-pursuit (v, w)
  -> Isaac wheel controller
```

`nova_carter_test_ticvla.py` currently owns too many responsibilities:

- camera capture from Isaac Replicator annotators
- robot pose and velocity reads from Isaac robot APIs
- TIC-VLA model loading and inference
- waypoint-to-command conversion
- command smoothing and backup behavior
- direct wheel action application through Isaac APIs

`infer_ros.py` already performs the inference-side observation path outside Isaac. It subscribes to image and odometry topics, builds the TIC-VLA model inputs, runs `model.predict_async()`, and publishes:

- `nav_msgs/Path` on `policy_path_topic`
- selected waypoint as `std_msgs/Float32MultiArray` on `waypoint_topic`
- flattened sampled action path on `sampled_actions_topic`
- optional overlay image on `overlay_topic`

It does not currently publish `geometry_msgs/Twist` directly. Today, `(v, w)` is normally produced by a separate controller such as `custom_utils/custom_utils/kinematic_controller_naive.py`, which subscribes to a waypoint and publishes `vel_topic`.

## Target Boundary

The process boundary should be `cmd_vel`, not Isaac wheel actions and not internal waypoint tensors.

Isaac Sim should behave like a ROS robot backend:

```text
Isaac ROS adapter
  publishes:
    /ticvla/camera/front/image_raw or /ticvla/camera/front/image_raw/compressed
    /ticvla/odom
    /ticvla/clock, if use_sim_time is enabled
    optional debug topics for benchmark state

  subscribes:
    /ticvla/cmd_vel

  owns:
    Isaac scene loading
    physics stepping
    rendering
    camera extraction
    odometry extraction
    Twist-to-wheel-control conversion
    benchmark metrics
```

`infer_ros.py` should own policy and command generation:

```text
infer_ros.py
  subscribes:
    image topic
    odom topic

  owns:
    TIC-VLA model loading
    image history
    async VLM/action inference
    waypoint-to-(v, w) conversion
    velocity limits and smoothing

  publishes:
    /ticvla/cmd_vel as geometry_msgs/Twist
    existing debug path/waypoint/overlay topics
```

The ROS contract between processes should be stable enough that Isaac can later move to a different machine:

```text
sensor_msgs/Image or sensor_msgs/CompressedImage
nav_msgs/Odometry
geometry_msgs/Twist
```

Avoid passing Isaac-specific prim paths, Replicator objects, USD handles, or Python objects across this boundary.

## Phase 1: Add an Isaac ROS Adapter

Create a simulator-side adapter whose only learned-policy interface is ROS 2. This can initially live as a new behavior script or as a helper used by the existing Nova Carter behavior script:

```text
DynaNav/behavior/nova_carter_ros_bridge.py
```

Responsibilities:

1. Initialize `rclpy` inside the Isaac behavior script or inside the benchmark child process.
2. Publish front camera frames from the same Isaac camera currently used by `nova_carter_test_ticvla.py`.
3. Publish odometry derived from Isaac robot pose and velocity.
4. Subscribe to `geometry_msgs/Twist` on `/ticvla/cmd_vel`.
5. Convert the latest Twist into the same differential wheel controller call used today.
6. Stop the robot if no command has arrived within a short watchdog timeout.

Suggested simulator topics:

```yaml
image_topic: /ticvla/camera/front/image_raw/compressed
odom_topic: /ticvla/odom
vel_topic: /ticvla/cmd_vel
policy_path_topic: /ticvla/policy_path
waypoint_topic: /ticvla/waypoint
sampled_actions_topic: /ticvla/sampled_actions
overlay_topic: /ticvla/path_overlay
```

Odometry requirements:

- `header.stamp`: use simulation time if `/clock` is published; otherwise use node time consistently.
- `header.frame_id`: `odom`
- `child_frame_id`: `base_link`
- `pose.pose.position`: Isaac robot world position in ROS coordinates.
- `pose.pose.orientation`: robot orientation as ROS quaternion.
- `twist.twist.linear`: robot-frame linear velocity if possible, because `infer_ros.py` currently treats odometry twist as base-frame velocity.
- `twist.twist.angular.z`: yaw rate in the robot/base frame.

Command application requirements:

- Interpret `Twist.linear.x` as forward velocity `v`.
- Interpret `Twist.angular.z` as yaw velocity `w`.
- Clamp to the robot's configured limits before applying wheel actions.
- Apply zero command when the command watchdog expires.
- Keep Isaac collision, goal-distance, and path-length metrics in `benchmark.py`.

## Phase 2: Remove Model Inference From Isaac

After the adapter publishes image and odometry and consumes `cmd_vel`, stop attaching `nova_carter_test_ticvla.py` for ROS-separated runs.

Add a benchmark mode such as:

```yaml
navigation_method: ticvla_ros
ros:
  image_topic: /ticvla/camera/front/image_raw/compressed
  odom_topic: /ticvla/odom
  vel_topic: /ticvla/cmd_vel
  command_timeout_s: 0.5
```

`benchmark.py` should then attach the ROS bridge behavior instead of the full TIC-VLA behavior:

```text
ticvla          -> current in-process behavior, kept as baseline
ticvla_ros      -> Isaac ROS bridge only, inference external
```

Keep the old in-process mode until the ROS path matches baseline behavior on a small set of episodes. It is useful as an oracle while debugging frame conventions, timestamps, and controller gains.

## Phase 3: Publish (v, w) From infer_ros.py

Change `infer_ros.py` from "waypoint inference plus external controller" to "policy plus direct velocity command".

Add arguments:

```text
--cmd-vel-topic /ticvla/cmd_vel
--publish-cmd-vel
--max-v
--max-w
--cmd-timeout
--lookahead-distance
--angular-gain
--yaw-filter-alpha
```

Load defaults from `configs/robot.yaml`:

```yaml
sim_nova:
  max_v: 0.7
  max_w: 1.0
  image_topic: /ticvla/camera/front/image_raw/compressed
  odom_topic: /ticvla/odom
  vel_topic: /ticvla/cmd_vel
  policy_path_topic: /ticvla/policy_path
  waypoint_topic: /ticvla/waypoint
  sampled_actions_topic: /ticvla/sampled_actions
  overlay_topic: /ticvla/path_overlay
```

Reuse the pure-pursuit logic currently embedded in `DynaNav/behavior/nova_carter_test_ticvla.py`:

```text
waypoints
  -> choose lookahead point by arc length
  -> yaw error
  -> curvature
  -> v_cmd
  -> w_cmd
  -> clamp
  -> publish Twist(linear.x=v_cmd, angular.z=w_cmd)
```

Keep the existing debug publishers. They are useful for validation, but they should no longer be required for control.

Implementation notes:

- Add `from geometry_msgs.msg import Twist`.
- Create a `cmd_vel_pub` publisher when direct command mode is enabled.
- Move waypoint-to-command conversion into a small helper function so it can be tested without ROS.
- Publish zero Twist on shutdown.
- Publish zero Twist if waypoints are invalid or inference has not produced a fresh result within the timeout.
- Keep the selected waypoint publisher for visualization and compatibility during the transition.

## Phase 4: Validate Locally With Two Processes

Run Isaac and inference as separate local processes before attempting two machines.

Terminal 1:

```bash
source .env.testing
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_single.yaml
```

Terminal 2:

```bash
python infer_ros.py \
  --robot sim_nova \
  --image-type compressed \
  --publish-cmd-vel \
  --cmd-vel-topic /ticvla/cmd_vel \
  --checkpoint checkpoints/TIC-VLA-model.ckpt \
  --base-model models/InternVL3-1B \
  --instruction "Move forward safely and avoid obstacles."
```

Basic ROS checks:

```bash
ros2 topic list
ros2 topic hz /ticvla/camera/front/image_raw/compressed
ros2 topic hz /ticvla/odom
ros2 topic echo /ticvla/cmd_vel
ros2 topic echo /ticvla/policy_path --once
```

Acceptance criteria:

- Isaac starts and runs without loading TIC-VLA model weights.
- `infer_ros.py` starts and receives both image and odometry.
- `infer_ros.py` publishes nonzero `Twist` messages when waypoints are valid.
- Isaac applies the Twist command and the robot moves.
- Isaac stops the robot when `infer_ros.py` exits or command messages stop.
- Benchmark success, timeout, collision, and path-length metrics still write under `DynaNav/benchmark_results/`.

## Phase 5: Split Across Machines

After the two-process local path works, put Isaac and inference on separate machines by changing ROS discovery/network configuration, not application code.

Recommended first setup:

- Same ROS 2 distribution on both machines.
- Same `ROS_DOMAIN_ID`.
- Synchronized clocks using NTP or chrony.
- DDS discovery works across the network, or use a configured discovery server.
- Start with compressed image transport to reduce bandwidth.

Example environment:

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Machine A, Isaac:

```bash
source .env.testing
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_single.yaml
```

Machine B, inference:

```bash
python infer_ros.py \
  --robot sim_nova \
  --image-type compressed \
  --publish-cmd-vel \
  --cmd-vel-topic /ticvla/cmd_vel
```

Network validation:

```bash
ros2 node list
ros2 topic hz /ticvla/camera/front/image_raw/compressed
ros2 topic hz /ticvla/odom
ros2 topic hz /ticvla/cmd_vel
```

If discovery is unreliable, switch to an explicit DDS discovery server before changing application code.

## Frame and Sign Conventions

Lock these conventions early and test them with simple commands:

- Positive `Twist.linear.x`: robot drives forward.
- Positive `Twist.angular.z`: robot turns left/counterclockwise.
- TIC-VLA body-frame waypoints use `x` forward and `y` left.
- Odometry twist supplied to `infer_ros.py` should be in the robot/base frame.
- The image topic should represent the same forward camera viewpoint used during in-process Isaac inference.

Smoke tests:

```bash
ros2 topic pub /ticvla/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2}, angular: {z: 0.0}}" --rate 10

ros2 topic pub /ticvla/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.5}}" --rate 10
```

The first command should move the robot forward. The second should rotate it left.

## Risks

- Timestamp mismatch: `infer_ros.py` resets model episode state if image stamps move backward. Use a consistent time source.
- Frame mismatch: if odometry twist is world-frame instead of base-frame, model state will be wrong.
- Camera mismatch: changing the camera pose or encoding changes the model input distribution.
- Latency: moving images over the network can make async inference operate on stale observations.
- Command watchdog: too short causes stutter; too long lets stale commands drive after inference stalls.
- DDS discovery: multi-machine ROS 2 often fails because of network/firewall/discovery configuration rather than code.

## Milestones

1. Add `sim_nova` topic config to `configs/robot.yaml`.
2. Add Isaac ROS bridge that publishes image and odometry.
3. Add Isaac `cmd_vel` subscriber and command watchdog.
4. Add `ticvla_ros` benchmark mode that attaches only the ROS bridge behavior.
5. Add direct `Twist` publishing to `infer_ros.py`.
6. Validate local two-process simulation.
7. Compare ROS-separated benchmark metrics against the current in-process `ticvla` baseline.
8. Move `infer_ros.py` to a second machine and validate ROS topic rates.

## Done Definition

This migration is complete when Isaac Sim can run an episode without importing TIC-VLA model code, `infer_ros.py` can run on another host, and the only runtime control channel from inference to simulation is:

```text
geometry_msgs/Twist on /ticvla/cmd_vel
```

At that point Isaac Sim is a robot simulator process, `infer_ros.py` is the policy process, and deployment to separate machines is a ROS 2 networking concern rather than a code architecture concern.
