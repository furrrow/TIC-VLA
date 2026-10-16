# Adapting a VLA to the DynaNav ROS pipeline

DynaNav can run a VLA outside Isaac Sim. Isaac publishes observations and episode
metadata over ROS 2; the VLA publishes a local path and chooses a lookahead
waypoint; and the controller converts that waypoint to wheel commands.

```text
Isaac Sim
  camera + odometry + episode metadata
              |
              v
       VLA ROS inference node
              |
   base-frame path + waypoint
              |
              v
       PD controller -> cmd_vel -> Isaac bridge
```

## Required ROS contract

Use the `sim_nova` profile in `configs/robot.yaml`. The model node subscribes to:

| Topic | Type | Meaning |
| --- | --- | --- |
| `/isaacsim/camera/front/image_raw/compressed` | `sensor_msgs/CompressedImage` | Forward RGB observation |
| `/isaacsim/odom` | `nav_msgs/Odometry` | World pose and base-frame twist |
| `/isaacsim/episode` | `std_msgs/String` | Transient-local JSON announcing a new episode |

The episode JSON contains `episode_id`, `run_id`, `episode_name`, `instruction`,
`robot_type`, and `state`. A node must wait for `state == "started"`, reject
other robot types, and reset all model, observation, and temporal state when
`episode_id` changes. Late-starting nodes receive the current episode because
the topic is transient-local. Duplicate messages for the same ID should be
ignored.

Publish the predicted trajectory on `/sim_nova/policy_path` as `nav_msgs/Path`
and the selected point on `/sim_nova/waypoint` as `Float32MultiArray`:

- Set `header.frame_id` to `base_link`.
- Express every point in meters, with `x` forward and `y` left.
- Reject empty, malformed, or non-finite model output.
- Preserve the input image timestamp when practical.

The inference adapter selects the first point at the configured
`lookahead_distance` along the path. For paths with at least five points, it
clamps the selection to indices `2` through `T-3`. The external PD controller
subscribes to that waypoint and publishes `/isaacsim/cmd_vel`; the simulator
bridge applies the configured velocity limits and stops the robot if commands
are stale for 0.5 seconds.

Use an episode-aware controller for a process that stays alive across benchmark
children. It must publish zero velocity before the first episode, clear its
cached waypoint on every new episode ID, wait for a fresh waypoint before moving,
and publish zero again when the waypoint becomes stale. In this repository,
`kinematic_controller_naive.py --simulation-episode-control` implements that
behavior.

## Model adapter checklist

Keep model-specific loading and preprocessing in the model repository. Adapt its
ROS entry point using `infer_ros.py` as the lifecycle reference:

1. Load the model once, before spinning the node.
   Resolve checkpoints, configs, and assets from `Path(__file__)` or explicit
   arguments so launching from another working directory does not change them.
2. Decode compressed images as BGR, then convert to the color order and size the
   model expects.
3. Cache callbacks under a lock and infer only after an episode, image, and
   odometry are available. Process each camera frame at most once.
4. Run inference outside the callback lock. If a new episode arrives during a
   slow inference call, discard the old result.
5. Reset recurrent state, image history, cached goals, counters, and outstanding
   outputs on every new episode ID. If timestamps can jump backward without a
   new episode message, treat that as a reset too.
6. Convert model actions to a finite `(T, 2+)` array in the `base_link` convention
   and meters. Document any axis swap, sign flip, normalization, or waypoint
   spacing in the adapter.
7. Publish `/started` only after the first valid inference result. Do not publish
   placeholder zero paths or waypoints while waiting for inputs.
8. Use a multithreaded executor or worker thread when inference can block longer
   than the sensor period, so episode and observation callbacks stay responsive.
9. Keep the model's conditioning mode explicit. Confirm that the wrapper really
   uses its modality/config argument and that the selected checkpoint was trained
   for that mode; passing an instruction is insufficient if language tokens are
   masked by the model.
10. Keep visualization optional. Camera calibration should not be required to
    publish a valid path.

Language-conditioned models should take the instruction from the episode
message. The current episode payload does not contain a goal pose, goal yaw, or
goal image. A pose- or image-goal adapter therefore requires a deliberate bridge
extension and benchmark definition for those fields; do not substitute a fixed
test goal in benchmark runs. If a model outputs velocity commands instead of a
path, add a small conversion layer or a dedicated controller while retaining
episode reset and stale-command safety.

## OmniVLA notes

OmniVLA now provides a complete reference adapter at
`/home/jim/Projects/OmniVLA/inference/omnivla_ros.py`. Its relevant choices are:

- Isaac evaluation uses OmniVLA's language-only modality (`lan_prompt=True`,
  with pose, image, and satellite goals disabled). Pose and goal-image values are
  harmless placeholders because that modality masks them.
- `run_omnivla.py` honors the passed modality. This required fixing a wrapper
  bug that previously replaced the constructor argument with `None`.
- `sim_nova` takes its instruction from `/isaacsim/episode`; real-robot profiles
  retain a fixed CLI `--instruction`.
- The node uses generation IDs to discard inference results completed after an
  episode change, processes each image sequence once, validates `(T, 2+)` output,
  preserves the camera timestamp, and publishes `/started` after its first path.
- Simulation publishes only the full policy path. The Isaac bridge chooses the
  waypoint; direct waypoint publication remains available for real-robot
  profiles.
- Repository and checkpoint paths are resolved from the source file instead of
  the launcher's working directory. The maintained wrapper is named
  `run_omnivla.py`; downstream imports no longer depend on a parallel
  `run_omnivla_modified.py`.

Two calibration questions still require empirical validation for a new OmniVLA
checkpoint. Confirm that its output uses robot-frame `x` forward and `y` left,
and confirm `metric_waypoint_spacing`. The current wrapper uses `0.38` m while
upstream OmniVLA commonly uses `0.1` m. This scale directly changes the bridge's
1 m lookahead point.

The benchmark method remains `ticvla_ros`: that name currently selects the
generic Nova Carter ROS sensor/path bridge, even when the external policy is a
different VLA.

## NaVILA notes

NaVILA keeps its original real-robot implementation in
`/home/jim/Projects/NaVILA/ros_inference.py`. That file is also the public entry
point for simulation: `--robot sim_nova` dispatches to the episode-aware backend
in `navila_ros.py` before loading legacy-only dependencies. The backend keeps the
model loaded across benchmark children, reads each language instruction from
`/isaacsim/episode`, and clears the four-frame observation history at every
episode boundary.

NaVILA emits metric commands such as `move forward 75 cm` and heading commands
such as `turn left 30 degrees`. Translation commands become evenly spaced
`base_link` path points in meters. Because the Isaac bridge consumes `(x, y)`
positions rather than heading vectors, pure turns become a short-radius arc;
`--rotation-path-radius` controls that radius. The simulation profile publishes
only the path and leaves waypoint selection to the Isaac bridge.

Run `nova_ros.bash` for an interactive normal or steered session, or
`run_nova_experiments.bash` to run both modes sequentially against the remote
simulator. Both use the episode-aware controller and the `sim_nova` profile in
NaVILA's `config/robot.yaml`.

## Launcher pattern

Keep model launchers in the model repository because each VLA has its own Python
environment. A useful setup has two scripts:

- a local tmux launcher that sources ROS, sets `ROS_DOMAIN_ID`, starts inference,
  starts the episode-aware controller, and optionally starts SteerNav;
- an experiment orchestrator that waits for local readiness, starts DynaNav over
  SSH, streams logs, and runs normal and steered modes sequentially so they never
  share live ROS topics.

Validate repository paths, scenario paths, session names, the remote X display,
and SSH access before loading the model. Use unique run IDs and keep simulator
results separate from local orchestration logs. OmniVLA's `nova_ros.bash` and
`run_nova_experiments.bash` are concrete examples.

## Run and inspect

Start the benchmark and persistent controller in their normal environments, then
run the adapted model from its own environment with the same ROS domain:

```bash
# Isaac Sim terminal
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_single.yaml

# ROS/controller terminal
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
python custom_utils/custom_utils/kinematic_controller_naive.py \
  --robot sim_nova --simulation-episode-control --dt 1.5

# Model terminal (example; use the adapter in the model repository)
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
uv run python inference/<model>_ros.py --robot sim_nova
```

Before enabling motion, inspect one episode and confirm that the camera and odom
rates are stable, the episode message is received, path points are finite and in
meters, and a forward scene produces mostly positive path `x` values:

```bash
ros2 topic echo /isaacsim/episode std_msgs/msg/String \
  --qos-durability transient_local --once
ros2 topic hz /isaacsim/camera/front/image_raw/compressed
ros2 topic echo /sim_nova/policy_path nav_msgs/msg/Path --once
ros2 topic echo /sim_nova/waypoint std_msgs/msg/Float32MultiArray --once
```
