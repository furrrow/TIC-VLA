# Simulation Migration Plan

Split benchmark execution into ROS 2 processes. The Isaac episode child hosts the **simulation node**; a separate process hosts the **inference node** (`infer_ros.py`), and a separate waypoint controller publishes velocity commands. The parent benchmark process still selects episodes, launches one fresh Isaac child per episode, and records results.

```text
benchmark parent -> one Isaac child per episode
                    simulation node: camera + odom -> ROS 2 -> inference node
                    inference node: waypoints -> controller: cmd_vel -> simulation node
                    benchmark runner: scene, humans, stepping, metrics, results
```

## Current state and constraints

Run the existing baseline with `source .env.testing` followed by `DynaNav/run_benchmark.sh DynaNav/configs/benchmark_example.yaml`. The script forces `--navigation_method ticvla`. Both `DynaNav/benchmark_config.py` and `DynaNav/benchmark.py` currently reject other methods. The Nova Carter and Spot behavior scripts load TIC-VLA and apply Isaac actions in the same process. `infer_ros.py` already subscribes to images and odometry and publishes path, waypoint, sampled-action, and overlay debug topics, but it does not publish `geometry_msgs/Twist`.

Recent benchmark changes matter for the split:

- `benchmark.py` owns the episode lifecycle, scene loading, robot spawn, 30 Hz stepping, goal/collision/path metrics, and result collection. Each child creates and closes its own `SimulationApp`; the parent writes incremental and final results under `DynaNav/benchmark_results/<run_id>/`.
- `benchmark_config.py` validates the global navigation method and episode fields. `scripted_humans` supplies deterministic human positions, headings, idle times, and routes; `physical_human_metrics` controls additional collision reporting. Both remain simulator-side episode configuration.
- `benchmark_runtime.BehaviorContext` currently passes the goal, instruction, log directory, frame count, and failure reason from the runner to an in-process behavior script. A ROS mode needs an explicit episode start/reset signal for the inference node, rather than relying on this Python registry across processes.
- `DynaNav/run_benchmark.sh` hardcodes `ticvla`; it must forward the selected method before any `ticvla_ros` example command can work.

## Node boundary and ROS contract

The simulation node runs inside the Isaac child process. It owns camera capture, odometry, conversion of `Twist` to robot actions, and a command watchdog. The benchmark runner retains episode setup, scripted or random humans, physics stepping, contact sensors, success and timeout decisions, cleanup, and result writing. The simulation node never imports TIC-VLA or loads weights.

The inference node runs in its own ROS process. It owns model loading, observation history, and inference, and publishes waypoints. A separate controller converts waypoints to `Twist` with smoothing and limits. Path, sampled-action, and overlay topics remain diagnostic.

| Direction | Topic | Type | Purpose |
| --- | --- | --- | --- |
| Simulation → inference | `/ticvla/camera/front/image_raw/compressed` | `sensor_msgs/CompressedImage` | Front camera used by the current Isaac policy |
| Simulation → inference | `/ticvla/odom` | `nav_msgs/Odometry` | Pose and robot-frame velocity |
| Simulation → inference | `/ticvla/episode` | Explicit episode metadata/reset message (choose schema during implementation) | Episode ID, instruction, robot type, start/reset state |
| Simulation → both nodes | `/clock` when using simulated time | `rosgraph_msgs/Clock` | Consistent message stamps and episode timing |
| Inference → controller | Waypoint topic (choose final name) | `std_msgs/Float32MultiArray` | Target waypoint |
| Controller → simulation | `/ticvla/cmd_vel` | `geometry_msgs/Twist` | Forward velocity and yaw rate |

Choose and document the episode message schema before implementing it. A new episode must clear image history, model state, old commands, and controller state. The instruction should come from the selected benchmark episode, not a manually copied CLI string. Use an episode ID or equivalent handshake so a long-running inference process cannot apply a prior episode's command to the next child. A new child must start with zero velocity and wait for a fresh command. Consider a stamped command or separate ready/ack mechanism if an episode ID cannot be enforced with plain `Twist`; the watchdog alone does not distinguish stale DDS samples.

Use `odom` and `base_link` frame IDs. Positive `Twist.linear.x` drives forward, positive `Twist.angular.z` turns left, and policy waypoints use body-frame x forward/y left. Supply odometry twist in the base frame. Match the baseline camera pose, resolution, and color encoding. Publish image and odometry with compatible QoS and timestamps; `infer_ros.py` currently subscribes with reliable QoS and resets model state when image stamps move backward. Decide whether all nodes use simulation time, then configure `/clock` and `use_sim_time` consistently.

## Phase 1: Extract the simulator node

Create an Isaac-side ROS node or adapter attached to the benchmark child, for example `DynaNav/behavior/nova_carter_ros_bridge.py` plus a small ROS lifecycle helper. The bridge may live in a behavior script, but the benchmark runner must create and tear down it with the episode. Do not put benchmark metrics or human generation into the bridge.

1. Capture the existing forward Replicator camera and publish compressed frames. Publish odometry from Isaac pose and velocity; convert world-frame velocity to the base frame.
2. Subscribe to `/ticvla/cmd_vel`, clamp `v` and `w` to robot limits, and apply differential wheel actions on the Isaac update thread. ROS callbacks should only store the latest command; they must not call Isaac APIs from an executor thread.
3. Spin ROS callbacks without blocking the 30 Hz benchmark loop. Use a monotonic-time watchdog to apply zero action when commands stop or the node disconnects. Clear command state on episode start, stop, and child shutdown.
4. Publish the episode metadata/reset event after scene setup and robot readiness, before accepting commands. Keep Isaac objects, USD paths, and Replicator handles inside this process.
5. Add an equivalent adapter only if Spot runs are in scope for the first migration milestone; otherwise reject `ticvla_ros` for Spot clearly. The current baseline supports both robot types, so this limitation must be explicit.

## Phase 2: Wire the benchmark mode

Add `ticvla_ros` to `benchmark_config.py` validation and the CLI validation in `benchmark.py`. In the runner's robot setup, attach the matching ROS bridge for this method and retain the current in-process scripts for `ticvla`. Pass episode metadata from the validated config, including `instruction`, through the simulator node. Extend `run_benchmark.sh` so it accepts the config's method or a CLI override without appending a conflicting hardcoded `--navigation_method ticvla`.

The runner must keep its present child-per-episode flow, scripted human placement and commands, contact sensing, 30 Hz timeout, and metrics. Close ROS subscriptions, publishers, executor, and controller resources in `_cleanup_episode_resources()` even on failed setup. Prevent commands from a prior episode being applied when a new Isaac child starts.

Configure ROS topics and watchdog in a dedicated validated ROS section or companion config; `benchmark_config.py` currently treats unknown top-level keys as episode names. Example configuration below is a **target schema**, not a working config yet:

```yaml
navigation_method: ticvla_ros
ros:
  image_topic: /ticvla/camera/front/image_raw/compressed
  odom_topic: /ticvla/odom
  cmd_vel_topic: /ticvla/cmd_vel
  episode_topic: /ticvla/episode
  command_timeout_s: 0.5
```

## Phase 3: Connect inference and the waypoint controller

Add a `sim_nova` robot profile in `configs/robot.yaml` with matching image, odometry, waypoint, debug, and command topics. Keep `infer_ros.py` publishing waypoints. The separate controller should turn those waypoints into `Twist` and publish zero for invalid or stale waypoints, episode reset, and shutdown. Keep debug publishers for comparing predicted paths with the baseline.

Subscribe to the episode event and reset `model.reset_episode_state()`, image history, generation references, and controller state before processing new observations. Bind the instruction and robot type to that episode. The existing `--instruction` flag remains useful for bags and non-benchmark robots. Make input freshness and command timeout use appropriate wall/monotonic clocks even when simulation time pauses. Ensure the inference loop or its executor can service callbacks while model inference is busy.

## Phase 4: Validate on one machine, then two

First run the simulation child and inference process on one host with a single deterministic `scripted_humans` episode. The current `benchmark_single.yaml` is a useful fixture after changing its method and after the wiring above is implemented. Confirm both ROS nodes appear in `ros2 node list`; check image and odometry rates, episode reset delivery, and `/ticvla/cmd_vel`. Test positive forward and left commands, then stop inference and verify the watchdog stops the robot. Run two consecutive episodes to catch stale model or command state. Verify the unchanged success, timeout, collision, and path-length result fields; compare trajectories and metrics against `ticvla` on the same seed and scripted humans.

Only after local validation, move the inference node to the second host. Use the same ROS 2 distribution and domain ID, synchronized host clocks, and working DDS discovery. Keep compressed images for the first network run and measure observation age and command rate under load. If DDS discovery fails, fix discovery or network configuration before changing the node contract.

## Completion criteria

The migration is complete when an Isaac child runs an episode without importing or loading TIC-VLA, the separate inference node receives that episode's instruction and observations and publishes waypoints, the controller publishes `Twist`, and the simulator safely applies only current commands. Benchmark results and deterministic human scenarios must remain reproducible. These processes must work locally and across two machines without changing application code between those deployments.
