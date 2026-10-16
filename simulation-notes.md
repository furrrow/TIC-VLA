# ROS Simulation Instructions

The DynaNav ROS benchmark runs Isaac Sim on the simulator machine and keeps inference and control on the local machine. Both machines use ROS 2 Humble and `ROS_DOMAIN_ID=42`.

The benchmark starts a fresh Isaac Sim child for every episode. The local ROS processes remain alive across episodes and reset from the transient-local `/isaacsim/episode` message.

## Pipeline modes

Normal mode sends the TIC-VLA path directly to the waypoint controller:

```text
Isaac camera + odometry
  -> infer_ros.py
  -> /sim_nova/policy_path
  -> Isaac bridge selects an approximately 1 m waypoint
  -> /sim_nova/waypoint
  -> kinematic_controller_naive.py
  -> /isaacsim/cmd_vel
  -> Isaac
```

Steered mode sends the TIC-VLA path through SteerNav before control:

```text
Isaac camera + odometry
  -> infer_ros.py
  -> /sim_nova/policy_path
  -> SteerNav ros_inference.py
  -> /sim_nova/steered_waypoint
  -> kinematic_controller_naive.py --steer
  -> /isaacsim/cmd_vel
  -> Isaac
```

The controller uses `--simulation-episode-control` in both modes. It publishes zero velocity while waiting for an episode, clears its waypoint on a new episode, and waits for a fresh waypoint before moving. In steered mode, SteerNav also clears its image, path, detection, tracker, velocity, and timing state for every new episode ID.

## Automated two-machine comparison

Use `run_nova_experiments.bash` from the local TIC-VLA repository to run normal and steered experiments sequentially on the same scenario:

```bash
cd /home/jim/Projects/TIC-VLA
BATCH_ID=scenario1_comparison_v1 ./run_nova_experiments.bash
```

The default scenario is `DynaNav/configs/icra_scenarios_v4/scenario1_5.yaml`.

The orchestrator performs these steps:

1. Verify over `ssh isaac` that the scenario and remote benchmark launcher exist under `/home/isaac/Projects/TIC-VLA`, and confirm that `.env.testing` can be sourced with strict unset-variable checking.
2. Start `nova_ros.bash normal` in a named, detached local tmux session.
3. Wait until TIC-VLA reports `Model ready` and the controller reports that it registered.
4. Run the five-episode scenario on the simulator with run ID `<BATCH_ID>_normal`.
5. Stop only the local tmux session created for the normal run.
6. Start `nova_ros.bash steered` in a new detached tmux session.
7. Wait for TIC-VLA, SteerNav, and the controller to report readiness.
8. Run the same remote scenario with run ID `<BATCH_ID>_steered`.
9. Stop the steered tmux session after the benchmark finishes.

The two runs are sequential. This prevents their ROS nodes from sharing topics and ensures both modes use the same scenario geometry and pedestrian idle times.

Choose a unique, descriptive batch ID for every comparison:

```bash
BATCH_ID=icra_v4_s1_checkpoint76000_r1 ./run_nova_experiments.bash
```

To select another scenario, pass its repository-relative path:

```bash
BATCH_ID=icra_v4_s2_checkpoint76000_r1 \
  ./run_nova_experiments.bash \
  DynaNav/configs/icra_scenarios_v4/scenario2_5.yaml
```

The default local readiness timeout is 600 seconds. Increase it when model loading takes longer:

```bash
READY_TIMEOUT_SECONDS=900 \
BATCH_ID=icra_v4_s1_checkpoint76000_r1 \
  ./run_nova_experiments.bash
```

Deployment overrides:

| Variable | Default |
| --- | --- |
| `REMOTE_HOST` | `isaac` |
| `REMOTE_REPO` | `/home/isaac/Projects/TIC-VLA` |
| `STEERNAV_ROOT` | `/home/jim/Projects/steernav` |
| `READY_TIMEOUT_SECONDS` | `600` |
| `BATCH_ID` | Current timestamp |

The orchestrator calls `DynaNav/run_benchmark.sh` on the simulator directly so it can pass the selected scenario and run ID. It does not use the remote `run_sim.bash`, which may be hardcoded to a different scenario suite.

## Results and logs

Remote benchmark results remain on the simulator:

```text
/home/isaac/Projects/TIC-VLA/DynaNav/benchmark_results/<BATCH_ID>_normal/
/home/isaac/Projects/TIC-VLA/DynaNav/benchmark_results/<BATCH_ID>_steered/
```

The local machine records streamed remote output at:

```text
/home/jim/Projects/TIC-VLA/logs/orchestration/<BATCH_ID>/normal.log
/home/jim/Projects/TIC-VLA/logs/orchestration/<BATCH_ID>/steered.log
```

If a readiness check or remote benchmark fails, the orchestrator exits and stops its active local tmux session. It does not remove unrelated tmux sessions. It stops after a failed normal run rather than continuing with a steered result from invalid conditions.

## Manual operation

Start either pipeline locally:

```bash
./nova_ros.bash normal
./nova_ros.bash steered
```

For automation or background use, assign a session name and detach:

```bash
./nova_ros.bash normal --detach --session-name nova_manual_normal
./nova_ros.bash steered --detach --session-name nova_manual_steered
```

Inspect or stop a detached session with:

```bash
tmux attach -t nova_manual_normal
tmux kill-session -t nova_manual_normal
```

After the local processes are ready, start a scenario on the simulator:

```bash
ssh isaac
cd /home/isaac/Projects/TIC-VLA
DynaNav/run_benchmark.sh \
  DynaNav/configs/icra_scenarios_v4/scenario1_5.yaml \
  --run_id manual_scenario1_normal
```

Do not run normal and steered local pipelines simultaneously because they use the same ROS topics and node names.

## Episode instructions and reset behavior

Each scenario YAML supplies the episode instruction. `benchmark.py` copies it into the episode context, and `nova_carter_ros_bridge.py` publishes it in the JSON payload on `/isaacsim/episode`. For `--robot sim_nova`, `infer_ros.py` replaces its CLI instruction with the episode instruction before inference.

The episode payload contains:

```text
episode_id
run_id
episode_name
instruction
robot_type: nova_carter
state: started
```

The episode ID includes the run ID, episode name, and child PID. A new ID resets inference, controller, and, when enabled, SteerNav state. Duplicate episode messages are ignored.

Real-robot profiles do not subscribe to `/isaacsim/episode`. They use the `--instruction` value for the lifetime of `infer_ros.py` and start inference after camera images and odometry arrive.

## ROS topics

| Direction | Topic | Type |
| --- | --- | --- |
| Isaac to inference and SteerNav | `/isaacsim/camera/front/image_raw/compressed` | `sensor_msgs/CompressedImage` |
| Isaac to inference and SteerNav | `/isaacsim/odom` | `nav_msgs/Odometry` |
| Isaac to persistent local nodes | `/isaacsim/episode` | `std_msgs/String` containing JSON |
| TIC-VLA to Isaac/SteerNav | `/sim_nova/policy_path` | `nav_msgs/Path` |
| Isaac bridge to normal controller | `/sim_nova/waypoint` | `std_msgs/Float32MultiArray` |
| SteerNav to steered controller | `/sim_nova/steered_waypoint` | `std_msgs/Float32MultiArray` |
| Controller to Isaac | `/isaacsim/cmd_vel` | `geometry_msgs/Twist` |

Policy paths and waypoints use the robot body frame: positive x is forward and positive y is left. Positive `Twist.linear.x` drives forward, and positive `Twist.angular.z` turns left.

## Troubleshooting

Confirm that both machines use domain 42:

```bash
echo "$ROS_DOMAIN_ID"
```

Inspect local nodes and topics:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
ros2 node list
ros2 topic list
ros2 topic echo /isaacsim/episode std_msgs/msg/String \
  --qos-durability transient_local --once
```

During an orchestrated run, list and inspect the generated tmux session:

```bash
tmux ls
tmux attach -t nova_<BATCH_ID>_normal
```

Detach without stopping it with `Ctrl-b`, then `d`.

If startup times out, the orchestrator prints recent output from every pane before cleaning up. Resolve model-loading, dependency, CUDA, ROS, or SSH errors shown there. Reuse the batch ID only if no remote result directory was created; otherwise use a new batch ID to avoid mixing attempts.

## Interpretation limits

The ROS protocol is asynchronous: Isaac physics and pedestrians continue while TIC-VLA and SteerNav perform inference. The original VEGA evaluation used synchronous HTTP inference and paused progression while waiting for a response. Results from these protocols must be labeled separately even when they share scenario geometry.

The current `sim_nova` limits are 1.5 m/s linear and 1.2 rad/s angular. The local benchmark also uses its own collision/contact implementation. These runs are ROS executions of the ICRA v4 scenarios, not reproductions of VEGA scores.

Infrastructure failures currently appear as unsuccessful benchmark results. Review the logs and separate simulator, ROS, model-loading, and network failures from valid task failures before aggregating metrics.
