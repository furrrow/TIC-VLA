# ROS Simulation Instructions

The DynaNav ROS benchmark runs Isaac Sim on the simulator machine and keeps inference and control on the local machine. Both machines use ROS 2 Humble and `ROS_DOMAIN_ID=42`.

The benchmark starts a fresh Isaac Sim child for every episode. The local ROS processes remain alive across episodes and reset from the transient-local `/isaacsim/episode` message.

## Pipeline modes

Normal mode sends the TIC-VLA path directly to the waypoint controller:

```text
Isaac camera + odometry
  -> infer_ros.py
  -> selects the configured lookahead from /sim_nova/policy_path
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

## Manual operation

### Four-panel steered run

Use four terminals or tmux panes when you want to run every component manually.
Panels 2–4 run on the local inference computer; panel 1 runs Isaac remotely.
Start panels 2, 3, and 4 first, wait until all three nodes are ready, and only
then start panel 1. Run exactly one VLA inference command and use the controller
from that same VLA repository.

All three local panels use the same ROS environment:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1
```

#### Panel 1 — remote Isaac Sim

Do not source `/opt/ros/humble/setup.bash` on the Isaac machine; `run_sim.bash`
loads Isaac Sim's bundled ROS environment from `.env.testing`.

```bash
ssh -t isaac
cd /home/isaac/Projects/TIC-VLA

findmnt /media/isaac/HardDisk
DISPLAY=:1 ./run_sim.bash \
  --scenario DynaNav/configs/steernav_scenarios/scenario3_southbound_hard_20trials.yaml \
  --inference-method TIC-VLA \
  -- \
  --run_id manual_ticvla_steered \
  --experiment_mode steered \
  --show-child-output
```

Change `--inference-method` and the run ID to `OmniVLA` or `NaVILA` when using
one of those models. `--show-child-output` is optional; without it, Isaac output
goes to the per-episode log and the parent still surfaces detected Python/native
crashes in the terminal.

#### Panel 2 — one VLA inference node

TIC-VLA:

```bash
cd /home/jim/Projects/TIC-VLA
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1

uv run infer_ros.py \
  --robot sim_nova \
  --config /home/jim/Projects/TIC-VLA/configs/robot.yaml \
  --ros-args -p use_sim_time:=true
```

OmniVLA:

```bash
cd /home/jim/Projects/OmniVLA
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1

uv run python inference/omnivla_ros.py \
  --robot sim_nova \
  --config /home/jim/Projects/OmniVLA/config/robot.yaml \
  --ros-args -p use_sim_time:=true
```

NaVILA:

```bash
cd /home/jim/Projects/NaVILA
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1

uv run python ros_inference.py \
  --robot sim_nova \
  --config /home/jim/Projects/NaVILA/config/robot.yaml \
  --ros-args -p use_sim_time:=true
```

Wait for the selected inference node to report that its model is ready. It will
then wait for `/isaacsim/episode`, camera observations, and odometry.

#### Panel 3 — episode-aware waypoint controller

Run this from the same repository selected in panel 2. Set `MODEL_ROOT` to one
of `/home/jim/Projects/TIC-VLA`, `/home/jim/Projects/OmniVLA`, or
`/home/jim/Projects/NaVILA`. TIC-VLA uses `configs/robot.yaml`; OmniVLA and
NaVILA use `config/robot.yaml`.

```bash
MODEL_ROOT=/home/jim/Projects/TIC-VLA
ROBOT_CONFIG="${MODEL_ROOT}/configs/robot.yaml"

cd "${MODEL_ROOT}"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1

uv run python custom_utils/custom_utils/kinematic_controller_naive.py \
  --robot sim_nova \
  --config "${ROBOT_CONFIG}" \
  --dt 0.75 \
  --steer \
  --simulation-episode-control \
  --ros-args -p use_sim_time:=true
```

For OmniVLA or NaVILA, change the first two assignments, for example:

```bash
MODEL_ROOT=/home/jim/Projects/OmniVLA
ROBOT_CONFIG="${MODEL_ROOT}/config/robot.yaml"
```

`--steer` is required for the four-panel pipeline. It makes the controller use
`/sim_nova/steered_waypoint`; without it, the controller bypasses SteerNav and
uses the VLA's baseline `/sim_nova/waypoint`.

#### Panel 4 — SteerNav

```bash
cd /home/jim/Projects/steernav
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1

uv run steernav/ros_inference.py \
  --robot sim_nova \
  --time-look-ahead 1.0 \
  --object-velocity-source sim-ground-truth \
  --ros-args -p use_sim_time:=true
```

Add `--fix-end` to keep the final VLA waypoint fixed during trajectory
optimization. Add `--no-velocity` instead of
`--object-velocity-source sim-ground-truth` for the no-velocity ablation.

Once panels 2–4 are ready, start panel 1. Do not run multiple VLA inference
nodes or controllers simultaneously because they share node names and ROS
topics.

### Bundled tmux alternative

For a simpler two-terminal manual run, let `nova_ros.bash` create the local
inference, controller, and optional SteerNav panes. In the first terminal on
this computer, start the desired local pipeline:

```bash
cd /home/jim/Projects/TIC-VLA
./nova_ros.bash normal
# Or, for the SteerNav pipeline:
./nova_ros.bash steered
# Or, for SteerNav without object velocity compensation:
./nova_ros.bash no-velocity
```

Leave that terminal running. `nova_ros.bash` opens a tmux session containing
the inference and controller processes (plus SteerNav in either steering mode). Wait
until the local processes are ready before starting the simulator.

Each pane is also saved under `logs/nova_ros/<TMUX_SESSION>/` as
`inference.log`, `controller.log`, and, in either steering mode, `steernav.log`. To
copy visible terminal text, hold Shift while selecting with the mouse. For
older output, press `Ctrl-b`, then `[`, scroll or select in tmux copy mode, and
press Enter to copy the selection. Press `q` to leave copy mode without
copying.

In a second terminal, SSH into the simulator machine and run its simulation
wrapper:

```bash
ssh isaac
cd /home/isaac/Projects/TIC-VLA
./run_sim.bash
```

The wrapper defaults to `DISPLAY=:1`, which is the simulator machine's local
desktop, so the Isaac Sim window appears on its monitor. If that machine's X
session uses another display, override it when launching:

```bash
DISPLAY=:0 ./run_sim.bash
```

`run_sim.bash` sources `.env.testing` and defaults to
`DynaNav/configs/icra_scenarios_v4/scenario3_5.yaml`. Select another scenario
file with `--scenario`; pass benchmark options such as a run ID after `--`:

```bash
./run_sim.bash \
  --scenario DynaNav/configs/icra_scenarios_v4/scenario1_5.yaml \
  -- \
  --run_id manual_scenario1_normal
```

When the simulation finishes, return to the local `nova_ros.bash` terminal,
detach with `Ctrl-b`, then `d`, and stop the session using the
`TMUX_SESSION=<name>` printed at startup:

```bash
tmux kill-session -t <name>
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

Do not run these local pipelines simultaneously because they use the same ROS topics and node names.

## Automated two-machine comparison

Use `run_nova_experiments.bash` from the local TIC-VLA repository to run normal and steered experiments sequentially on the same scenario:

```bash
cd /home/jim/Projects/TIC-VLA
BATCH_ID=scenario1_comparison_v1 ./run_nova_experiments.bash
```

The default scenario is `DynaNav/configs/icra_scenarios_v4/scenario1_5.yaml`.

Use `--mode` to run only one pipeline instead of the default two-mode comparison:

```bash
BATCH_ID=scenario1_normal_v1 ./run_nova_experiments.bash --mode normal
BATCH_ID=scenario1_steered_v1 ./run_nova_experiments.bash --mode steered
BATCH_ID=scenario1_no_velocity_v1 ./run_nova_experiments.bash --mode no-velocity
```

Use `--mode ablation` to run `normal`, `steered`, and `no-velocity` sequentially:

```bash
BATCH_ID=scenario1_ablation_v1 ./run_nova_experiments.bash --mode ablation
```

The accepted values are `normal`, `steered`, `no-velocity`, `both`, and
`ablation`. You can combine the option with a scenario path:

```bash
BATCH_ID=scenario2_steered_v1 ./run_nova_experiments.bash \
  --mode steered \
  --scenario DynaNav/configs/icra_scenarios_v4/scenario2_5.yaml
```

The orchestrator performs these steps:

1. Ask the remote `run_sim.bash --check` command to validate the scenario, display, benchmark launcher, environment and Isaac Sim Python installation.
2. Start `nova_ros.bash normal` in a named local tmux session.
3. Wait until TIC-VLA reports `Model ready` and the controller reports that it registered.
4. Add a pane that runs the five-episode scenario over SSH with run ID `<BATCH_ID>_normal`, then attach your terminal to the session.
5. Stop the tracked remote simulator process group and the local tmux session created for the normal run.
6. Start `nova_ros.bash steered` in a new local tmux session.
7. Wait for TIC-VLA, SteerNav, and the controller to report readiness.
8. Add the remote simulator-status pane, attach your terminal, and run the same scenario with run ID `<BATCH_ID>_steered`.
9. Stop the tracked remote simulator process group and the steered tmux session after the benchmark finishes.
10. In `ablation` mode, repeat the steering run with SteerNav's `--no-velocity` option and run ID `<BATCH_ID>_no-velocity`.

The runs are sequential. This prevents their ROS nodes from sharing topics and ensures all modes use the same scenario geometry and pedestrian idle times.

The normal session shows three tiled panes: TIC-VLA inference, controller output, and remote Isaac benchmark status. The steered session shows four panes by adding SteerNav output. Although `nova_ros.bash` initially creates the sessions in detached mode for setup, `run_nova_experiments.bash` attaches your terminal after readiness and remote launch. Run the orchestrator outside another tmux client. Each session closes three seconds after its remote benchmark finishes, then the orchestrator proceeds to the next mode.

If the orchestrator is interrupted, it asks the remote `run_sim.bash` supervisor
to stop the benchmark process group. The supervisor sends `SIGINT` first so the
episode can close `SimulationApp`, then uses `SIGTERM` and `SIGKILL` only if the
process group remains alive.

Choose a unique, descriptive batch ID for every comparison:

```bash
BATCH_ID=icra_v4_s1_checkpoint76000_r1 ./run_nova_experiments.bash
```

To select another scenario, pass its repository-relative path:

```bash
BATCH_ID=icra_v4_s2_checkpoint76000_r1 \
  ./run_nova_experiments.bash \
  --scenario DynaNav/configs/icra_scenarios_v4/scenario2_5.yaml
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
| `REMOTE_DISPLAY` | `:1` |
| `STEERNAV_ROOT` | `/home/jim/Projects/steernav` |
| `READY_TIMEOUT_SECONDS` | `600` |
| `BATCH_ID` | Current timestamp |

The orchestrator calls `run_sim.bash` on the simulator with the selected
scenario, run ID and a tracked job ID. `run_sim.bash` delegates to
`DynaNav/run_benchmark.sh` and owns cleanup of the benchmark and episode child
processes.

The benchmark uses `headless: false`. The orchestrator exports `DISPLAY=:1` by default and verifies the corresponding X socket before loading local models. The Isaac window appears on the simulator machine's graphical desktop, not on the local machine running the SSH command. Override `REMOTE_DISPLAY` if the simulator desktop uses another display:

```bash
REMOTE_DISPLAY=:0 BATCH_ID=scenario1_comparison_v1 \
  ./run_nova_experiments.bash
```

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
| TIC-VLA to normal controller | `/sim_nova/waypoint` | `std_msgs/Float32MultiArray` |
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
