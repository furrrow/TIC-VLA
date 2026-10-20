# DynaNav

DynaNav is the Isaac Sim benchmark environment used by TIC-VLA. It contains the benchmark runner, benchmark configs, TIC-VLA behavior scripts, and local scene assets. 

## Setup

Install Isaac Sim and the DynaNav Python dependencies from the repository root first:

```bash
source .env.testing
"${ISAAC_SIM_PYTHON}" -m pip install -e .
"${ISAAC_SIM_PYTHON}" -m pip install -r requirements-test.txt
```

Edit `.env.testing` before running so it points to your Isaac Sim install, model, and checkpoint.

## Run

Run the smoke benchmark from the repository root:

```bash
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_example.yaml
# Full benchmark set:
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_full.yaml
```

The shell launcher stops an episode at the first detected pedestrian collision
and records it as a failure:

```bash
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_full.yaml
```

The launcher supplies `--terminate-on-human-collision` to `benchmark.py` by
default. Direct `benchmark.py` invocations still require the option explicitly.

### Benchmark workflow

`run_benchmark.sh` starts `benchmark.py` with Isaac Sim Python. The execution path is:

```text
main() in parent process: validate config and select episodes
  └─ run_episode_in_subprocess() for each selected episode
       └─ main() in a new child process, with --child
            └─ run_child_episode(): create and close SimulationApp
                 └─ BenchmarkRunner.run_single_episode_worker()
                      └─ BenchmarkRunner.run_episode()
```

The worker collects metrics and cleans up episode resources. The child writes one JSON result; the parent reads it and saves incremental and final results under `DynaNav/benchmark_results/<run_id>/`.

The child process exits after its episode, so the next episode starts with a fresh SimulationApp. There is no separate `run_benchmark()` method in the current workflow.

### Multi-scenario Nova experiments

From the repository root, run all three five-episode scenarios sequentially with:

```bash
./run_nova_scenarios.bash scenario1_5 scenario2_5 scenario3_5
```

The default mode is `ablation`, so every scenario runs `normal`, `steered`, and
`no-velocity`. Pass `--mode` to select a different mode. The wrapper uses one
timestamp as its batch prefix and adds the scenario name. For example, a prefix
of `20261018_193500` creates these local orchestration directories:

```text
logs/orchestration/20261018_193500_scenario1_5/
logs/orchestration/20261018_193500_scenario2_5/
logs/orchestration/20261018_193500_scenario3_5/
```

Remote benchmark results add the mode to that name, for example:

```text
DynaNav/benchmark_results/20261018_193500_scenario1_5_normal/
DynaNav/benchmark_results/20261018_193500_scenario1_5_steered/
DynaNav/benchmark_results/20261018_193500_scenario1_5_no-velocity/
```

Set `BATCH_ID` to choose the shared prefix:

```bash
BATCH_ID=ticvla_eval ./run_nova_scenarios.bash \
  scenario1_5 scenario2_5 scenario3_5
```

### Nova Carter ROS sensor preview

The `ticvla_ros` method publishes Nova Carter's forward camera as
`sensor_msgs/CompressedImage` on `/isaacsim/camera/front/image_raw/compressed` and
base-frame velocity and pose as `nav_msgs/Odometry` on `/isaacsim/odom`. It
publishes the same Isaac simulation timestamp on `/clock`; image and odometry
messages produced by one simulation update carry identical timestamps. ROS
consumers should set `use_sim_time:=true`. Sensor timestamps can jump backward
when a new episode starts, so consumers should clear timestamped history on the
transient-local `/isaacsim/episode` reset signal or when they detect a backward
timestamp jump.
The bridge subscribes to `geometry_msgs/Twist` on `/isaacsim/cmd_vel` and applies its
`linear.x` and `angular.z` to the wheels. Commands are limited using `max_v` and
`max_w` from the `sim_nova` profile in `configs/robot.yaml`, and the robot stops
if no command arrives for 0.5 seconds. Set
`TICVLA_CMD_VEL_TOPIC` in the Isaac terminal to override the command topic.
Spot is not supported in this preview.

At episode start, the bridge publishes a reliable, transient-local
`std_msgs/String` on `/isaacsim/episode`. Its `data` is JSON with
`episode_id` (`run_id:episode_name:child_pid`), `run_id`, `episode_name`,
`instruction`, `robot_type` (`nova_carter`), and `state` (`started`). Subscribers
can use the ID to reset state when a new child starts. The benchmark closes
the bridge and stops its wheels when the episode exits.
With `--robot sim_nova`, `infer_ros.py` subscribes to this topic, waits for an
episode before inference, uses its instruction, and resets model and observation
history on each new episode ID. Other robot profiles continue using
`--instruction`.
For `sim_nova`, inference publishes the full predicted path on
`/sim_nova/policy_path` and selects `/sim_nova/waypoint` using the configured
`lookahead_distance`. The selection uses path arc length and is clamped to
indices 2 through `T-3` for paths of at least five points.

Isaac's ROS 2 bridge extension supplies an `rclpy` build compatible with its
Python 3.11 runtime. `run_benchmark.sh` sources `.env.testing` to use its
bundled Humble library path for the Isaac ROS bridge.
Run a Nova Carter episode with:

```bash
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_single.yaml
```

In a separate terminal, source the system ROS installation and inspect the streams:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
ros2 node list
ros2 topic hz /isaacsim/camera/front/image_raw/compressed
ros2 topic hz /isaacsim/odom
ros2 topic echo /clock --once
ros2 topic echo /isaacsim/odom --once
ros2 topic echo /isaacsim/episode std_msgs/msg/String --qos-durability transient_local --once
```

To test wheel control during an episode, run in the ROS terminal:

```bash
ros2 topic pub --rate 10 /isaacsim/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.1}, angular: {z: 0.0}}'
```

The `sim_nova` entry in `configs/robot.yaml` uses the same image, odometry,
and velocity topics for the waypoint controller.

Use the same `ROS_DOMAIN_ID` from `.env.testing` in the ROS inspection terminal.

To connect OmniVLA or another external policy to these topics, follow the
[VLA adapter guide](VLA_ADAPTER_GUIDE.md). It defines the ROS message contract,
episode reset behavior, coordinate convention, and model-specific checks.

## Data Collection

### Generate controlled Scenario 3 configs

`generate_scenario3_configs.py` creates one benchmark YAML per requested crowd
size. The stationary target is always present. Additional people are added in
this order: head-on pedestrian, first lateral crosser, second lateral crosser.

Generate 50 paired head-on timing trials for two-, three-, and four-person
conditions while holding the lateral-crossing delays fixed:

```bash
python DynaNav/generate_scenario3_configs.py \
  --people 2,3,4 \
  --head-on-delays 0.2:1.5:50 \
  --crosser-a-delays 4.302 \
  --crosser-b-delays 5.349
```

Delay schedules accept either comma-separated values or an inclusive
`start:stop:count` range. The default `product` mode generates every combination
of delays belonging to pedestrians present in that crowd condition:

```bash
python DynaNav/generate_scenario3_configs.py \
  --people 4 \
  --head-on-delays 0.2,0.8,1.4 \
  --crosser-a-delays 2.0,3.0 \
  --crosser-b-delays 4.0,5.0 \
  --output-dir DynaNav/configs/generated_scenario3
```

That example produces 12 episodes. Use `--combination-mode paired` to combine
delay lists by index instead. In paired mode each list must contain one value,
which is broadcast, or the same number of values as the other varying lists.
Pass `--force` to replace an existing generated file.

Generated files use names such as
`scenario3_people4_50trials.yaml` and can be passed directly to
`DynaNav/run_benchmark.sh`.

Data collection uses Isaac Sim Python and the configs in `configs/data_collection/`:

```bash
source .env.testing
cd DynaNav

"${ISAAC_SIM_PYTHON}" collect_data.py -c configs/data_collection/office.yaml
"${ISAAC_SIM_PYTHON}" collect_data.py -c configs/data_collection/outdoor.yaml
"${ISAAC_SIM_PYTHON}" collect_data.py -c configs/data_collection/hospital.yaml
"${ISAAC_SIM_PYTHON}" collect_data.py -c configs/data_collection/warehouse.yaml
```

Each config writes data under `${TICVLA_DYNANAV_ROOT}/output/<scene>`. Use `--save_usd` to save the generated scene, `--debug_print` for verbose Isaac logs, and `--sensor_placment_file path/to/cameras.json` when using explicit camera placements.

To convert collected camera JSON poses into a trajectory CSV:

```bash
python camera_pose_trajectory.py output/office --out_csv output/office/trajectory.csv
```
