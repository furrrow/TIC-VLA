# ICRA v4 scenarios for the local ROS benchmark

This directory adapts the frozen VEGA-DynaNav ICRA v4 scenario geometry for this checkout's local `ticvla_ros` pipeline. Each YAML sets:

```yaml
navigation_method: ticvla_ros
```

The source scene geometry and paired pedestrian timings came from:

```text
jianyu34@nexusgamma00.umiacs.umd.edu:
/fs/nexus-projects/REAL2SIM3/project-VEGA/external-benchmarks/vega-dynanav/icra_scenarios_v4
```

## Local pipeline compatibility

The three configs are compatible with the current local benchmark configuration parser and Nova Carter ROS bridge. They use only supported fields:

- Hospital scene URL
- `start`, `start_yaw` and `goal`
- `nova_carter`
- `scripted_humans` with fixed positions, routes and initial idle times
- `physical_human_metrics`
- 40-second configured timeout
- 1.5 m success threshold

When run locally, `benchmark.py` starts a fresh Isaac Sim child for every episode. `nova_carter_ros_bridge.py` publishes the front camera, odometry and episode metadata. The long-running `infer_ros.py` process publishes a body-frame policy path, the bridge selects an approximately 1 m lookahead waypoint, and the separate ROS controller publishes `Twist` commands back to Isaac.

Start the inference and controller processes from the repository root:

```bash
./nova_ros.bash
```

Then run one scenario in another terminal:

```bash
DynaNav/run_benchmark.sh \
  DynaNav/configs/icra_scenarios_v4/scenario1_5.yaml \
  --run_id icra_v4_ros_s1
```

Use `scenario2_5.yaml` with `icra_v4_ros_s2` or `scenario3_5.yaml` with `icra_v4_ros_s3` for the other locations. A CLI `--navigation_method` override is unnecessary because these local copies already specify `ticvla_ros`.

The three scenario files are intended to be evaluated together as one 15-episode suite. For a complete run, use the combined configuration:

```bash
DynaNav/run_benchmark.sh \
  DynaNav/configs/icra_scenarios_v4/all_scenarios_15.yaml \
  --run_id icra_v4_ros_all
```

`all_scenarios_15.yaml` runs scenario 1, then scenario 2, then scenario 3, preserving the five v4 timing trials within each scenario. The individual files remain useful for incremental development, targeted reruns and diagnosing a particular interaction pattern.

Compatibility here means that the files validate and use the local runner's supported scene, robot and scripted-human features. It does not make the execution protocol identical to the original VEGA v4 evaluation:

- Local inference and control are asynchronous ROS processes. Simulation and pedestrians keep moving during inference.
- VEGA v4 used synchronous HTTP inference and waited for each policy response.
- Local `sim_nova` limits are currently 1.5 m/s and 1.2 rad/s. VEGA v4 used 1.0 m/s and 0.3 rad/s.
- The local benchmark currently uses its own collision/contact implementation rather than v4's `physx_capsules_fov_v1` measurement layer.
- These local files preserve v4 geometry and timing conditions, but local results must be described as ROS executions of the v4 scenarios rather than reproductions of VEGA v4 scores.

The separate ROS controller currently has a known stale-command risk between episode children. See `DynaNav/VEGA_WORKFLOW_COMPARISON.md` before using these scenarios for final reported results.

## Why there are three scenarios

The three files are three distinct Hospital locations and interaction patterns, not three fragments of one episode. A navigation policy can succeed in a simple head-on encounter but fail when people cross laterally, when several pedestrians compete for space, or when the visual background and route orientation change. Using three locations avoids drawing a conclusion from one geometry and increases crowd complexity in controlled steps.

Each file contains five episodes. Within a file, the scene geometry, robot start, target and pedestrian routes remain fixed. The five episodes vary pedestrian `initial_idle` values, changing when the robot and moving people meet. This produces paired timing trials rather than five different maps.

Across the complete local suite there are 15 episodes:

```text
3 scenario geometries x 5 paired timing trials = 15 episodes
```

Running a single scenario produces a valid targeted test, but it does not cover the complete v4 scenario set. Use all three scenarios for a reported suite-level comparison.

## Scenario differences

All three scenarios use the same task: navigate to a stationary person wearing a long white coat and light trousers. V4 moved each target laterally so that the target is visible in the initial centered camera crop rather than being hidden behind the head-on pedestrian.

| Config | Location and robot route | People | Moving interaction | Primary difficulty |
|---|---|---:|---|---|
| `scenario1_5.yaml` | Reception/diagonal route; start `(4.0, -1.2)`, yaw `31.68 deg`; goal `(8.5727, 4.65)` | 2 | One head-on walker plus the stationary target | Basic target recognition and head-on avoidance |
| `scenario2_5.yaml` | East corridor; start `(19.0, 25.0)`, yaw `-90 deg`; goal `(19.95, 18.441)` | 3 | One head-on walker, one lateral crosser and the target | Coordinating longitudinal and crossing motion |
| `scenario3_5.yaml` | West corridor; start `(-26.0, 11.0)`, yaw `0 deg`; goal `(-19.441, 9.85)` | 4 | One head-on walker, two opposite lateral crossers and the target | Highest crowd density and conflicting crossings |

### Scenario 1

Scenario 1 is the minimum dynamic case. One person approaches along the robot's general direction of travel while the white-coat target remains stationary. Use it first for ROS connectivity, reset, controller and camera checks.

### Scenario 2

Scenario 2 adds a person crossing the corridor laterally. The robot must respond to both an oncoming person and a crossing trajectory. It also uses a different Hospital location and camera orientation, so it tests more than simply adding one actor.

### Scenario 3

Scenario 3 adds two lateral pedestrians moving in opposite directions around the robot's route, in addition to the head-on person. It is the most crowded and interaction-heavy condition. Run it after scenarios 1 and 2 when diagnosing the pipeline, because failures have more possible causes.

## Suggested use

For integration work, run scenario 1 first and inspect ROS topic rates, episode reset, the first waypoint and the first applied command. Once that passes, run scenarios 2 and 3 to exercise crossing interactions and increasing crowd density.

For a reported comparison, run all 15 episodes and preserve every valid outcome. Keep infrastructure failures separate from task failures, and report that the local execution is asynchronous. Do not select only the scenarios or timing trials on which a particular policy succeeds.
