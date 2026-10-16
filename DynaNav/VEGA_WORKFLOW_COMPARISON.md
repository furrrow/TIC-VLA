# Local ROS Benchmark vs. VEGA-DynaNav

This document records the workflow comparison performed on 2026-10-15 between this checkout's ROS-based DynaNav benchmark and the VEGA evaluation workspace on Nexus. It is also a handoff document for future implementation work.

## Repositories and reference files

Local checkout:

```text
/home/jim/Projects/TIC-VLA
```

Remote VEGA evaluation workspace:

```text
jianyu34@nexusgamma00.umiacs.umd.edu:
/fs/nexus-projects/REAL2SIM3/project-VEGA/external-benchmarks/vega-dynanav
```

The remote directory is an experiment workspace, not a Git checkout. Its live top-level `icra_suite_manifest.json` has advanced beyond the experiment discussed here. Use these frozen files when referring to ICRA scenario v3:

```text
icra_archive_person_goal_v3_20260915/icra_suite_manifest.json
icra_scenarios_v3/scenario1_5.yaml
icra_scenarios_v3/scenario2_5.yaml
icra_scenarios_v3/scenario3_5.yaml
icra_scenarios_v3/*_validated.json
icra_scenarios_v3/code_snapshot/
```

Important local files:

```text
DynaNav/benchmark.py
DynaNav/benchmark_config.py
DynaNav/benchmark_reporting.py
DynaNav/run_benchmark.sh
DynaNav/behavior/nova_carter_ros_bridge.py
infer_ros.py
custom_utils/custom_utils/kinematic_controller_naive.py
configs/robot.yaml
nova_ros.bash
DynaNav/configs/farther_goals_50_v1_scn1_50trials.yaml
```

## Executive summary

Both systems use Isaac Sim, Nova Carter, the DynaNav scenario format, language-conditioned waypoint policies, and benchmark-owned success/collision metrics. Their main difference is the policy and control boundary.

The local system is an asynchronous ROS 2 pipeline. Isaac publishes camera and odometry, a persistent inference process publishes a body-frame path, a persistent controller converts a selected waypoint to `Twist`, and the simulator applies wheel commands. Each episode gets a fresh Isaac process.

VEGA-DynaNav uses a manifest-driven evaluation pipeline. A dispatcher selects validated work, a persistent Isaac application opens a fresh USD stage per episode, and the simulator calls a separately hosted policy through synchronous HTTP. Returned waypoints are consumed by a controller inside the simulator process. It adds strong validation, provenance, resumption, and measurement-integrity checks.

The best direction is to retain the local ROS architecture and asynchronous execution while borrowing VEGA's episode validity, resumption, provenance, readiness, reset audits, traces, and physical-contact measurement practices.

## Architecture

### Local ROS workflow

```text
run_benchmark.sh
  -> benchmark.py parent
     -> one benchmark.py child per episode
        -> fresh SimulationApp
        -> BenchmarkRunner owns scene, people, physics, goal and metrics
        -> nova_carter_ros_bridge.py
           publishes camera, odometry and episode metadata
           subscribes to cmd_vel
           applies Twist to Nova Carter wheels

persistent external ROS processes:
  infer_ros.py
    camera + odometry + episode metadata -> full body-frame path + lookahead waypoint

  kinematic_controller_naive.py
    selected waypoint -> geometry_msgs/Twist
```

`nova_ros.bash` starts the inference and controller processes in tmux. The benchmark is normally started separately.

### VEGA-DynaNav workflow

```text
policy launcher
  -> long-running model-specific HTTP server
  -> writes policy_url.txt after readiness succeeds

run_icra_dispatch.sh
  -> icra_dispatch.py
     validates config hash and review record
     finds missing or invalid episodes
     obtains a per-policy lock
     checks policy readiness
     records launch provenance
     -> persistent_suite.py
        one persistent SimulationApp
        one fresh USD stage per episode
        -> DynaNav BenchmarkRunner
           RGB + instruction -> synchronous HTTP policy request
           returned waypoints -> in-simulator controller
           contact/FOV instrumentation -> per-episode result
     -> summarize_icra.py
```

## Side-by-side comparison

| Concern | Local ROS benchmark | VEGA-DynaNav v3 |
|---|---|---|
| Top-level work selection | YAML episode order in `benchmark.py` | Frozen manifest groups handled by `icra_dispatch.py` |
| Simulator lifetime | Fresh process and `SimulationApp` per episode | Persistent app, fresh USD stage per episode |
| Policy service | Persistent ROS inference node | Persistent HTTP policy server |
| Observation transport | Compressed ROS image plus odometry | Image path and JSON HTTP request |
| Policy execution | Asynchronous wall-clock inference | Synchronous; simulation waits for the response |
| Policy output | Full body-frame path | Waypoints or a model-specific native action |
| Waypoint selection | External ROS inference node uses configured lookahead | In-simulator DynaNav controller |
| Low-level control | Separate ROS node publishes `Twist` | Controller runs in the simulator process |
| Robot command | ROS `/isaacsim/cmd_vel` | Direct simulator wheel/action command |
| Episode reset | Transient-local episode message and inference generation guard | Episode/frame metadata and policy reset, followed by reset audit |
| Existing-result resume | `--start_from`; no validity-based automatic resume | Scans individual result files and skips only valid episodes |
| Infrastructure failures | Currently enter aggregate results as failures | Excluded from task metrics and left pending for retry |
| Scenario validation | Schema validation | Config SHA256 plus explicit validation record |
| Provenance | Run ID and config path | Config/code hashes, checkpoint, policy, host, job, runtime and environment |
| Result storage | Incrementally rewritten combined result | Atomic JSON, log and trace directories per episode |
| Parallel execution | Separate run IDs | Shared worker state and per-policy locks |
| Collision measurement | Distance test plus sustained contact sensor | PhysX contacts against explicit human capsules and static geometry |
| Extra safety metric | None equivalent | Per-frame camera-FOV human clearance |
| v3 controller limits | Local `sim_nova`: 1.5 m/s, 1.2 rad/s | 1.0 m/s, 0.3 rad/s |

## Timing semantics

The workflows do not measure the same timing regime.

VEGA v3 uses synchronous policy calls. While the policy server performs inference, the simulator waits. A slow inference response therefore does not allow the robot or pedestrians to advance through additional simulated time.

The local ROS system is asynchronous. Isaac physics and scripted pedestrians continue while inference runs. The controller continues operating from the most recently received waypoint. Policy latency consequently changes the physical encounter, especially in head-on and crossing scenarios.

Keep asynchronous execution because it is representative of deployment, but record enough timestamps to measure it:

```text
episode ID
simulation frame
camera timestamp
inference start and end
path publication timestamp
selected waypoint timestamp
Twist publication timestamp
frame in which the command was applied
observation-to-action age
```

Results from the synchronous VEGA protocol and the asynchronous ROS protocol should be labeled separately even when they use the same YAML geometry.

## Local reset behavior and stale-command risk

The local inference reset is already strong:

1. The Isaac bridge publishes a reliable, transient-local `/isaacsim/episode` JSON message.
2. Its `episode_id` contains the run ID, episode name and child PID.
3. `infer_ros.py` clears model state, image history, cached generation references, timestamps and counters for every new episode ID.
4. An `episode_generation` check prevents an inference started for an old episode from publishing into a new episode.

The controller is not episode-aware. This leaves a concrete stale-command path:

1. An Isaac child exits, but the ROS controller remains alive.
2. The controller retains its previous waypoint and `vel_msg`.
3. It continues publishing the previous `Twist` at the controller rate.
4. A new Isaac child subscribes and can accept that command before receiving a path for its episode.

The Isaac bridge's 0.5-second command watchdog does not solve this because the controller continually republishes the stale command, making it fresh at the transport layer.

There is a second issue in `kinematic_controller_naive.py`: when its waypoint becomes older than the one-second timeout, it stops recomputing velocity but still publishes the previous `vel_msg`. The command can therefore persist indefinitely.

The first local improvement should make the controller subscribe to the episode topic and do the following for every new episode ID:

```text
clear the stored waypoint
set vel_msg to zero
publish zero immediately
refuse nonzero output until a current-episode waypoint arrives
publish zero whenever the waypoint expires
publish zero during shutdown
```

For an end-to-end guarantee, replace untagged `Float32MultiArray`/`Twist` at the policy boundary with a stamped message containing the episode ID, source observation timestamp and sequence number. Plain `Twist` cannot prove which episode produced a command.

## Result validity difference

The local child writes an error-shaped result when setup or execution fails. The parent normalizes it, appends it to `all_results`, and `benchmark_reporting.summarize_results()` includes it in the denominator as an unsuccessful episode. An Isaac startup failure or disconnected ROS process can therefore lower reported success rate.

VEGA separates three outcomes:

```text
valid task success
valid task failure
invalid or infrastructure failure
```

Only valid task executions enter success and collision metrics. Invalid executions remain pending and are retried. A VEGA result is accepted only when it has a positive duration, the expected measurement version, complete per-frame measurement coverage, valid inference evidence, and no infrastructure error.

The local reporter should adopt the same distinction. Do not turn an infrastructure failure into a zero-success, zero-collision navigation trial.

## ICRA v3 scenario reference

ICRA v3 evaluates four model/controller stacks over three Hospital locations with five paired pedestrian timings per location. VEGA checkpoint 76000 received priority, followed by OmniVLA, NaVILA and TIC-VLA.

All v3 scenarios use:

```text
Nova Carter
seed 36
success threshold 1.5 m
configured timeout 40 s
stationary white-coat target person at the goal
instruction: go to person described by the person wearing a long white coat and
             light trousers standing farther away
physical_human_metrics: true
```

Scenario structure:

| Scenario | People | Dynamic interaction |
|---|---:|---|
| 1 | 2 | Stationary target plus one head-on walker |
| 2 | 3 | Target, head-on walker and one lateral crosser |
| 3 | 4 | Target, head-on walker and two opposite lateral crossers |

The five trials preserve geometry and vary pedestrian `initial_idle` values. They are paired timing conditions rather than independently randomized environments.

V3 has a known initial-visibility limitation. In scenario 1, the head-on walker heavily occluded the stationary target in the initial centered crop. Scenario 2 and scenario 3 still had pending new-camera review in the frozen validation records. V4 later shifted target positions laterally to improve initial visibility.

## Local farther-goals scenario relative to v3

`DynaNav/configs/farther_goals_50_v1_scn1_50trials.yaml` preserves the v3 scenario-1 concept but changes its geometry and timing coverage.

| Property | ICRA v3 scenario 1 | Local farther-goals scenario 1 |
|---|---:|---:|
| Trials | 5 | 50 |
| Start | `(4.0, -1.2)` | `(5.0, 0.0)` |
| Start yaw | `31.68 deg` | `52.46 deg` |
| Goal | `(8.573, 3.502)` | `(10.400, 7.029)` |
| Straight-line goal distance | 6.56 m | 8.86 m |
| Robot-to-walker initial distance | 4.52 m | 3.89 m |
| Walker route length | 5.52 m | 4.30 m |
| Robot start to walker endpoint | 1.004 m | 0.986 m |
| People | Target plus one head-on walker | Target plus one head-on walker |

The local goal is 2.31 m, or about 35%, farther from the start. The nearly identical robot-start-to-walker-endpoint distance indicates that the encounter structure was deliberately preserved while extending the target distance.

All 50 local trials have fixed geometry and unique walker idle times spanning 0.2 to 1.5 seconds, with mean and median 0.85 seconds. They are 50 timing conditions for one scenario, not 50 independent scenes.

The episode identifiers still use the inherited `episode_far20_s1_*` prefix even though there are 50 trials. Rename them before freezing a new protocol unless existing results already depend on those identifiers.

## VEGA elements worth borrowing

### Priority 1: episode-aware safe control

- Reset the controller on the episode message.
- Publish zero on reset, stale waypoint and shutdown.
- Reject commands from a different episode or an old observation.
- Add an explicit current-episode readiness state.

### Priority 2: readiness barrier

Before allowing navigation physics to proceed, require acknowledgements that:

- Inference has loaded its model and subscribed to the current episode.
- The controller has reset and is publishing zero.
- The simulator bridge has published the current episode metadata.
- The first current-episode observation has reached inference.

The existing `/started` signal is published after first inference, but the benchmark does not currently use it as a start barrier.

### Priority 3: atomic per-episode artifacts and resume

Use a layout such as:

```text
DynaNav/benchmark_results/<run_id>/
  manifest.json
  status.json
  episodes/<episode_name>.json
  logs/<episode_name>.log
  traces/<episode_name>/policy.jsonl
  traces/<episode_name>/commands.jsonl
```

Write JSON through a temporary sibling followed by `os.replace`. On startup, skip only result files that pass the full validity predicate. Retry missing, malformed or infrastructure-invalid episodes.

### Priority 4: valid-run reporting

- Exclude infrastructure-invalid records from SR, CR, SPL and navigation-error denominators.
- Show valid, missing and invalid counts separately.
- Retain invalid records and error logs for audit.
- Require complete groups for final model comparisons.

### Priority 5: frozen manifest and provenance

A run manifest should include:

```text
protocol name and revision
config path and SHA256
ordered expected episode IDs
Git commit and dirty-state description
Isaac Sim version and runtime path
model/checkpoint identifier and hash
inference code hash
controller code hash and numeric settings
ROS domain and topic names
measurement version
worker host and scheduler job, when applicable
```

Each external process should expose or write its own provenance. The benchmark should collect it before declaring the run ready.

### Priority 6: reset and inference audits

For each episode verify:

```text
first observation belongs to the current episode
first path derives from that observation
first command derives from that path and episode
no old command was applied
first nonzero command occurs near the configured start
inference and measurement traces cover the navigation interval
```

### Priority 7: durable latency/action traces

Record one JSONL event per observation, inference, selected waypoint, command publication and applied command. This allows model failures, controller failures, transport delay and simulator delay to be distinguished.

### Priority 8: PhysX capsule/contact/FOV metrics

Consider porting the frozen v3 `contact_fov_metrics.py` approach:

- Invisible kinematic human capsules, radius 0.25 m and total height 1.8 m.
- Actual PhysX contact reports.
- Separate human and static-obstacle contact.
- Per-frame measurement coverage.
- Camera-FOV human surface clearance.

Port the instrumentation and define a new measurement version; do not copy only its output field names. The current local collision semantics differ from v3.

### Lower priority: persistent Isaac application

The VEGA persistent app reduces startup cost but requires careful stage, sensor, behavior, ROS and policy reset. The local fresh-child design gives stronger isolation. Keep fresh children until the reset and validity audits above pass reliably. Consider persistence later if startup dominates the 50-trial workload.

## Suggested implementation order

1. Fix stale controller output and make control episode-aware.
2. Add an episode readiness/acknowledgement barrier.
3. Add durable inference and command traces with timestamps.
4. Add atomic per-episode results and a validity predicate.
5. Add automatic resume and retry of invalid episodes.
6. Separate infrastructure-invalid runs from task failures in reports.
7. Add a frozen manifest and process provenance.
8. Add first-action/reset audits.
9. Port or redesign the PhysX contact and FOV measurement layer.
10. Evaluate persistent Isaac execution only after reset testing.

## Handoff template for a fresh GPT chat

Use this template to start work on one item:

```text
Work in /home/jim/Projects/TIC-VLA.

Read DynaNav/VEGA_WORKFLOW_COMPARISON.md first. It compares the current local
ROS DynaNav workflow with the frozen VEGA-DynaNav ICRA v3 workflow and records
the verified risks and borrowing plan.

Implement this specific item:
<INSERT ONE ITEM FROM "Suggested implementation order">

Preserve these architectural constraints:
- benchmark.py owns episode lifecycle, simulation, people and metrics;
- infer_ros.py remains a separate long-running ROS process;
- the waypoint controller remains a separate ROS process unless the requested
  item explicitly changes that boundary;
- local execution remains asynchronous;
- each episode continues to use a fresh Isaac child unless explicitly requested;
- infrastructure failures must not be silently converted into task failures.

Inspect the relevant files and current uncommitted changes before editing.
Implement the change, add only meaningful tests or validation, and update
DynaNav/VEGA_WORKFLOW_COMPARISON.md if the documented architecture or status
changes.
```

For remote verification, use read-only SSH access to:

```text
ssh jianyu34@nexusgamma00.umiacs.umd.edu
cd /fs/nexus-projects/REAL2SIM3/project-VEGA/external-benchmarks/vega-dynanav
```

Do not treat the live top-level manifest as the v3 definition. Use the archived v3 manifest and `icra_scenarios_v3/code_snapshot` listed at the beginning of this document.
