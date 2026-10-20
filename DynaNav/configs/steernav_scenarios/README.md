# SteerNav scenario suite

This directory contains four generated navigation conditions designed to
separate increasing interaction complexity, transfer to a new visual
environment, and the effect of a second opposing crosser. Every condition has
20 deterministic, paired timing trials.

## Suite design

| Config | Environment | People | Moving interaction | Experimental role |
| --- | --- | ---: | --- | --- |
| `scenario1_easy_hospital_20trials.yaml` | Hospital | 2 | One head-on pedestrian | Basic navigation and avoidance |
| `scenario2_challenging_hospital_20trials.yaml` | Hospital | 3 | Head-on pedestrian plus one lateral crosser | Added interaction complexity in the same geometry |
| `scenario3_transfer_warehouse_20trials.yaml` | Warehouse | 3 | Head-on pedestrian plus one lateral crosser | Transfer of a similar interaction class to a new environment |
| `scenario4_stress_warehouse_20trials.yaml` | Warehouse | 4 | Head-on pedestrian plus two opposing lateral crossers | Isolate the effect of the second crosser |

The people count includes the stationary target. Scenario 1 therefore has one
moving person; Scenarios 2 and 3 each have two moving people; Scenario 4 has
three moving people.

The intended comparisons are:

1. Scenario 1 versus Scenario 2 measures the effect of adding a crossing
   conflict while retaining the Hospital template.
2. Scenario 2 versus Scenario 3 measures transfer from Hospital to Warehouse
   with the same number and types of moving interactions.
3. Scenario 3 versus Scenario 4 measures the incremental effect of adding a
   second opposing crosser in the Warehouse.

Scenario 3 is called `transfer`, rather than `hard`, because a performance
change can come from appearance, geometry, target grounding, or timing—not only
from greater crowd difficulty.

Scenario 4 is a paired stress ablation. For every trial index, its robot,
target, head-on delay, and first-crosser delay match Scenario 3 exactly. The
second crosser is the only added actor.

## Templates

The Hospital generator uses the geometry from ICRA v4 Scenario 3:

- Robot start: `(-26.0, 11.0)`, yaw `0 degrees`
- Target: `(-19.441016, 9.85)`
- Target description: long white coat and light trousers
- Head-on route: `(-21.8, 11.0)` toward `(-26.8, 11.0)`
- First crossing route: `(-23.4, 9.55)` toward `(-23.4, 11.85)`

The Warehouse generator uses `icra_scenarios_v4/scenario4_1.yaml`:

- Robot start: `(-10.0, 1.0)`, yaw `90 degrees`
- Target: `(-8.2, 8.2)`
- Target description: blue shirt and dark trousers
- Head-on route: `(-11.2, 4.9)` toward `(-11.2, 0.1)`
- First crossing route: `(-12.5, 5.2)` toward `(-6.0, 5.2)`
- Second crossing route: `(-7.5, 6.2)` toward `(-12.5, 6.2)`

Both use Nova Carter, a 40-second timeout, a 1.5-meter success threshold,
physical-human metrics, and the `ticvla_ros` navigation method.

## Timing design

The generators use `--combination-mode stratified` with seed 36. Each active
pedestrian uses every value in its delay range exactly once. Crosser delays are
independently shuffled, preventing all pedestrians from always starting early
or late together.

| Scenario | Head-on delay | First-crosser delay | Second-crosser delay |
| --- | --- | --- | --- |
| Scenario 1 | 20 values from 0.2–1.5 s | Not present | Not present |
| Scenario 2 | 20 values from 0.2–1.5 s | 20 shuffled values from 2.0–4.0 s | Not present |
| Scenario 3 | 20 values from 0.2–1.5 s | 20 shuffled values from 0.2–1.5 s | Not present |
| Scenario 4 | Same as Scenario 3 | Same as Scenario 3 | 20 shuffled values from 0.5–2.5 s |

The head-on schedule is identical at each trial index across all four files.
Scenario 3 and Scenario 4 also have identical first-crosser schedules at each
index. This makes their difference a direct paired test of the second crosser.
Scenario 2's crosser schedule was selected for the Hospital crossing geometry.
Scenario 3's initial range follows the 0.2–1.5-second timing envelope used by
the source Warehouse evaluation.

The raw Hospital and Warehouse delays do not guarantee identical time to
conflict because their routes differ. Before a final reported evaluation, run a
pilot and inspect robot and pedestrian arrival times at the crossing point. If
Warehouse outcomes cluster near 0% or 100% collision, adjust its crosser delay
range and regenerate the complete suite before collecting final results.

## Recreate the suite

Run these commands from the repository root.

```bash
python DynaNav/configs/generate_hospital_scenarios.py \
  --people 2 \
  --head-on-delays 0.2:1.5:20 \
  --combination-mode stratified \
  --output-dir DynaNav/configs/steernav_scenarios \
  --output-name scenario1_easy_hospital_20trials.yaml \
  --episode-prefix episode_s1_hospital \
  --force
```

```bash
python DynaNav/configs/generate_hospital_scenarios.py \
  --people 3 \
  --head-on-delays 0.2:1.5:20 \
  --crosser-a-delays 2.0:4.0:20 \
  --combination-mode stratified \
  --output-dir DynaNav/configs/steernav_scenarios \
  --output-name scenario2_challenging_hospital_20trials.yaml \
  --episode-prefix episode_s2_hospital \
  --force
```

```bash
python DynaNav/configs/generate_warehouse_scenarios.py \
  --people 3 \
  --head-on-delays 0.2:1.5:20 \
  --crosser-a-delays 0.2:1.5:20 \
  --combination-mode stratified \
  --output-dir DynaNav/configs/steernav_scenarios \
  --output-name scenario3_transfer_warehouse_20trials.yaml \
  --episode-prefix episode_s3_warehouse \
  --force
```

```bash
python DynaNav/configs/generate_warehouse_scenarios.py \
  --people 4 \
  --head-on-delays 0.2:1.5:20 \
  --crosser-a-delays 0.2:1.5:20 \
  --crosser-b-delays 0.5:2.5:20 \
  --combination-mode stratified \
  --output-dir DynaNav/configs/steernav_scenarios \
  --output-name scenario4_stress_warehouse_20trials.yaml \
  --episode-prefix episode_s4_warehouse \
  --force
```

The generators refuse to replace files unless `--force` is supplied.

## Run the suite

Each file can be passed directly to the benchmark launcher:

```bash
DynaNav/run_benchmark.sh \
  DynaNav/configs/steernav_scenarios/scenario1_easy_hospital_20trials.yaml \
  --run_id steernav_s1_easy_hospital
```

Use distinct run IDs for Scenarios 2–4 and preserve the trial index in all
result analysis so paired comparisons remain possible. In particular, compare
Scenario 3 and Scenario 4 episode by episode before aggregating their metrics.
