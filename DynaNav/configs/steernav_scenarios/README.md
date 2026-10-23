# SteerNav scenario suite

This directory contains three Hospital navigation conditions with increasing
interaction complexity. Every condition has 20 deterministic, stratified
timing trials.

## Suite design

| Config | Environment | People | Moving interaction | Experimental role |
| --- | --- | ---: | --- | --- |
| `scenario3_easy_20trials.yaml` | Hospital | 2 | One head-on pedestrian | Basic navigation and avoidance |
| `scenario3_medium_20trials.yaml` | Hospital | 3 | Head-on pedestrian plus one lateral crosser | Added crossing conflict |
| `scenario3_hard_20trials.yaml` | Hospital | 4 | Head-on pedestrian plus two opposing lateral crossers | Highest crowd interaction complexity |
| `scenario3_southbound_easy_20trials.yaml` | Hospital | 2 | One head-on pedestrian | Southbound-suite control condition |
| `scenario3_southbound_medium_20trials.yaml` | Hospital | 3 | Two southbound walkers in adjacent lanes | Two-lane directional flow |
| `scenario3_southbound_hard_20trials.yaml` | Hospital | 4 | Three southbound walkers in separate lanes | Three-lane directional flow |

The people count includes the stationary target. Easy therefore has one moving
person, medium has two, and hard has three.

The files share the same robot start, target, head-on timing schedule, and
Hospital geometry. Medium adds the first lateral crosser to easy; hard adds a
second, opposing lateral crosser to medium.

## Templates

The Hospital generator uses the geometry from ICRA v4 Scenario 3:

- Robot start: `(-26.0, 11.0)`, yaw `0 degrees`
- Goal: `(-19.441016, 11.0)` on the hallway centerline
- Stationary target: `(-18.8, 11.0)`, facing the robot
- Target description: security guard wearing a purple shirt and dark trousers
- Head-on route: `(-21.8, 11.0)` toward `(-26.8, 11.0)`
- First crossing route: `(-23.4, 9.55)` toward `(-23.4, 11.85)`
- Second crossing route: `(-22.3, 11.85)` toward `(-22.3, 9.55)`

All three use Nova Carter, a 40-second timeout, a 1.5-meter success threshold,
physical-human metrics, and the `ticvla_ros` navigation method.

## Timing design

The generators use `--combination-mode stratified` with seed 36. Each active
pedestrian uses every value in its delay range exactly once. Crosser delays are
independently shuffled, preventing all pedestrians from always starting early
or late together.

| Difficulty | Head-on delay | First crosser delay | Second crosser delay |
| --- | --- | --- | --- |
| Easy | 20 values from 0.2–1.5 s | Not present | Not present |
| Medium | Same as easy | 20 shuffled values from 2.0–4.0 s | Not present |
| Hard | Same as easy | 20 shuffled values from 2.0–4.0 s | 20 shuffled values from 3.0–5.0 s |

The head-on schedule is identical at each trial index across all three files.
The crosser schedules are independently shuffled with seed 36, so each delay
stratum appears once without forcing all pedestrians to start early or late
together.

## Southbound variants

The three `scenario3_southbound_*` files retain the same security-guard target,
robot geometry, crowd counts, and stratified timing schedules. The original
walker travels from `x = -21.8` to `x = -26.8` in the center lane at `y = 11.0`.
Medium adds a walker following the same direction in the `y = 11.85` lane.
Hard adds a third walker in the `y = 9.55` lane, to the robot's right. The
added walkers remain at their destinations after completing their one-way
routes.

Regenerate these additional files from the repository root:

```bash
python3 DynaNav/configs/generate_hospital_scenarios.py \
  --people 2,3,4 \
  --head-on-delays 0.2:1.5:20 \
  --crosser-a-delays 2.0:4.0:20 \
  --crosser-b-delays 3.0:5.0:20 \
  --combination-mode stratified \
  --lateral-motion southbound \
  --difficulty-names \
  --filename-prefix scenario3_southbound \
  --output-dir DynaNav/configs/steernav_scenarios \
  --force
```

## Recreate the suite

Run these commands from the repository root.

```bash
python3 DynaNav/configs/generate_hospital_scenarios.py \
  --people 2,3,4 \
  --head-on-delays 0.2:1.5:20 \
  --crosser-a-delays 2.0:4.0:20 \
  --crosser-b-delays 3.0:5.0:20 \
  --combination-mode stratified \
  --difficulty-names \
  --output-dir DynaNav/configs/steernav_scenarios \
  --force
```

The generators refuse to replace files unless `--force` is supplied.

## Run the suite

Run all three difficulty levels sequentially across OmniVLA, TIC-VLA, and
NaVILA from the repository root:

```bash
./run_all_nova_scenarios.bash
```

This defaults to ablation mode and leaves SteerNav's final waypoint free. You
do not need a flag to disable `fix-end`. To run only the hard condition with
the same default:

```bash
./run_all_nova_scenarios.bash --mode ablation \
  DynaNav/configs/steernav_scenarios/scenario3_hard_20trials.yaml
```

Pass `--fix-end` only when you intentionally want to hold the final waypoint
fixed during trajectory optimization. The compatibility flag `--no-fix-end`
is accepted but is redundant because that is already the default.
