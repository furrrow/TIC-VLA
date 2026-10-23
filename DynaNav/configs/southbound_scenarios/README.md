# Southbound Scenario 3 suite

This suite is a directional-flow variant of `scenario3_difficulty`. It keeps
the Hospital scene, robot start and goal, stationary target, head-on walker,
timing schedules, and 20-trial structure unchanged. The additional pedestrians
both walk south on separate fixed-x lanes instead of shuttling in opposing
directions.

| Difficulty | People | Moving pedestrians |
| --- | ---: | --- |
| Easy | 2 | One head-on walker |
| Medium | 3 | Southbound walkers at `y = 11.0` and `y = 11.85` |
| Hard | 4 | Southbound walkers at `y = 11.0`, `y = 11.85`, and `y = 9.55` |

The moving pedestrians travel from `x = -21.8` toward `x = -26.8` in separate
fixed-y lanes. The original walker occupies the center lane at `y = 11.0`; the
added walkers occupy `y = 11.85` and `y = 9.55`, with the latter on the robot's
right. Each added walker travels once and remains at its destination. The
stratified start delays control their longitudinal spacing.

Regenerate the suite from the repository root:

```bash
python3 DynaNav/configs/generate_hospital_scenarios.py \
  --people 2,3,4 \
  --head-on-delays 0.2:1.5:20 \
  --crosser-a-delays 2.0:4.0:20 \
  --crosser-b-delays 3.0:5.0:20 \
  --combination-mode stratified \
  --lateral-motion southbound \
  --start-yaw-degrees=-30,-15,0,15,30 \
  --target-profile white-coat \
  --difficulty-names \
  --output-dir DynaNav/configs/southbound_scenarios \
  --force
```
