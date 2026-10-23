# Southbound Scenario 3 suite

This suite is a directional-flow variant of `scenario3_difficulty`. It keeps
the Hospital scene, robot start and goal, stationary target, head-on walker,
timing schedules, and 20-trial structure unchanged. The additional pedestrians
both walk south on separate fixed-x lanes instead of shuttling in opposing
directions.

| Difficulty | People | Moving pedestrians |
| --- | ---: | --- |
| Easy | 2 | One head-on walker |
| Medium | 3 | Head-on walker plus southbound walker at `x = -23.4` |
| Hard | 4 | Head-on walker plus southbound walkers at `x = -23.4` and `x = -22.3` |

Each southbound pedestrian spawns at `y = 11.85`, walks once to `y = 9.55`,
and remains at that destination. Their routes are fixed and do not track or aim
at the robot. They still pass through the robot corridor geometrically, so the
stratified start delays control whether they arrive near the robot's trajectory
at the same time.

Regenerate the suite from the repository root:

```bash
python3 DynaNav/configs/generate_hospital_scenarios.py \
  --people 2,3,4 \
  --head-on-delays 0.2:1.5:20 \
  --crosser-a-delays 2.0:4.0:20 \
  --crosser-b-delays 3.0:5.0:20 \
  --combination-mode stratified \
  --lateral-motion southbound \
  --target-profile white-coat \
  --difficulty-names \
  --output-dir DynaNav/configs/southbound_scenarios \
  --force
```
