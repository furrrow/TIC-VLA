# Crossover Scenario 3 suite

This Hospital suite keeps the white-coat target, robot start and goal, head-on
walker, stratified timing schedules, and 20-trial difficulty structure from the
southbound suite. The additional pedestrians cross the robot's path diagonally
while continuing south.

| Difficulty | People | Moving pedestrians |
| --- | ---: | --- |
| Easy | 2 | One head-on walker |
| Medium | 3 | Head-on walker plus right-to-left crossover walker |
| Hard | 4 | Head-on walker plus opposing crossover walkers |

The right-side pedestrian starts at `[-21.8, 9.8]` and walks diagonally to
`[-26.8, 11.6]`. The left-side pedestrian starts at `[-21.8, 11.6]` and walks
diagonally to `[-26.8, 9.8]`. Both routes cross the robot's centerline at
`y = 11.0` while progressing south toward decreasing `x`.

The robot's initial yaw cycles through `-15`, `0`, and `15` degrees. Across 20
trials, `-15` and `0` occur seven times each and `15` occurs six times. Trial
indices use the same yaw in easy, medium, and hard.

Regenerate the suite from the repository root:

```bash
python3 DynaNav/configs/generate_hospital_scenarios.py \
  --people 2,3,4 \
  --head-on-delays 0.2:1.5:20 \
  --crosser-a-delays 2.0:4.0:20 \
  --crosser-b-delays 2.0:4.0:20 \
  --combination-mode stratified \
  --lateral-motion crossover \
  --start-yaw-degrees=-15,0,15 \
  --target-profile white-coat \
  --difficulty-names \
  --filename-prefix scenario3_crossover \
  --output-dir DynaNav/configs/crossover_scenarios \
  --force
```
