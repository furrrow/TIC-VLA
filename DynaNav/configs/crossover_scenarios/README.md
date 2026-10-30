# Crossover Scenario 3 suite

This Hospital suite keeps the white-coat target, robot start and goal, head-on
walker, stratified timing schedules, and 20-trial difficulty structure from the
southbound suite. The head-on walker is offset `0.15 m` to the robot's right,
so continuing straight still causes a collision while avoidance is biased to
the left. The additional pedestrians cross the robot's path diagonally while
continuing south.

| Difficulty | People | Moving pedestrians |
| --- | ---: | --- |
| Easy | 2 | One head-on walker |
| Medium | 3 | Head-on walker plus one south-left crossover walker |
| Hard | 4 | Head-on walker plus two staggered south-left crossover walkers |

The right-side pedestrian starts farther down the hallway and to the right at
`[-20.8, 9.2]`, then walks south-left to `[-25.8, 12.8]`. Its steeper diagonal
brings it into the robot's intended left-avoidance region after the head-on
encounter. The second crossover pedestrian starts still farther down the
hallway and to the right at `[-19.8, 8.8]`, then walks south-left to
`[-24.8, 12.4]`. Both routes cross the robot's centerline at `y = 11.0` while
progressing south toward decreasing `x`.

The robot's initial yaw cycles through `-15`, `0`, and `15` degrees. Across 20
trials, `-15` and `0` occur seven times each and `15` occurs six times. Trial
indices use the same yaw in easy, medium, and hard.

Both crossover pedestrians use delays from `2.0` to `3.0` seconds, making them
more likely to enter the robot's left-side avoidance path soon after the
head-on encounter.

Regenerate the suite from the repository root:

```bash
python3 DynaNav/configs/generate_hospital_scenarios.py \
  --people 2,3,4 \
  --head-on-delays 0.2:1.5:20 \
  --crosser-a-delays 2.0:3.0:20 \
  --crosser-b-delays 2.0:3.0:20 \
  --combination-mode stratified \
  --lateral-motion crossover \
  --start-yaw-degrees=-15,0,15 \
  --target-profile white-coat \
  --difficulty-names \
  --filename-prefix scenario3_crossover \
  --output-dir DynaNav/configs/crossover_scenarios \
  --force
```
