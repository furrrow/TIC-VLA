# Crossover Scenario 3 suite

This Hospital suite keeps the white-coat target, robot start and goal, head-on
walker, stratified timing schedules, and 20-trial difficulty structure from the
southbound suite. The head-on walker is offset `0.25 m` to the robot's right,
so continuing straight still causes a collision while avoidance is biased to
the left. The additional pedestrians create a diagonal crossing and a
near-southbound obstacle in that left-side avoidance region.

| Difficulty | People | Moving pedestrians |
| --- | ---: | --- |
| Easy | 2 | One head-on walker |
| Medium | 3 | Head-on walker plus one south-left crossover walker |
| Hard | 4 | Head-on walker, diagonal crosser, and near-southbound left walker |

The right-side pedestrian starts farther down the hallway and to the right at
`[-20.8, 9.2]`, then walks south-left to `[-23.8, 12.8]`. Its steep diagonal
brings it into the robot's intended left-avoidance region after the head-on
encounter. The second pedestrian starts farther left at `[-19.8, 11.6]`, then
veers south-left to `[-23.3, 13.8]`. Both now use
aggressive leftward slopes while progressing south toward decreasing `x`.

The robot's initial yaw cycles through `0`, `7.5`, and `15` degrees, so it
always starts facing straight ahead or toward the left. Across 20 trials, `0`
and `7.5` occur seven times each and `15` occurs six times. Trial indices use
the same yaw in easy, medium, and hard.

Both crossover pedestrians use delays from `2.0` to `3.0` seconds, making them
more likely to enter the robot's left-side avoidance path soon after the
head-on encounter.

This suite uses the generator's `0.50 m` center-to-center pedestrian collision
threshold.

The success threshold is `2.0 m`, inherited from the generator default.

Regenerate the suite from the repository root:

```bash
python3 DynaNav/configs/generate_hospital_scenarios.py \
  --people 2,3,4 \
  --head-on-delays 0.2:1.5:20 \
  --crosser-a-delays 2.0:3.0:20 \
  --crosser-b-delays 2.0:3.0:20 \
  --combination-mode stratified \
  --lateral-motion crossover \
  --start-yaw-degrees=0,7.5,15 \
  --target-profile white-coat \
  --difficulty-names \
  --filename-prefix scenario3_crossover \
  --output-dir DynaNav/configs/crossover_scenarios \
  --force
```
