# Corrected person-target evaluation

The goal is the stationary person in the long white coat and light trousers. Prompt: `go to person described by the person wearing a long white coat and light trousers standing farther away`.

Scene 1: stationary target person plus one head-on moving person (2 people total).
Scene 2: stationary target plus one head-on and one sideways moving person (3 total).
Scene 3: stationary target plus one head-on and two opposite sideways moving people (4 total).

The three hospital locations, controller limits (1 m/s and 0.3 rad/s), VEGA checkpoint 76000 with 10 denoising steps, five paired trials per group, physical-contact measurement and camera-FOV clearance are preserved. All 15 VEGA trials have priority before the other models.

No target-object relocation is configured. The original green bin remains ordinary scene geometry. Green-bin-target trials resulted from an assistant misunderstanding and are archived under ../icra_archive_greenbin_v2_20260915, with their raw result directories intact. They do not contribute to these results.

Validation initially combines the original person-target scene evidence with the separately verified moving routes. New camera evidence is recorded as the full trials run, without selecting outcomes or using short trials for scores.
