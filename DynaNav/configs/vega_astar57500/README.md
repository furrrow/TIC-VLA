# VEGA ASTAR-57500 language evaluation scenarios

These are exact copies of the three scenario configs selected by the remote
`vega-dynanav/astar_57500/manifest.json` used by
`vega-dynanav/astar_57500/evaluate.sh`.

Remote source root:

`/fs/nexus-projects/REAL2SIM3/project-VEGA/external-benchmarks/vega-dynanav`

| Local file | Remote source | Scenario |
| --- | --- | --- |
| `scene1_50trials.yaml` | `farther_goals_50_v1/scene1_50trials.yaml` | Hospital, two people, one head-on moving distractor and a farther stationary target |
| `scene2_50trials.yaml` | `farther_goals_50_v1/scene2_50trials.yaml` | Hospital, three people, two crossing distractors and a stationary target |
| `scene3_50trials.yaml` | `scene3_warehouse_v6/scene3_50trials.yaml` | Warehouse, four people, three moving distractors and a stationary target |

Each file contains 50 trials with fixed scene geometry. The paired trials vary
the onset delay (`initial_idle`) of the primary moving person according to the
same 50-value schedule. All use a 40-second timeout, a Nova Carter robot, a
1.5-meter success threshold, and physical human collision/FOV metrics.

The evaluator workflow is:

1. `evaluate.sh` starts `astar_57500/resume_worker.py` in the simulator job.
2. The worker invokes `icra_dispatch.py` with `astar_57500/manifest.json`.
3. The dispatcher validates config hashes and validation records, waits for the
   separately launched ASTAR HTTP policy service, and chooses the least-complete
   scene in scene cycles.
4. It runs one pending episode at a time through `persistent_suite.py`, using a
   fresh Isaac Sim stage for every episode while keeping the application alive.
5. Results, prediction traces, logs, launch provenance, and summary reports are
   written after each chunk so a requeued evaluator can resume completed work.

Pinned SHA-256 checksums from the remote manifest:

```text
2be961f5ae4e9cb1c51f4b73df23cf6bbcd08784f79c8ced7c2fbcec3197178f  scene1_50trials.yaml
eba7dc6eab218d1fca7f0df33f299991d23129aee9df8747688c997890b37d36  scene2_50trials.yaml
d07e7ba9e25ddd0e54c3c9d90f14ab87208dec4a7777922c57319d3d1560b770  scene3_50trials.yaml
```
