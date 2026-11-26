# Crossover v3 campaign plan

Prepared 2026-11-26. Status: planned; no trials launched. This folder is a planning artifact, not a results archive.

Run all 540 episodes prospectively: OmniVLA, NaVILA, TIC-VLA × normal, no-velocity, steered × easy, medium, hard × 20 original initializations. Retain original episode IDs, pedestrian routes/delays, goals, and 40 simulation-second budgets. Run conditions sequentially.

Freeze success at horizontal goal distance <=1.5 m before collecting results. Retain stopping at <=0.5 m, goal-person collision at <0.4 m, separate goal/non-goal collision reporting, and safety-failure precedence over earlier goal proximity. Preserve moving-away and stall termination. Report the primary 1.5 m endpoint and a prespecified 1.0 m distance-based sensitivity analysis with the same safety exclusions. Do not treat runtime SPL as rescored for another radius.

Carry forward corrected first-waypoint fallback (only waypoint 1 may replace rejected waypoint 0), actual gate overlays/telemetry, 1.0 wall-second simulator command lease, ROS-clock behavioral timers, removal of extra controller wall watchdog, elapsed-simulation-time behavior budgets, and clock-reset handling. Snapshot actual loaded sources and settings across all policies and the simulator.

Planning assumption: retain corrected policy-specific additional-clearance settings: NaVILA 0.25, OmniVLA/TIC-VLA 0.5; robot radius 0.5 throughout. These are transformed-field thresholds, not guaranteed physical metres. This compares configured systems, not a common-clearance configuration. Freeze this choice explicitly before launch.

Other settings: estimated planning velocity (simulator truth shadow-only), 1.5 s projection horizon, five observations, full-width sweep/no taper, extrapolated odometry with maximum 0.25 s, image queue one, no repeated images, controller dt 0.75, free final waypoint. Preserve and record policy-specific controller lease lengths and NaVILA scale 2.

Use a new timestamped crossover_v3 campaign ID. Preserve all outcomes, including navigation failures; recover infrastructure failures only with distinct attempt IDs and explicit episode mapping. Require all 20 unique expected episodes per condition, validated manifests/scenario hashes, matching local/remote source provenance, clean writer shutdown and recorded drop/error counters. Preserve raw results and prior archives.

The earlier corrected archive remains exploratory: 420 old outcomes plus 120 new NaVILA outcomes and a post-hoc 1.5 m analysis. V3 is a prospective replication on familiar scenarios; it does not isolate the individual fixes or constitute held-out validation.

## Preparation required before launch

Local v3 support is implemented in the shared launch hook, scenario validation, manifests, result acceptance, goal identity requirements and physical-contact termination. V2 compatibility is retained. V3 dry-run tests cover all three policies and scenarios with fail-on-launch command sentinels; synthetic acceptance covers both versions. Navigation tests verify 1.5 m latching followed by continued approach to 0.5 m and physical-contact failure overriding prior success.

Before real launch: deploy matching simulator sources and v3 scenarios to /home/isaac/Projects/TIC-VLA, verify source parity through preflight, and freeze actual runtime provenance. All three policy drivers use that remote simulator repository; local policy inference remains in the local policy repositories. No remote NaVILA update is required by these v3 changes. No deployment or simulation was performed here. The existing PROTOCOL_V2_APPROVED variable is retained as a compatibility authorization switch for both versions.

Known unrelated check: the existing NaVILA normal-mode launcher equality test expects empty steering flags but receives --visualization-save-interval 2.0. This was observed during validation; no visualization settings were changed here.

The benchmark_results directory is Git-ignored; these files are local unless deliberately archived elsewhere.


## Launch from the local policy workstation

Run `bash scripts/run_crossover_v3.bash --dry-run` to validate all policies/scenarios without launching. After remote synchronization, run `bash scripts/run_crossover_v3.bash --run` to start the suite. This script uses sibling local policy repositories and the remote Isaac simulator; it is not intended to start inference on Isaac.
