#!/usr/bin/env bash
set -e

source .env.testing

# SSH sessions do not inherit the simulator desktop's display. Default to the
# active local X session so Isaac Sim opens visibly on the simulator machine.
# Set DISPLAY before invoking this script if the desktop uses another display.
export DISPLAY="${DISPLAY:-:1}"
echo "Launching Isaac Sim on DISPLAY=${DISPLAY}"

#DynaNav/run_benchmark.sh DynaNav/configs/farther_goals_50_v1_scn1_50trials.yaml
DynaNav/run_benchmark.sh DynaNav/configs/icra_scenarios_v4/all_scenarios_15.yaml
