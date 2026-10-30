#!/usr/bin/env bash
set -euo pipefail

PROJECTS_ROOT="${PROJECTS_ROOT:-/home/jim/Projects}"
BATCH_PREFIX="${BATCH_PREFIX:-steernav}"
DEFAULT_SCENARIO_DIR="DynaNav/configs/steernav_scenarios"
DEFAULT_SCENARIOS=(
  "${DEFAULT_SCENARIO_DIR}/scenario3_crossover_hard_20trials.yaml"
  "${DEFAULT_SCENARIO_DIR}/scenario3_crossover_medium_20trials.yaml"
  "${DEFAULT_SCENARIO_DIR}/scenario3_crossover_easy_20trials.yaml"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [run_nova_scenarios arguments]

Run the same Nova scenarios sequentially in NaVILA, TIC-VLA, and OmniVLA.
With no arguments, runs the hard, medium, and easy Hospital scenarios in
ablation mode. SteerNav's final waypoint is free by default; pass --fix-end to
hold it fixed during trajectory optimization.

Environment:
  BATCH_PREFIX   Batch prefix used before each project suffix (default: steernav)
  PROJECTS_ROOT  Directory containing the three repositories

Examples:
  $(basename "$0")
  $(basename "$0") --mode ablation ${DEFAULT_SCENARIO_DIR}/scenario3_hard_20trials.yaml
  $(basename "$0") --fix-end --mode ablation ${DEFAULT_SCENARIO_DIR}/scenario3_hard_20trials.yaml
  BATCH_PREFIX=pilot $(basename "$0") --mode steered ${DEFAULT_SCENARIO_DIR}/scenario3_medium_20trials.yaml
EOF
}

if (( $# == 1 )) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

if (( $# == 0 )); then
  ARGS=(--mode ablation "${DEFAULT_SCENARIOS[@]}")
else
  ARGS=("$@")
fi

PROJECTS=(
  "NaVILA:navila"
  "TIC-VLA:ticvla"
  "OmniVLA:omnivla"
)

for project_spec in "${PROJECTS[@]}"; do
  project_name="${project_spec%%:*}"
  batch_suffix="${project_spec#*:}"
  launcher="${PROJECTS_ROOT}/${project_name}/run_nova_scenarios.bash"

  if [[ ! -x "${launcher}" ]]; then
    echo "Missing executable launcher: ${launcher}" >&2
    exit 2
  fi

  echo
  echo "### ${project_name}: batch ${BATCH_PREFIX}_${batch_suffix} ###"
  BATCH_ID="${BATCH_PREFIX}_${batch_suffix}" "${launcher}" "${ARGS[@]}"
done

echo
echo "All three Nova scenario batches completed successfully."
