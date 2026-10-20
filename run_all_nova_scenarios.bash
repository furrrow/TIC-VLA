#!/usr/bin/env bash
set -euo pipefail

PROJECTS_ROOT="${PROJECTS_ROOT:-/home/jim/Projects}"
BATCH_PREFIX="${BATCH_PREFIX:-steernav}"
DEFAULT_SCENARIO_DIR="DynaNav/configs/steernav_scenarios"
DEFAULT_SCENARIOS=(
  "${DEFAULT_SCENARIO_DIR}/scenario1_easy_hospital_20trials.yaml"
  "${DEFAULT_SCENARIO_DIR}/scenario2_challenging_hospital_20trials.yaml"
  "${DEFAULT_SCENARIO_DIR}/scenario3_transfer_warehouse_20trials.yaml"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [run_nova_scenarios arguments]

Run the same Nova scenarios sequentially in OmniVLA, TIC-VLA, and NaVILA.
With no arguments, runs the first three SteerNav scenarios in ablation mode.

Environment:
  BATCH_PREFIX   Batch prefix used before each project suffix (default: steernav)
  PROJECTS_ROOT  Directory containing the three repositories

Examples:
  $(basename "$0")
  $(basename "$0") --fix-end --mode ablation \
    ${DEFAULT_SCENARIO_DIR}/scenario3_transfer_warehouse_20trials.yaml
  BATCH_PREFIX=pilot $(basename "$0") --mode steered \
    ${DEFAULT_SCENARIO_DIR}/scenario3_transfer_warehouse_20trials.yaml
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
  "OmniVLA:omnivla"
  "TIC-VLA:ticvla"
  "NaVILA:navila"
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
