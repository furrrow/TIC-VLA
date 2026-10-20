#!/usr/bin/env bash
set -euo pipefail

PROJECTS_ROOT="${PROJECTS_ROOT:-/home/jim/Projects}"
BATCH_PREFIX="${BATCH_PREFIX:-scn3}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [run_nova_scenarios arguments]

Run the same Nova scenarios sequentially in OmniVLA, TIC-VLA, and NaVILA.
With no arguments, runs: --mode ablation scenario4_5

Environment:
  BATCH_PREFIX   Batch prefix used before each project suffix (default: scn3)
  PROJECTS_ROOT  Directory containing the three repositories

Examples:
  $(basename "$0")
  BATCH_PREFIX=sn4 $(basename "$0") --mode ablation scenario4_5
EOF
}

if (( $# == 1 )) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

if (( $# == 0 )); then
  ARGS=(--mode ablation scenario4_5)
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
