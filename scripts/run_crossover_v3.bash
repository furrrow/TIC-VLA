#!/usr/bin/env bash
# Prospective v3 entry point; see docs/crossover_protocol_v3.md.
set -euo pipefail
PROJECTS=/home/jim/Projects
ROOT="$PROJECTS/TIC-VLA"
ACTION="${1:---dry-run}"
case "$ACTION" in --dry-run|--run) ;; *) echo 'Usage: run_suite.bash [--dry-run|--run]' >&2; exit 2 ;; esac
SCENARIOS=()
for level in easy medium hard; do
  relative="DynaNav/configs/crossover_scenarios_v3/scenario3_crossover_${level}_20trials.yaml"
  [[ -f "$ROOT/$relative" ]] || { echo "V3 preparation incomplete: missing $ROOT/$relative (see docs/crossover_protocol_v3.md)." >&2; exit 2; }
  SCENARIOS+=("$relative")
done
export CROSSOVER_TOOLS_ROOT="$ROOT"
export RUN_ATTEMPT_ID=attempt1
CAMPAIGN_ID="crossover_v3_$(date -u +%Y%m%d_%H%M%S)"
# All policies must pass read-only validation before starting the first policy.
for policy in OmniVLA NaVILA TIC-VLA; do
  BATCH_ID="${CAMPAIGN_ID}_${policy,,}" bash "$PROJECTS/$policy/run_nova_scenarios.bash" \
    --mode ablation --no-fix-end --dry-run "${SCENARIOS[@]}"
done
[[ "$ACTION" == --run ]] || exit 0
export PROTOCOL_V2_APPROVED=true # Existing driver's authorization variable; retain until version-aware implementation.
for policy in OmniVLA NaVILA TIC-VLA; do
  BATCH_ID="${CAMPAIGN_ID}_${policy,,}" bash "$PROJECTS/$policy/run_nova_scenarios.bash" \
    --mode ablation --no-fix-end "${SCENARIOS[@]}"
done
