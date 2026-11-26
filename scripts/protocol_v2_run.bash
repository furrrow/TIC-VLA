# Hooks for the existing experiment driver, not an alternative orchestrator.
CROSSOVER_TOOLS_ROOT="${CROSSOVER_TOOLS_ROOT:-/home/jim/Projects/TIC-VLA}"
PROTOCOL_V2=false
if [[ "$SCENARIO" == DynaNav/configs/crossover_scenarios_v[23]/* ]]; then
  PROTOCOL_V2=true
  export OBJECT_VELOCITY_SOURCE="${OBJECT_VELOCITY_SOURCE:-estimated}"
  export STEERNAV_TIME_LOOK_AHEAD="${STEERNAV_TIME_LOOK_AHEAD:-1.5}"
  export STEERNAV_VELOCITY_WINDOW="${STEERNAV_VELOCITY_WINDOW:-5}"
  export STEERNAV_ODOM_ALIGNMENT="${STEERNAV_ODOM_ALIGNMENT:-extrapolate}"
  export STEERNAV_ODOM_POSE_MAX_EXTRAPOLATION="${STEERNAV_ODOM_POSE_MAX_EXTRAPOLATION:-0.25}"
  export STEERNAV_IMAGE_QUEUE_DEPTH="${STEERNAV_IMAGE_QUEUE_DEPTH:-1}"
  export STEERNAV_PROCESS_REPEATED_IMAGES="${STEERNAV_PROCESS_REPEATED_IMAGES:-false}"
  export STEERNAV_TAPER_PEDESTRIAN_SWEEP="${STEERNAV_TAPER_PEDESTRIAN_SWEEP:-false}"
  export STEERNAV_ROBOT_RADIUS="${STEERNAV_ROBOT_RADIUS:-0.5}"
  campaign_clearance=0.5
  if [[ "$INFERENCE_METHOD" == NaVILA ]]; then campaign_clearance=0.25; fi
  export STEERNAV_ADDITIONAL_CLEARANCE="${STEERNAV_ADDITIONAL_CLEARANCE:-$campaign_clearance}"
  python3 "$CROSSOVER_TOOLS_ROOT/scripts/protocol_v2.py" prepare --dry-run \
    --scenario "$CROSSOVER_TOOLS_ROOT/$SCENARIO" --mode "$MODE" --policy "$INFERENCE_METHOD" \
    --dt "$CONTROLLER_DT" --fix-end "$FIX_END"
  if [[ "${DRY_RUN:-false}" == true ]]; then exit 0; fi
  [[ "${PROTOCOL_V2_APPROVED:-false}" == true ]] || { echo 'Protocol v2 requires explicit user approval before simulator execution.' >&2; exit 2; }
fi
if [[ "${DRY_RUN:-false}" == true ]]; then echo 'Dry run supports protocol v2/v3 scenarios only.' >&2; exit 2; fi
export BATCH_ID RUN_ATTEMPT_ID

prepare_run_capture() {
  # Dynamic scope supplies mode, run_id and run_log_dir from run_mode.
  local mode="$1"
  [[ ! -e "$run_log_dir" ]] || { echo "Refusing existing capture directory: $run_log_dir" >&2; return 1; }
  if "$PROTOCOL_V2"; then
    (cd "$CROSSOVER_TOOLS_ROOT" && python3 scripts/protocol_v2.py prepare \
      --scenario "$SCENARIO" --capture "$run_log_dir" --mode "$mode" --policy "$INFERENCE_METHOD" \
      --run-id "$run_id" --host "$REMOTE_HOST" --remote-repo "$REMOTE_REPO" --dt "$CONTROLLER_DT" --fix-end "$FIX_END")
  else
    mkdir -p "$run_log_dir"
  fi
  RECOVERY_REMOTE_ARGS=""
  RECOVERY_REMOTE_GUARD=""
  if [[ -n "${PROTOCOL_V2_RECOVERY_EPISODE:-}" ]]; then
    [[ "$PROTOCOL_V2" == true && "$PROTOCOL_V2_RECOVERY_EPISODE" =~ ^episode_[A-Za-z0-9_]+$ ]] || return 2
    RECOVERY_REMOTE_GUARD="test ! -e /tmp/${run_id}_recovery.json && "
    RECOVERY_REMOTE_ARGS=" --child --episode_name ${PROTOCOL_V2_RECOVERY_EPISODE} --result_json /tmp/${run_id}_recovery.json"
  fi
  # Deliberately replace any global timing path for every mode.
  export STEERNAV_RECEIVER_TIMING_LOG='' STEERNAV_VELOCITY_REFERENCE_LOG=false
  if [[ "$mode" == steered && "${OBJECT_VELOCITY_SOURCE:-estimated}" == estimated ]]; then
    export STEERNAV_RECEIVER_TIMING_LOG="${run_log_dir}/receiver_timing.jsonl"
    export STEERNAV_VELOCITY_REFERENCE_LOG=true
  fi
}

finish_run_capture() {
  "$PROTOCOL_V2" || return 0
  local mode="$1" source="${REMOTE_REPO}/DynaNav/benchmark_results/${run_id}/$(basename "$SCENARIO" .yaml)_results_latest.json"
  # Never consume a result from another attempt. Do not trust the remote exit code.
  if [[ -n "${PROTOCOL_V2_RECOVERY_EPISODE:-}" ]]; then
    source="/tmp/${run_id}_recovery.json"
    scp -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST:$source" "$run_log_dir/raw_episode_result.json" || return 1
    python3 "$CROSSOVER_TOOLS_ROOT/scripts/recover_crossover_episodes.py" wrap --folder "$run_log_dir" || return 1
  else
    scp -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST:$source" "$run_log_dir/benchmark_results.json" || true
  fi
  local validation_status=0
  python3 "$CROSSOVER_TOOLS_ROOT/scripts/protocol_v2.py" finish --capture "$run_log_dir" \
    --mode "$mode" --policy "$INFERENCE_METHOD" --run-id "$run_id" || validation_status=$?
  if [[ "$validation_status" == 10 && "${CONTINUE_ON_EPISODE_CRASH:-false}" == true ]]; then
    echo "INCOMPLETE: isolated Isaac crash(es) recorded in $run_log_dir/recovery_needed.json; continuing scheduling." >&2
    return 0
  fi
  return "$validation_status"
}
