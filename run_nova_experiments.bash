#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-isaac}"
REMOTE_REPO="${REMOTE_REPO:-/home/isaac/Projects/TIC-VLA}"
REMOTE_DISPLAY="${REMOTE_DISPLAY:-:1}"
SCENARIO="DynaNav/configs/icra_scenarios_v3/scenario3_5.yaml"
MODE="ablation"
CONTROLLER_DT="0.75"
TIME_LOOK_AHEAD="1.0"
INFERENCE_METHOD="${INFERENCE_METHOD:-TIC-VLA}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-600}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/logs/orchestration/${BATCH_ID}"
RUN_ATTEMPT_ID="${RUN_ATTEMPT_ID:-$(date +%Y%m%d_%H%M%S)_$$}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mode normal|steered|no-velocity|both|ablation] [--scenario FILE] [--inference-method NAME] [--time-look-ahead SECONDS]

Run one experiment mode, both existing modes, or an ablation of normal,
steered, and steered without velocity compensation.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --mode)
      if (( $# < 2 )); then
        echo "--mode requires one of: normal, steered, no-velocity, both, ablation." >&2
        usage >&2
        exit 2
      fi
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    -s|--scenario)
      if (( $# < 2 )); then
        echo "$1 requires a scenario file." >&2
        usage >&2
        exit 2
      fi
      SCENARIO="$2"
      shift 2
      ;;
    --scenario=*)
      SCENARIO="${1#*=}"
      shift
      ;;
    --inference-method)
      if (( $# < 2 )); then
        echo "--inference-method requires a model name." >&2
        usage >&2
        exit 2
      fi
      INFERENCE_METHOD="$2"
      shift 2
      ;;
    --inference-method=*)
      INFERENCE_METHOD="${1#*=}"
      shift
      ;;
    --time-look-ahead)
      if (( $# < 2 )); then
        echo "--time-look-ahead requires a value." >&2
        usage >&2
        exit 2
      fi
      TIME_LOOK_AHEAD="$2"
      shift 2
      ;;
    --time-look-ahead=*)
      TIME_LOOK_AHEAD="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${MODE}" != "normal" && "${MODE}" != "steered" && "${MODE}" != "no-velocity" && "${MODE}" != "both" && "${MODE}" != "ablation" ]]; then
  echo "Invalid mode '${MODE}'; expected normal, steered, no-velocity, both, or ablation." >&2
  exit 2
fi

if [[ ! "${SCENARIO}" =~ ^[A-Za-z0-9_./-]+\.yaml$ ]]; then
  echo "Scenario must be a relative YAML path containing only letters, numbers, _, -, ., and /." >&2
  exit 2
fi
if [[ ! "${REMOTE_REPO}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
  echo "REMOTE_REPO must be an absolute path containing only letters, numbers, _, -, ., and /." >&2
  exit 2
fi
if [[ ! "${REMOTE_DISPLAY}" =~ ^:[0-9]+(\.[0-9]+)?$ ]]; then
  echo "REMOTE_DISPLAY must be an X display such as :0 or :1." >&2
  exit 2
fi
if [[ ! "${BATCH_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "BATCH_ID may contain only letters, numbers, dots, underscores, and hyphens." >&2
  exit 2
fi
if [[ ! "${RUN_ATTEMPT_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_ATTEMPT_ID may contain only letters, numbers, dots, underscores, and hyphens." >&2
  exit 2
fi
if [[ ! "${INFERENCE_METHOD}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Inference method may contain only letters, numbers, dots, underscores, and hyphens." >&2
  exit 2
fi
if [[ ! "${READY_TIMEOUT_SECONDS}" =~ ^[0-9]+$ || "${READY_TIMEOUT_SECONDS}" -eq 0 ]]; then
  echo "READY_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if [[ ! "${TIME_LOOK_AHEAD}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "SteerNav time look-ahead must be a non-negative number of seconds." >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
active_session=""
active_remote_job=""

stop_remote_job() {
  if [[ -z "${active_remote_job}" ]]; then
    return 0
  fi
  echo "Stopping remote simulator job ${active_remote_job} on ${REMOTE_HOST}..."
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" \
    "cd ${REMOTE_REPO} && ./run_sim.bash --stop ${active_remote_job}" || \
    echo "Warning: could not confirm that remote simulator job ${active_remote_job} stopped." >&2
  active_remote_job=""
}

cleanup_session() {
  stop_remote_job
  if [[ -n "${active_session}" ]] && tmux has-session -t "${active_session}" 2>/dev/null; then
    echo "Stopping local tmux session ${active_session}"
    tmux kill-session -t "${active_session}"
  fi
  active_session=""
}
trap cleanup_session EXIT
trap 'exit 130' INT TERM

wait_until_ready() {
  local mode="$1"
  local session="$2"
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  local pane_output=""
  local pane_id=""

  echo "Waiting up to ${READY_TIMEOUT_SECONDS}s for local ${mode} nodes to initialize..."
  while (( SECONDS < deadline )); do
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      echo "Local tmux session exited before becoming ready: ${session}" >&2
      return 1
    fi
    pane_output=""
    while IFS= read -r pane_id; do
      pane_output+="$(tmux capture-pane -p -S -200 -t "${pane_id}" 2>/dev/null || true)"
      pane_output+=$'\n'
    done < <(tmux list-panes -t "${session}:0" -F '#{pane_id}')
    if [[ "${pane_output}" == *"Model ready."* && "${pane_output}" == *"Registered with master node."* ]]; then
      if [[ "${mode}" == "normal" || "${pane_output}" == *"Waiting for simulation episodes"* ]]; then
        echo "Local ${mode} pipeline is ready."
        return 0
      fi
    fi
    sleep 2
  done

  echo "Timed out waiting for local ${mode} pipeline. Recent pane output:" >&2
  while IFS= read -r pane_id; do
    tmux capture-pane -p -S -100 -t "${pane_id}" >&2 || true
  done < <(tmux list-panes -t "${session}:0" -F '#{pane_id}')
  return 1
}

run_mode() {
  local mode="$1"
  local run_id="${BATCH_ID}_${mode}"
  # tmux uses dots when parsing session:window.pane targets, so keep them out of
  # the internal session name. Preserve the original run ID in result/log paths.
  local session="nova_${run_id//./_}_${RUN_ATTEMPT_ID//./_}"
  local run_log_dir="${LOG_DIR}/${run_id}/${RUN_ATTEMPT_ID}"
  local status_file="${run_log_dir}/.remote.status"
  local remote_command=""
  local remote_pane=""
  local watcher_pid=""
  local remote_status=""

  echo
  echo "=== ${mode}: run_id=${run_id} ==="
  mkdir -p "${run_log_dir}"
  # runs nova_ros.bash
  NOVA_ROS_LOG_ROOT="${run_log_dir}" \
    "${REPO_ROOT}/nova_ros.bash" "${mode}" --detach --session-name "${session}" --dt "${CONTROLLER_DT}" --time-look-ahead "${TIME_LOOK_AHEAD}"
  # Only clean up sessions successfully created by this invocation.
  active_session="${session}"
  wait_until_ready "${mode}" "${session}"

  echo "Starting simulator benchmark on ${REMOTE_HOST} in a visible tmux pane..."
  active_remote_job="${run_id}"
  remote_command="set -o pipefail; ssh -o BatchMode=yes -o ConnectTimeout=10 ${REMOTE_HOST} 'cd ${REMOTE_REPO} && DISPLAY=${REMOTE_DISPLAY} ./run_sim.bash --scenario ${SCENARIO} --job-id ${run_id} --inference-method ${INFERENCE_METHOD} -- --run_id ${run_id} --experiment_mode ${mode}' 2>&1 | tee ${run_log_dir}/remote.log; status=\${PIPESTATUS[0]}; echo \${status} > ${status_file}; exit \${status}"
  remote_pane="$(tmux split-window -v -P -F '#{pane_id}' -t "${session}:0" -c "${REPO_ROOT}" /bin/bash -lc "${remote_command}")"
  tmux select-pane -t "${remote_pane}" -T "Isaac ${mode} status"
  tmux select-layout -t "${session}:0" tiled

  (
    while tmux has-session -t "${session}" 2>/dev/null && [[ ! -s "${status_file}" ]]; do
      sleep 1
    done
    if tmux has-session -t "${session}" 2>/dev/null; then
      sleep 3
      tmux kill-session -t "${session}"
    fi
  ) &
  watcher_pid=$!

  echo "Attaching to ${session}; it will close shortly after the remote benchmark exits."
  if [[ -n "${TMUX:-}" ]]; then
    echo "Run this orchestrator outside an existing tmux client so it can display its experiment session." >&2
    tmux kill-session -t "${session}"
    wait "${watcher_pid}" || true
    return 1
  fi
  tmux attach-session -t "${session}" || true
  wait "${watcher_pid}" || true
  active_session=""

  if [[ ! -s "${status_file}" ]]; then
    echo "The ${mode} tmux session ended before the remote benchmark reported status." >&2
    return 1
  fi
  remote_status="$(<"${status_file}")"
  active_remote_job=""
  if [[ "${remote_status}" != "0" ]]; then
    echo "Remote ${mode} benchmark failed with status ${remote_status}." >&2
    return "${remote_status}"
  fi

  echo "Remote ${mode} benchmark completed."
  sleep 5
}

echo "Batch ID: ${BATCH_ID}"
echo "Run attempt: ${RUN_ATTEMPT_ID}"
echo "Mode: ${MODE}"
echo "Inference method: ${INFERENCE_METHOD}"
echo "SteerNav time look-ahead: ${TIME_LOOK_AHEAD}s"
echo "Scenario: ${SCENARIO}"
echo "Remote: ${REMOTE_HOST}:${REMOTE_REPO} (DISPLAY=${REMOTE_DISPLAY})"
echo "Local logs: ${LOG_DIR}"
ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" \
  "cd ${REMOTE_REPO} && DISPLAY=${REMOTE_DISPLAY} ./run_sim.bash --check --scenario ${SCENARIO}"

if [[ "${MODE}" == "steered" || "${MODE}" == "both" || "${MODE}" == "ablation" ]]; then
  run_mode steered
fi
if [[ "${MODE}" == "no-velocity" || "${MODE}" == "ablation" ]]; then
  run_mode no-velocity
fi
if [[ "${MODE}" == "normal" || "${MODE}" == "both" || "${MODE}" == "ablation" ]]; then
  run_mode normal
fi

echo
echo "Requested experiment mode '${MODE}' completed successfully."
if [[ "${MODE}" == "ablation" ]]; then
  echo "Remote run IDs: ${BATCH_ID}_normal, ${BATCH_ID}_steered, ${BATCH_ID}_no-velocity"
elif [[ "${MODE}" == "both" ]]; then
  echo "Remote run IDs: ${BATCH_ID}_normal, ${BATCH_ID}_steered"
else
  echo "Remote run ID: ${BATCH_ID}_${MODE}"
fi
echo "Local orchestration logs: ${LOG_DIR}"
