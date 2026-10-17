#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
REMOTE_HOST="${REMOTE_HOST:-isaac}"
REMOTE_REPO="${REMOTE_REPO:-/home/isaac/Projects/TIC-VLA}"
REMOTE_DISPLAY="${REMOTE_DISPLAY:-:1}"
CONFIG="DynaNav/configs/icra_scenarios_v4/scenario3_5.yaml"
MODE="both"
CONTROLLER_DT="0.75"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-600}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/logs/orchestration/${BATCH_ID}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mode normal|steered|both] [CONFIG]

Run the normal experiment, the steered experiment, or both sequentially.
EOF
}

config_set=false
while (( $# > 0 )); do
  case "$1" in
    --mode)
      if (( $# < 2 )); then
        echo "--mode requires one of: normal, steered, both." >&2
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
      if [[ "${config_set}" == true ]]; then
        echo "Only one CONFIG path may be supplied." >&2
        usage >&2
        exit 2
      fi
      CONFIG="$1"
      config_set=true
      shift
      ;;
  esac
done

if [[ "${MODE}" != "normal" && "${MODE}" != "steered" && "${MODE}" != "both" ]]; then
  echo "Invalid mode '${MODE}'; expected normal, steered, or both." >&2
  exit 2
fi

if [[ ! "${CONFIG}" =~ ^[A-Za-z0-9_./-]+\.yaml$ ]]; then
  echo "Config must be a relative YAML path containing only letters, numbers, _, -, ., and /." >&2
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
display_number="${REMOTE_DISPLAY#:}"
display_number="${display_number%%.*}"
if [[ ! "${BATCH_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "BATCH_ID may contain only letters, numbers, dots, underscores, and hyphens." >&2
  exit 2
fi
if [[ ! "${READY_TIMEOUT_SECONDS}" =~ ^[0-9]+$ || "${READY_TIMEOUT_SECONDS}" -eq 0 ]]; then
  echo "READY_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
active_session=""

cleanup_session() {
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
  local session="nova_${run_id}"
  local status_file="${LOG_DIR}/.${mode}.$$.status"
  local remote_command=""
  local remote_pane=""
  local watcher_pid=""
  local remote_status=""

  echo
  echo "=== ${mode}: run_id=${run_id} ==="
  active_session="${session}"
  # runs nova_ros.bash
  "${REPO_ROOT}/nova_ros.bash" "${mode}" --detach --session-name "${session}" --dt "${CONTROLLER_DT}"
  wait_until_ready "${mode}" "${session}"

  echo "Starting simulator benchmark on ${REMOTE_HOST} in a visible tmux pane..."
  remote_command="set -o pipefail; ssh -o BatchMode=yes -o ConnectTimeout=10 ${REMOTE_HOST} 'cd ${REMOTE_REPO} && export DISPLAY=${REMOTE_DISPLAY} && export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-} && DynaNav/run_benchmark.sh ${CONFIG} --run_id ${run_id}' 2>&1 | tee ${LOG_DIR}/${mode}.log; status=\${PIPESTATUS[0]}; echo \${status} > ${status_file}; exit \${status}"
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
  if [[ "${remote_status}" != "0" ]]; then
    echo "Remote ${mode} benchmark failed with status ${remote_status}." >&2
    return "${remote_status}"
  fi

  echo "Remote ${mode} benchmark completed."
  sleep 5
}

echo "Batch ID: ${BATCH_ID}"
echo "Mode: ${MODE}"
echo "Scenario: ${CONFIG}"
echo "Remote: ${REMOTE_HOST}:${REMOTE_REPO} (DISPLAY=${REMOTE_DISPLAY})"
echo "Local logs: ${LOG_DIR}"
# preflight check for remote machine:
ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" \
  "cd ${REMOTE_REPO} && { test -S /tmp/.X11-unix/X${display_number} || { echo 'ERROR: Remote X display ${REMOTE_DISPLAY} is unavailable. Check whether the remote machine is logged in to a graphical desktop session.' >&2; exit 1; }; } && test -f ${CONFIG} && test -x DynaNav/run_benchmark.sh && export DISPLAY=${REMOTE_DISPLAY} && export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-} && bash -u -c 'source .env.testing'"

if [[ "${MODE}" == "normal" || "${MODE}" == "both" ]]; then
  run_mode normal
fi
if [[ "${MODE}" == "steered" || "${MODE}" == "both" ]]; then
  run_mode steered
fi

echo
echo "Requested experiment mode '${MODE}' completed successfully."
if [[ "${MODE}" == "both" ]]; then
  echo "Remote run IDs: ${BATCH_ID}_normal, ${BATCH_ID}_steered"
else
  echo "Remote run ID: ${BATCH_ID}_${MODE}"
fi
echo "Local orchestration logs: ${LOG_DIR}"
