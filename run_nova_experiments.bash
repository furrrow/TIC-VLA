#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
REMOTE_HOST="${REMOTE_HOST:-isaac}"
REMOTE_REPO="${REMOTE_REPO:-/home/isaac/Projects/TIC-VLA}"
CONFIG="${1:-DynaNav/configs/icra_scenarios_v4/scenario1_5.yaml}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-600}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/logs/orchestration/${BATCH_ID}"

if [[ ! "${CONFIG}" =~ ^[A-Za-z0-9_./-]+\.yaml$ ]]; then
  echo "Config must be a relative YAML path containing only letters, numbers, _, -, ., and /." >&2
  exit 2
fi
if [[ ! "${REMOTE_REPO}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
  echo "REMOTE_REPO must be an absolute path containing only letters, numbers, _, -, ., and /." >&2
  exit 2
fi
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

  echo
  echo "=== ${mode}: run_id=${run_id} ==="
  active_session="${session}"
  "${REPO_ROOT}/nova_ros.bash" "${mode}" --detach --session-name "${session}"
  wait_until_ready "${mode}" "${session}"

  echo "Starting simulator benchmark on ${REMOTE_HOST}..."
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" \
    "cd ${REMOTE_REPO} && DynaNav/run_benchmark.sh ${CONFIG} --run_id ${run_id}" \
    2>&1 | tee "${LOG_DIR}/${mode}.log"

  echo "Remote ${mode} benchmark completed."
  cleanup_session
  sleep 5
}

echo "Batch ID: ${BATCH_ID}"
echo "Scenario: ${CONFIG}"
echo "Remote: ${REMOTE_HOST}:${REMOTE_REPO}"
echo "Local logs: ${LOG_DIR}"

ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" \
  "cd ${REMOTE_REPO} && test -f ${CONFIG} && test -x DynaNav/run_benchmark.sh"

run_mode normal
run_mode steered

echo
echo "Both experiments completed successfully."
echo "Remote run IDs: ${BATCH_ID}_normal, ${BATCH_ID}_steered"
echo "Local orchestration logs: ${LOG_DIR}"
