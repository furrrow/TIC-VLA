#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_CONFIG="${REPO_ROOT}/configs/robot.yaml"
ROS_SETUP="/opt/ros/humble/setup.bash"
STEERNAV_ROOT="${STEERNAV_ROOT:-/home/jim/Projects/steernav}"
MODE="${1:-normal}"

if [[ "${MODE}" != "normal" && "${MODE}" != "steered" ]]; then
  echo "Usage: $0 [normal|steered]" >&2
  exit 2
fi

echo "Mode: ${MODE}"
echo "Robot config: ${ROBOT_CONFIG}"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2
  exit 1
fi
if [[ "${MODE}" == "steered" && ! -f "${STEERNAV_ROOT}/steernav/ros_inference.py" ]]; then
  echo "SteerNav inference script not found under: ${STEERNAV_ROOT}" >&2
  echo "Set STEERNAV_ROOT to the SteerNav repository path." >&2
  exit 1
fi

session_name="nova_ros_$(date +%s)_$$"
tmux new-session -d -s "${session_name}" -c "${REPO_ROOT}"
inference_pane="$(tmux display-message -p -t "${session_name}:0.0" '#{pane_id}')"
controller_pane="$(tmux split-window -h -P -F '#{pane_id}' -t "${inference_pane}" -c "${REPO_ROOT}")"
tmux set-option -t "${session_name}" mouse on

tmux send-keys -t "${inference_pane}" \
  "source ${ROS_SETUP} && export ROS_DOMAIN_ID=42 && uv run infer_ros.py --robot sim_nova" C-m

controller_flags="--simulation-episode-control"
if [[ "${MODE}" == "steered" ]]; then
  controller_flags="--steer ${controller_flags}"
  steernav_pane="$(tmux split-window -v -P -F '#{pane_id}' -t "${controller_pane}" -c "${STEERNAV_ROOT}")"
  tmux send-keys -t "${steernav_pane}" \
    "source ${ROS_SETUP} && export ROS_DOMAIN_ID=42 && uv run steernav/ros_inference.py --robot sim_nova" C-m
  tmux select-layout -t "${session_name}:0" tiled
fi

tmux send-keys -t "${controller_pane}" \
  "source ${ROS_SETUP} && export ROS_DOMAIN_ID=42 && uv run custom_utils/custom_utils/kinematic_controller_naive.py --robot sim_nova --config ${ROBOT_CONFIG} ${controller_flags}" C-m

tmux select-pane -t "${inference_pane}"
if [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "${session_name}"
else
  tmux attach-session -t "${session_name}"
fi
