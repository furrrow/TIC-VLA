#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2
  exit 1
fi

session_name="nova_ros_$(date +%s)_$$"
tmux new-session -d -s "${session_name}" -c "${REPO_ROOT}"
tmux split-window -h -t "${session_name}:0" -c "${REPO_ROOT}"
tmux set-option -t "${session_name}" mouse on

tmux send-keys -t "${session_name}:0.0" \
  "source ${ROS_SETUP} && export ROS_DOMAIN_ID=42 && uv run infer_ros.py --robot sim_nova" C-m
tmux send-keys -t "${session_name}:0.1" \
  "source ${ROS_SETUP} && export ROS_DOMAIN_ID=42 && uv run custom_utils/custom_utils/kinematic_controller_naive.py --robot sim_nova --config configs/robot.yaml" C-m

tmux select-pane -t "${session_name}:0.0"
if [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "${session_name}"
else
  tmux attach-session -t "${session_name}"
fi
