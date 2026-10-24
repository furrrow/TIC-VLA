#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_CONFIG="${REPO_ROOT}/configs/robot.yaml"
ROS_SETUP="/opt/ros/humble/setup.bash"
STEERNAV_ROOT="${STEERNAV_ROOT:-/home/jim/Projects/steernav}"
MODE="ablation"
DETACH=false
SESSION_NAME=""
CONTROLLER_DT="0.75"
TIME_LOOK_AHEAD="1.0"
FIX_END=false
LOG_ROOT="${NOVA_ROS_LOG_ROOT:-${REPO_ROOT}/logs/nova_ros}"

if [[ $# -gt 0 && ( "$1" == "normal" || "$1" == "steered" || "$1" == "no-velocity" ) ]]; then
  MODE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      DETACH=true
      shift
      ;;
    --session-name)
      if [[ $# -lt 2 ]]; then
        echo "--session-name requires a value" >&2
        exit 2
      fi
      SESSION_NAME="$2"
      shift 2
      ;;
    --dt)
      if [[ $# -lt 2 ]]; then
        echo "--dt requires a value" >&2
        exit 2
      fi
      CONTROLLER_DT="$2"
      shift 2
      ;;
    --time-look-ahead)
      if [[ $# -lt 2 ]]; then
        echo "--time-look-ahead requires a value" >&2
        exit 2
      fi
      TIME_LOOK_AHEAD="$2"
      shift 2
      ;;
    --time-look-ahead=*)
      TIME_LOOK_AHEAD="${1#*=}"
      shift
      ;;
    --fix-end)
      FIX_END=true
      shift
      ;;
    --no-fix-end)
      FIX_END=false
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [normal|steered|no-velocity] [--detach] [--session-name NAME] [--dt SECONDS] [--time-look-ahead SECONDS] [--fix-end|--no-fix-end]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [normal|steered|no-velocity] [--detach] [--session-name NAME] [--dt SECONDS] [--time-look-ahead SECONDS] [--fix-end|--no-fix-end]" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${SESSION_NAME}" && ! "${SESSION_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Session name may contain only letters, numbers, dots, underscores, and hyphens." >&2
  exit 2
fi
if [[ ! "${CONTROLLER_DT}" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ ! "${CONTROLLER_DT}" =~ [1-9] ]]; then
  echo "Controller dt must be a positive number of seconds." >&2
  exit 2
fi
if [[ ! "${TIME_LOOK_AHEAD}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "SteerNav time look-ahead must be a non-negative number of seconds." >&2
  exit 2
fi

echo "Mode: ${MODE}"
echo "Robot config: ${ROBOT_CONFIG}"
echo "Controller dt: ${CONTROLLER_DT}s"
echo "SteerNav time look-ahead: ${TIME_LOOK_AHEAD}s"
echo "SteerNav fix end: ${FIX_END}"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2
  exit 1
fi
if [[ "${MODE}" != "normal" && ! -f "${STEERNAV_ROOT}/steernav/ros_inference.py" ]]; then
  echo "SteerNav inference script not found under: ${STEERNAV_ROOT}" >&2
  echo "Set STEERNAV_ROOT to the SteerNav repository path." >&2
  exit 1
fi

session_name="${SESSION_NAME:-nova_ros_$(date +%s)_$$}"
if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  exit 1
fi

log_dir="${LOG_ROOT}/${session_name}"
mkdir -p "${log_dir}"
inference_log="${log_dir}/inference.log"
controller_log="${log_dir}/controller.log"
steernav_log="${log_dir}/steernav.log"

run_in_pane() {
  local pane="$1"
  local log_file="$2"
  local command="$3"
  local quoted_log
  printf -v quoted_log '%q' "${log_file}"
  tmux send-keys -t "${pane}" \
    "set -o pipefail; export PYTHONUNBUFFERED=1; ( ${command} ) 2>&1 | tee -a ${quoted_log}" C-m
}

# set up tmux session
tmux new-session -d -s "${session_name}" -c "${REPO_ROOT}"
inference_pane="$(tmux display-message -p -t "${session_name}:0.0" '#{pane_id}')"
controller_pane="$(tmux split-window -h -P -F '#{pane_id}' -t "${inference_pane}" -c "${REPO_ROOT}")"
tmux set-option -t "${session_name}" mouse on
tmux select-pane -t "${inference_pane}" -T "TIC-VLA inference"
tmux select-pane -t "${controller_pane}" -T "Waypoint controller"

# run inference for TICVLA
run_in_pane "${inference_pane}" "${inference_log}" \
  "source ${ROS_SETUP} && export ROS_DOMAIN_ID=42 && uv run infer_ros.py --robot sim_nova --ros-args -p use_sim_time:=true"

# Run the steering node in both steered modes. The no-velocity variant disables
# object detection and velocity compensation inside SteerNav.
controller_flags="--simulation-episode-control"
if [[ "${MODE}" != "normal" ]]; then
  controller_flags="--steer ${controller_flags}"
#  steernav_flags="--debug-motion"
  steernav_flags="--time-look-ahead ${TIME_LOOK_AHEAD}"
  if [[ "${MODE}" != "no-velocity" ]]; then
    steernav_flags="${steernav_flags} --object-velocity-source sim-ground-truth"
  fi
  if [[ "${FIX_END}" == true ]]; then
    steernav_flags="${steernav_flags} --fix-end"
  fi
  if [[ "${MODE}" == "no-velocity" ]]; then
    steernav_flags="${steernav_flags} --no-velocity"
  fi
  steernav_pane="$(tmux split-window -v -P -F '#{pane_id}' -t "${controller_pane}" -c "${STEERNAV_ROOT}")"
  tmux select-pane -t "${steernav_pane}" -T "SteerNav"
  run_in_pane "${steernav_pane}" "${steernav_log}" \
    "source ${ROS_SETUP} && export ROS_DOMAIN_ID=42 && uv run steernav/ros_inference.py --robot sim_nova ${steernav_flags} --ros-args -p use_sim_time:=true"
  tmux select-layout -t "${session_name}:0" tiled
fi

# Run the controller with --steer in both steering modes.
run_in_pane "${controller_pane}" "${controller_log}" \
  "source ${ROS_SETUP} && export ROS_DOMAIN_ID=42 && uv run custom_utils/custom_utils/kinematic_controller_naive.py --robot sim_nova --config ${ROBOT_CONFIG} --dt ${CONTROLLER_DT} ${controller_flags} --ros-args -p use_sim_time:=true"

tmux select-pane -t "${inference_pane}"
echo "TMUX_SESSION=${session_name}"
echo "LOG_DIR=${log_dir}"
if [[ "${DETACH}" == true ]]; then
  exit 0
elif [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "${session_name}"
else
  tmux attach-session -t "${session_name}"
fi
