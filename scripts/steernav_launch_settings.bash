# Common downstream settings; sourced before tmux by all three policy launchers.
fail_setting() { echo "$*" >&2; exit 2; }
OBJECT_VELOCITY_SOURCE="${OBJECT_VELOCITY_SOURCE:-estimated}"
STEERNAV_TIME_LOOK_AHEAD="${STEERNAV_TIME_LOOK_AHEAD:-1.5}"
STEERNAV_VELOCITY_WINDOW="${STEERNAV_VELOCITY_WINDOW:-5}"
STEERNAV_ODOM_ALIGNMENT="${STEERNAV_ODOM_ALIGNMENT:-extrapolate}"
STEERNAV_ODOM_POSE_MAX_EXTRAPOLATION="${STEERNAV_ODOM_POSE_MAX_EXTRAPOLATION:-0.25}"
STEERNAV_IMAGE_QUEUE_DEPTH="${STEERNAV_IMAGE_QUEUE_DEPTH:-1}"
STEERNAV_PROCESS_REPEATED_IMAGES="${STEERNAV_PROCESS_REPEATED_IMAGES:-false}"
STEERNAV_TAPER_PEDESTRIAN_SWEEP="${STEERNAV_TAPER_PEDESTRIAN_SWEEP:-false}"
STEERNAV_ROBOT_RADIUS="${STEERNAV_ROBOT_RADIUS:-0.5}"
STEERNAV_ADDITIONAL_CLEARANCE="${STEERNAV_ADDITIONAL_CLEARANCE:-0.5}"
STEERNAV_VELOCITY_REFERENCE_LOG="${STEERNAV_VELOCITY_REFERENCE_LOG:-false}"
STEERNAV_RECEIVER_TIMING_LOG="${STEERNAV_RECEIVER_TIMING_LOG:-}"
steernav_settings_flags=""
[[ "$OBJECT_VELOCITY_SOURCE" == estimated || "$OBJECT_VELOCITY_SOURCE" == sim-ground-truth ]] || fail_setting 'Invalid velocity source'
[[ "$MODE" == normal || "$MODE" == steered || "$MODE" == no-velocity ]] || fail_setting 'Select a single launcher mode'
for name in STEERNAV_PROCESS_REPEATED_IMAGES STEERNAV_TAPER_PEDESTRIAN_SWEEP STEERNAV_VELOCITY_REFERENCE_LOG; do
  [[ "${!name}" == true || "${!name}" == false ]] || fail_setting "Invalid $name"
done
[[ "$STEERNAV_ODOM_ALIGNMENT" == extrapolate || "$STEERNAV_ODOM_ALIGNMENT" == endpoint ]] || fail_setting 'Invalid alignment'
[[ "$STEERNAV_VELOCITY_WINDOW" =~ ^[0-9]+$ && "$STEERNAV_VELOCITY_WINDOW" -ge 3 && "$STEERNAV_VELOCITY_WINDOW" -le 20 ]] || fail_setting 'Invalid velocity window'
[[ "$STEERNAV_IMAGE_QUEUE_DEPTH" =~ ^[1-9][0-9]*$ ]] || fail_setting 'Invalid queue depth'
for name in STEERNAV_TIME_LOOK_AHEAD STEERNAV_ODOM_POSE_MAX_EXTRAPOLATION STEERNAV_ROBOT_RADIUS STEERNAV_ADDITIONAL_CLEARANCE; do
  [[ "${!name}" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail_setting "Invalid $name"
done
if [[ "$STEERNAV_VELOCITY_REFERENCE_LOG" == true ]]; then
  [[ "$MODE" == steered && "$OBJECT_VELOCITY_SOURCE" == estimated && -n "$STEERNAV_RECEIVER_TIMING_LOG" ]] || fail_setting 'Reference logging requires steered estimated mode and a timing path'
fi
add_setting() {
  local flag="$1" value="${2:-}" quoted
  rg -q -- "$flag" "$STEERNAV_ROOT/steernav/ros_inference.py" || fail_setting "SteerNav lacks $flag"
  steernav_settings_flags+=" $flag"
  if [[ -n "$value" ]]; then printf -v quoted '%q' "$value"; steernav_settings_flags+=" $quoted"; fi
}
if [[ "$MODE" != normal ]]; then
  for pair in 'STEERNAV_TIME_LOOK_AHEAD time-look-ahead' 'STEERNAV_VELOCITY_WINDOW velocity-window' 'STEERNAV_ODOM_ALIGNMENT odom-alignment' 'STEERNAV_ODOM_POSE_MAX_EXTRAPOLATION odom-pose-max-extrapolation' 'STEERNAV_IMAGE_QUEUE_DEPTH image-queue-depth' 'STEERNAV_ROBOT_RADIUS robot-radius' 'STEERNAV_ADDITIONAL_CLEARANCE additional-clearance'; do
    read -r name flag <<< "$pair"; add_setting "--$flag" "${!name}"
  done
  for pair in 'STEERNAV_PROCESS_REPEATED_IMAGES process-repeated-images' 'STEERNAV_TAPER_PEDESTRIAN_SWEEP taper-pedestrian-sweep' 'STEERNAV_VELOCITY_REFERENCE_LOG velocity-reference-log'; do
    read -r name flag <<< "$pair"; if [[ "${!name}" == true ]]; then add_setting "--$flag"; fi
  done
  if [[ -n "$STEERNAV_RECEIVER_TIMING_LOG" ]]; then add_setting --receiver-timing-log "$STEERNAV_RECEIVER_TIMING_LOG"; fi
  add_setting --object-velocity-source "$OBJECT_VELOCITY_SOURCE"
fi
