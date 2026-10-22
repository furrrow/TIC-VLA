#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SCENARIO="DynaNav/configs/steernav_scenarios/scenario3_easy_20trials.yaml"
SCENARIO="${DEFAULT_SCENARIO}"
CHECK_ONLY=false
JOB_ID=""
STOP_JOB_ID=""
INFERENCE_METHOD=""
STATE_DIR="${TMPDIR:-/tmp}/ticvla-run-sim"

usage() {
  cat <<EOF
Usage: $(basename "$0") [-s FILE | --scenario FILE] [--inference-method NAME] [--job-id ID] [-- BENCHMARK_ARGS...]
       $(basename "$0") --check [-s FILE | --scenario FILE]
       $(basename "$0") --stop JOB_ID

Launch the DynaNav benchmark in Isaac Sim.

Options:
  -s, --scenario FILE  Scenario file (default: ${DEFAULT_SCENARIO})
      --inference-method NAME
                         Model name recorded in benchmark results
      --job-id ID       Track the benchmark as a stoppable remote job
      --stop JOB_ID     Stop a tracked benchmark and its Isaac Sim child
      --check           Validate the scenario and simulator environment
  -h, --help           Show this help message and exit

FILE may be absolute or relative to the repository root. Arguments after -- are
passed to DynaNav/run_benchmark.sh.
EOF
}

benchmark_args=()
while (( $# > 0 )); do
  case "$1" in
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
    --job-id)
      if (( $# < 2 )); then
        echo "--job-id requires an ID." >&2
        usage >&2
        exit 2
      fi
      JOB_ID="$2"
      shift 2
      ;;
    --job-id=*)
      JOB_ID="${1#*=}"
      shift
      ;;
    --inference-method|--inference_method)
      if (( $# < 2 )); then
        echo "$1 requires a model name." >&2
        usage >&2
        exit 2
      fi
      INFERENCE_METHOD="$2"
      shift 2
      ;;
    --inference-method=*|--inference_method=*)
      INFERENCE_METHOD="${1#*=}"
      shift
      ;;
    --stop)
      if (( $# < 2 )); then
        echo "--stop requires a job ID." >&2
        usage >&2
        exit 2
      fi
      STOP_JOB_ID="$2"
      shift 2
      ;;
    --check)
      CHECK_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      benchmark_args=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

validate_job_id() {
  [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]]
}

stop_process_group() {
  local pid_file="$1"
  local child_pgid=""
  local group_commands=""
  local deadline=0

  if [[ ! -f "${pid_file}" ]]; then
    echo "No active simulator job found for $(basename "${pid_file}" .pid)."
    return 0
  fi
  child_pgid="$(<"${pid_file}")"
  if [[ ! "${child_pgid}" =~ ^[0-9]+$ ]]; then
    echo "Invalid simulator job state: ${pid_file}" >&2
    return 1
  fi
  group_commands="$(ps -eo pgid=,args= | awk -v pgid="${child_pgid}" '$1 == pgid { $1=""; print }')"
  if [[ "${group_commands}" != *"benchmark.py"* && "${group_commands}" != *"run_benchmark.sh"* ]]; then
    echo "Tracked simulator process ${child_pgid} is no longer running."
    rm -f "${pid_file}"
    return 0
  fi

  echo "Stopping simulator process group ${child_pgid}..."
  kill -INT -- "-${child_pgid}" 2>/dev/null || true
  deadline=$((SECONDS + 20))
  while kill -0 -- "-${child_pgid}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 1
  done
  if kill -0 -- "-${child_pgid}" 2>/dev/null; then
    echo "Simulator did not exit after SIGINT; sending SIGTERM." >&2
    kill -TERM -- "-${child_pgid}" 2>/dev/null || true
    deadline=$((SECONDS + 5))
    while kill -0 -- "-${child_pgid}" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 1
    done
  fi
  if kill -0 -- "-${child_pgid}" 2>/dev/null; then
    echo "Simulator did not exit after SIGTERM; sending SIGKILL." >&2
    kill -KILL -- "-${child_pgid}" 2>/dev/null || true
  fi
  rm -f "${pid_file}"
}

if [[ -n "${STOP_JOB_ID}" ]]; then
  if ! validate_job_id "${STOP_JOB_ID}"; then
    echo "Job ID may contain only letters, numbers, dots, underscores, and hyphens." >&2
    exit 2
  fi
  stop_process_group "${STATE_DIR}/${STOP_JOB_ID}.pid"
  exit 0
fi

if [[ -n "${JOB_ID}" ]] && ! validate_job_id "${JOB_ID}"; then
  echo "Job ID may contain only letters, numbers, dots, underscores, and hyphens." >&2
  exit 2
fi
if [[ -n "${INFERENCE_METHOD}" ]]; then
  if [[ ! "${INFERENCE_METHOD}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Inference method may contain only letters, numbers, dots, underscores, and hyphens." >&2
    exit 2
  fi
  benchmark_args+=("--inference_method" "${INFERENCE_METHOD}")
fi

if [[ "${SCENARIO}" != /* ]]; then
  SCENARIO="${REPO_ROOT}/${SCENARIO}"
fi
if [[ ! -f "${SCENARIO}" ]]; then
  echo "Scenario file not found: ${SCENARIO}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source .env.testing

# SSH sessions do not inherit the simulator desktop's display. Default to the
# active local X session so Isaac Sim opens visibly on the simulator machine.
# Set DISPLAY before invoking this script if the desktop uses another display.
export DISPLAY="${DISPLAY:-:1}"
if [[ ! "${DISPLAY}" =~ ^:([0-9]+)(\.[0-9]+)?$ ]]; then
  echo "DISPLAY must be a local X display such as :0 or :1." >&2
  exit 2
fi
display_number="${BASH_REMATCH[1]}"
if [[ ! -S "/tmp/.X11-unix/X${display_number}" ]]; then
  echo "X display ${DISPLAY} is unavailable on this machine." >&2
  exit 1
fi
if [[ ! -x "${REPO_ROOT}/DynaNav/run_benchmark.sh" ]]; then
  echo "Benchmark launcher is missing or not executable." >&2
  exit 1
fi
if [[ ! -x "${ISAAC_SIM_PYTHON}" ]]; then
  echo "Isaac Sim Python is missing or not executable: ${ISAAC_SIM_PYTHON}" >&2
  exit 1
fi

if [[ "${CHECK_ONLY}" == true ]]; then
  echo "Simulator environment is ready on DISPLAY=${DISPLAY}."
  echo "Scenario file: ${SCENARIO}"
  exit 0
fi

echo "Launching Isaac Sim on DISPLAY=${DISPLAY}"
echo "Scenario file: ${SCENARIO}"
if [[ -n "${INFERENCE_METHOD}" ]]; then
  echo "Inference method: ${INFERENCE_METHOD}"
fi

if [[ -z "${JOB_ID}" ]]; then
  exec DynaNav/run_benchmark.sh "${SCENARIO}" "${benchmark_args[@]}"
fi

mkdir -p "${STATE_DIR}"
pid_file="${STATE_DIR}/${JOB_ID}.pid"
if [[ -f "${pid_file}" ]]; then
  echo "Simulator job already exists: ${JOB_ID}" >&2
  echo "Stop it first with: $(basename "$0") --stop ${JOB_ID}" >&2
  exit 1
fi

child_pgid=""
cleanup_child() {
  if [[ -n "${child_pgid}" ]]; then
    stop_process_group "${pid_file}"
    child_pgid=""
  fi
}
handle_signal() {
  local status="$1"
  trap - HUP INT TERM EXIT
  cleanup_child
  exit "${status}"
}
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
trap cleanup_child EXIT

setsid DynaNav/run_benchmark.sh "${SCENARIO}" "${benchmark_args[@]}" &
child_pgid=$!
printf '%s\n' "${child_pgid}" > "${pid_file}"
set +e
wait "${child_pgid}"
status=$?
set -e
child_pgid=""
rm -f "${pid_file}"
exit "${status}"
