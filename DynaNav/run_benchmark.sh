#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALLER_DIR="${PWD}"
cd "${SCRIPT_DIR}/.."
# .env.testing appends Isaac's ROS libraries to this variable. Non-interactive
# SSH shells may not define it, while this script deliberately enables set -u.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source .env.testing

CONFIG="${1:-${TICVLA_DYNANAV_ROOT}/configs/benchmark_example.yaml}"
shift || true

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="$(cd "${CALLER_DIR}/$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
fi

cd "${TICVLA_DYNANAV_ROOT}"
exec "${ISAAC_SIM_PYTHON}" benchmark.py -c "${CONFIG}" \
  --terminate-on-human-collision "$@"
