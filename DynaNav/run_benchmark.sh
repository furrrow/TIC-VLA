#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TICVLA_DYNANAV_ROOT="${TICVLA_DYNANAV_ROOT:-${SCRIPT_DIR}}"
export ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${HOME}/isaacsim}"
export ISAAC_SIM_PYTHON="${ISAAC_SIM_PYTHON:-${ISAAC_SIM_ROOT}/python.sh}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

CONFIG="${1:-${TICVLA_DYNANAV_ROOT}/configs/benchmark_example.yaml}"
shift || true

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
fi

cd "${TICVLA_DYNANAV_ROOT}"
echo "Launching Isaac Sim with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
exec "${ISAAC_SIM_PYTHON}" benchmark.py -c "${CONFIG}" --navigation_method ticvla "$@"
