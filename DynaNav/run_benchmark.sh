#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALLER_DIR="${PWD}"
cd "${SCRIPT_DIR}/.."
source .env.testing

CONFIG="${1:-${TICVLA_DYNANAV_ROOT}/configs/benchmark_example.yaml}"
shift || true

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="$(cd "${CALLER_DIR}/$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
fi

cd "${TICVLA_DYNANAV_ROOT}"
exec "${ISAAC_SIM_PYTHON}" benchmark.py -c "${CONFIG}" "$@"
