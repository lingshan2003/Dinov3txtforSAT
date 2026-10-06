#!/usr/bin/env bash
# Separate tests of batch size, image epoch budget and stronger alignment-module updates.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run with bash; do not source this script." >&2
  return 2
fi
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing .venv/bin/python" >&2; exit 1; }
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "${PYTHON}" -u tools/run_sat_maskpos_next.py "$@"
