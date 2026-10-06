#!/usr/bin/env bash
# Native DINOv3.txt baseline and one-epoch remote-sensing adaptation.
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
exec "${PYTHON}" -u tools/run_web_native.py "$@"
