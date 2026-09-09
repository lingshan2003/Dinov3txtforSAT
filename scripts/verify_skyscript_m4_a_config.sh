#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run: bash scripts/verify_skyscript_m4_a_config.sh" >&2
  return 2
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv; run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi

.venv/bin/python tools/verify_skyscript_m4_config_pair.py \
  --web-config configs/skyscript_web_adapter_500step_seed11.toml \
  --sat-config configs/skyscript_sat_adapter_500step_seed11.toml

.venv/bin/python -m pytest -q tests/test_skyscript_m4_config_pair.py

echo "M4-A seed11 configuration and focused tests passed. No training was started."
