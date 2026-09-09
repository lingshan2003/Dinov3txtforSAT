#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run: bash scripts/summarize_skyscript_m4_a_seed11.sh" >&2
  return 2
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv; run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi

WEB_SUMMARY="outputs/skyscript_gate_s2_seed11/summary.json"
SAT_SUMMARY="outputs/skyscript_gate_m4_sat_seed11/summary.json"
OUTPUT="outputs/skyscript_m4_rq1_seed11/summary.json"

for path in \
  "${WEB_SUMMARY}" \
  "${SAT_SUMMARY}" \
  configs/skyscript_web_adapter_500step_seed11.toml \
  configs/skyscript_sat_adapter_500step_seed11.toml; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required M4-A input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${OUTPUT}")"
.venv/bin/python tools/summarize_skyscript_m4_rq1_seed11.py \
  --web-summary "${WEB_SUMMARY}" \
  --sat-summary "${SAT_SUMMARY}" \
  --web-config configs/skyscript_web_adapter_500step_seed11.toml \
  --sat-config configs/skyscript_sat_adapter_500step_seed11.toml \
  --output "${OUTPUT}"

echo "M4-A matched seed11 comparison: ${OUTPUT}"
