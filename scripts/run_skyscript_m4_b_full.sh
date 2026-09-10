#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it with bash." >&2
  return 2
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv; run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Verifying the complete M4-B configuration matrix..."
python tools/verify_skyscript_m4_b_configs.py
echo "Running code checks once before GPU work..."
ruff check .
pytest
python -m compileall -q src tools

for stage in 100 250 500; do
  failures=()
  echo "Starting all four registered M4-B runs through stage${stage}..."
  for domain_seed in web:23 sat:23 web:47 sat:47; do
    domain="${domain_seed%%:*}"
    seed="${domain_seed##*:}"
    echo "Running M4-B ${domain} seed${seed} through stage${stage}..."
    if ! bash scripts/run_skyscript_m4_b.sh \
        "$@" \
        --domain "${domain}" \
        --seed "${seed}" \
        --stop-after-stage "${stage}" \
        --skip-code-checks; then
      failures+=("${domain}:seed${seed}")
      echo "M4-B ${domain} seed${seed} failed at/before stage${stage}; continuing this stage." >&2
    fi
  done

  summary="outputs/skyscript_m4_rq1_m4_b_stage${stage}/summary.json"
  echo "Aggregating seed11/23/47 stage${stage} evidence..."
  if ! python tools/summarize_skyscript_m4_b.py \
      --stage "${stage}" \
      --output "${summary}"; then
    failures+=("cross-seed-summary")
  fi

  if (( ${#failures[@]} > 0 )); then
    echo "M4-B stopped after stage${stage}: ${failures[*]}" >&2
    echo "No run will advance to the next stage. Preserved report (if complete): ${summary}" >&2
    exit 2
  fi
  echo "All six domain×seed reports passed stage${stage}: ${summary}"
done

echo "M4-B completed through step500."
echo "Final cross-seed report: outputs/skyscript_m4_rq1_m4_b_stage500/summary.json"
