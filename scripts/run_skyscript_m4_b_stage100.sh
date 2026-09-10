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
echo "Running code checks once before the four GPU jobs..."
ruff check . || exit $?
pytest || exit $?
python -m compileall -q src tools || exit $?

failures=()
for domain_seed in web:23 sat:23 web:47 sat:47; do
  domain="${domain_seed%%:*}"
  seed="${domain_seed##*:}"
  echo "Starting M4-B ${domain} seed${seed} through stage100..."
  if ! bash scripts/run_skyscript_m4_b.sh \
      "$@" \
      --domain "${domain}" \
      --seed "${seed}" \
      --stop-after-stage 100 \
      --skip-code-checks; then
    failures+=("${domain}:seed${seed}")
    echo "M4-B ${domain} seed${seed} did not pass; preserving evidence and continuing." >&2
  fi
done

SUMMARY="outputs/skyscript_m4_rq1_m4_b_stage100/summary.json"
echo "Aggregating seed11/23/47 stage100 evidence..."
python tools/summarize_skyscript_m4_b.py \
  --stage 100 \
  --output "${SUMMARY}" || failures+=("cross-seed-summary")

if (( ${#failures[@]} > 0 )); then
  echo "M4-B stage100 completed with failures: ${failures[*]}" >&2
  echo "Cross-seed report (when all six reports exist): ${SUMMARY}" >&2
  exit 2
fi

echo "M4-B seed23/47 Web and SAT runs all passed stage100. Summary: ${SUMMARY}"
