#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_REL="outputs/gate_a_step0_parity"
INPUT_MANIFEST_REL="assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77_fixed16.jsonl"
WEB_CONFIG_REL="configs/pilot_web_500step_formal_schedule.toml"
SAT_CONFIG_REL="configs/pilot_sat_500step_formal_schedule_fast_validation.toml"
WEB_TRAINING_OUTPUT_REL="outputs/m3_web_global77_formalschedule_500step_pilot_seed11"
SAT_TRAINING_OUTPUT_REL="outputs/m3_sat_global77_formalschedule_500step_pilot_fastval_seed11"

cd "${PROJECT_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv. Run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi
if [[ -e "${OUTPUT_REL}" ]]; then
  echo "Refusing to overwrite existing parity output: ${OUTPUT_REL}" >&2
  exit 1
fi
for path in \
  "${INPUT_MANIFEST_REL}" \
  "${WEB_CONFIG_REL}" \
  "${SAT_CONFIG_REL}" \
  "${WEB_TRAINING_OUTPUT_REL}/config.toml" \
  "${WEB_TRAINING_OUTPUT_REL}/provenance.json" \
  "${WEB_TRAINING_OUTPUT_REL}/step_0000000.pt" \
  "${SAT_TRAINING_OUTPUT_REL}/config.toml" \
  "${SAT_TRAINING_OUTPUT_REL}/provenance.json" \
  "${SAT_TRAINING_OUTPUT_REL}/step_0000000.pt"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required parity input is missing: ${path}" >&2
    exit 1
  fi
done
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run parity with uncommitted or untracked project changes." >&2
  echo "Commit and push the exact parity code first." >&2
  exit 1
fi

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ruff check .
pytest
python -m compileall -q src tools

mkdir -p "${OUTPUT_REL}"
{
  echo "project_commit=$(git rev-parse HEAD)"
  echo "input_manifest_sha256=$(sha256sum "${INPUT_MANIFEST_REL}" | awk '{print $1}')"
  echo "batch_size=16"
  echo "precision=from_training_config"
  echo "models=web_official_vs_step0,sat_official_vs_step0"
} > "${OUTPUT_REL}/preflight.txt"

python -m dinotxt_rs.cli.check_step0_parity \
  --config "${WEB_CONFIG_REL}" \
  --checkpoint "${WEB_TRAINING_OUTPUT_REL}/step_0000000.pt" \
  --training-output "${WEB_TRAINING_OUTPUT_REL}" \
  --input-manifest "${INPUT_MANIFEST_REL}" \
  --batch-size 16 \
  --output "${OUTPUT_REL}/web.json" \
  2>&1 | tee "${OUTPUT_REL}/web.log"

python -m dinotxt_rs.cli.check_step0_parity \
  --config "${SAT_CONFIG_REL}" \
  --checkpoint "${SAT_TRAINING_OUTPUT_REL}/step_0000000.pt" \
  --training-output "${SAT_TRAINING_OUTPUT_REL}" \
  --input-manifest "${INPUT_MANIFEST_REL}" \
  --batch-size 16 \
  --output "${OUTPUT_REL}/sat.json" \
  2>&1 | tee "${OUTPUT_REL}/sat.log"

python tools/verify_step0_parity.py \
  --output-dir "${OUTPUT_REL}" \
  --report "${OUTPUT_REL}/verification_report.json" \
  | tee "${OUTPUT_REL}/verification.log"
