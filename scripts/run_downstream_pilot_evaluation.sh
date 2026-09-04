#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_REL="outputs/m3_downstream_500step_pilot_seed11"
EUROSAT_ROOT_REL="assets/data/raw/eurosat"
RSICD_ROOT_REL="assets/data/raw/rsicd"
RSICD_ANNOTATION_REL="assets/data/raw/rsicd/dataset_rsicd.json"
EUROSAT_MANIFEST_REL="assets/data/manifests/eurosat_all_zero_shot_v1.jsonl"
EUROSAT_AUDIT_REL="assets/data/manifests/eurosat_all_zero_shot_v1.audit.json"
RSICD_MANIFEST_REL="assets/data/manifests/rsicd_test_retrieval_v1.jsonl"
RSICD_AUDIT_REL="assets/data/manifests/rsicd_test_retrieval_v1.audit.json"
WEB_CONFIG_REL="configs/pilot_web_500step_formal_schedule.toml"
SAT_CONFIG_REL="configs/pilot_sat_500step_formal_schedule_fast_validation.toml"
WEB_OUTPUT_REL="outputs/m3_web_global77_formalschedule_500step_pilot_seed11"
SAT_OUTPUT_REL="outputs/m3_sat_global77_formalschedule_500step_pilot_fastval_seed11"
EXPECTED_RSICD_ANNOTATION_SHA256="5e342037d469d074711676bdb9c02b6942a624530b1959d24d2734e68af9cede"

cd "${PROJECT_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv. Run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi
if [[ -e "${OUTPUT_REL}" ]]; then
  echo "Refusing to overwrite existing downstream evaluation: ${OUTPUT_REL}" >&2
  exit 1
fi
for path in \
  "${WEB_CONFIG_REL}" \
  "${SAT_CONFIG_REL}" \
  "${WEB_OUTPUT_REL}/config.toml" \
  "${WEB_OUTPUT_REL}/provenance.json" \
  "${WEB_OUTPUT_REL}/best.pt" \
  "${SAT_OUTPUT_REL}/config.toml" \
  "${SAT_OUTPUT_REL}/provenance.json" \
  "${SAT_OUTPUT_REL}/best.pt" \
  "${RSICD_ANNOTATION_REL}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required evaluation input is missing: ${path}" >&2
    exit 1
  fi
done
if [[ ! -d "${EUROSAT_ROOT_REL}" || ! -d "${RSICD_ROOT_REL}" ]]; then
  echo "EuroSAT or RSICD raw data directory is missing." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run with uncommitted or untracked project changes." >&2
  echo "Commit and push the exact evaluation code first." >&2
  exit 1
fi

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

observed_annotation_sha256="$(sha256sum "${RSICD_ANNOTATION_REL}" | awk '{print $1}')"
if [[ "${observed_annotation_sha256}" != "${EXPECTED_RSICD_ANNOTATION_SHA256}" ]]; then
  echo "Unexpected RSICD annotation SHA-256: ${observed_annotation_sha256}" >&2
  exit 1
fi

ruff check .
pytest
python -m compileall -q src tools

python tools/prepare_eurosat_manifest.py \
  --images-root "${EUROSAT_ROOT_REL}" \
  --output "${EUROSAT_MANIFEST_REL}" \
  --audit-output "${EUROSAT_AUDIT_REL}"
python tools/prepare_rsicd_retrieval_manifest.py \
  --annotations "${RSICD_ANNOTATION_REL}" \
  --images-root "${RSICD_ROOT_REL}" \
  --split test \
  --output "${RSICD_MANIFEST_REL}" \
  --audit-output "${RSICD_AUDIT_REL}"

mkdir -p "${OUTPUT_REL}"
{
  echo "project_commit=$(git rev-parse HEAD)"
  echo "eurosat_manifest_sha256=$(sha256sum "${EUROSAT_MANIFEST_REL}" | awk '{print $1}')"
  echo "rsicd_manifest_sha256=$(sha256sum "${RSICD_MANIFEST_REL}" | awk '{print $1}')"
  echo "rsicd_annotation_sha256=${observed_annotation_sha256}"
  echo "batch_size=64"
  echo "num_workers=4"
  echo "retrieval_chunk_size=256"
  echo "models=web_official,sat_official,web_500step_best,sat_500step_best"
  echo "test_data=EuroSAT_all,RSICD_test"
} > "${OUTPUT_REL}/preflight.txt"

run_evaluation() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local training_output="$4"
  local args=(
    -m dinotxt_rs.cli.evaluate_downstream
    --config "${config}"
    --eurosat-manifest "${EUROSAT_MANIFEST_REL}"
    --rsicd-manifest "${RSICD_MANIFEST_REL}"
    --eurosat-output "${OUTPUT_REL}/eurosat_${label}.json"
    --rsicd-output "${OUTPUT_REL}/rsicd_${label}.json"
    --batch-size 64
    --num-workers 4
    --retrieval-chunk-size 256
  )
  if [[ -n "${checkpoint}" ]]; then
    args+=(--checkpoint "${checkpoint}" --training-output "${training_output}")
  fi
  python "${args[@]}" 2>&1 | tee "${OUTPUT_REL}/${label}.log"
}

run_evaluation "web_official" "${WEB_CONFIG_REL}" "" ""
run_evaluation "sat_official" "${SAT_CONFIG_REL}" "" ""
run_evaluation "web_500step_best" "${WEB_CONFIG_REL}" "${WEB_OUTPUT_REL}/best.pt" "${WEB_OUTPUT_REL}"
run_evaluation "sat_500step_best" "${SAT_CONFIG_REL}" "${SAT_OUTPUT_REL}/best.pt" "${SAT_OUTPUT_REL}"

python tools/verify_downstream_evaluation.py --output "${OUTPUT_REL}" \
  | tee "${OUTPUT_REL}/verification_report.json"
