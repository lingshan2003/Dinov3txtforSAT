#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_REL="configs/verify_sat_100step_validation_resume.toml"
OUTPUT_REL="outputs/m3_sat_global77_100step_validation_resume_seed11"
WEB_OUTPUT_REL="outputs/m3_web_global77_100step_validation_resume_seed11"
TRAIN_MANIFEST_REL="assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77.jsonl"
VAL_MANIFEST_REL="assets/data/manifests/chatearthnet_35_val_no_nodata_global77.jsonl"
MONITOR_MANIFEST_REL="assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77_fixed16.jsonl"
MONITOR_AUDIT_REL="assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77_fixed16.audit.json"
EXPECTED_TRAIN_MANIFEST_SHA256="78abc613fbc8d98ea4617770473b30662d9eda31c0deb0dd06b51b1965d9fc0b"
EXPECTED_VAL_MANIFEST_SHA256="1040ccf2ec07100ceb81ad665e28527d38b948cc9e1e547eb76e23a265c25f88"
EXPECTED_MONITOR_MANIFEST_SHA256="983964ad9622daa05b67b71bd2b5a9fafa06866eff7c815f2abdd3adbc3237a8"
EXPECTED_DINOV3_COMMIT="6876159a11b4df116f30f667f8c9888617df0751"

cd "${PROJECT_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv. Run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi
if [[ ! -f "${CONFIG_REL}" || ! -f "${TRAIN_MANIFEST_REL}" || ! -f "${VAL_MANIFEST_REL}" ]]; then
  echo "Missing the SAT validation/resume config or canonical manifests." >&2
  exit 1
fi
if [[ ! -f "${WEB_OUTPUT_REL}/verification_report.json" ]]; then
  echo "Run and retain the passing Web validation/resume verification first." >&2
  exit 1
fi
if [[ -e "${OUTPUT_REL}" ]]; then
  echo "Refusing to overwrite existing experiment output: ${OUTPUT_REL}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run with uncommitted or untracked project changes." >&2
  echo "Commit and push the exact code/config first so provenance records the correct commit." >&2
  exit 1
fi

source .venv/bin/activate
web_project_commit="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["project_commit"])' "${WEB_OUTPUT_REL}/provenance.json")"
current_project_commit="$(git rev-parse HEAD)"
if [[ "${web_project_commit}" != "${current_project_commit}" ]]; then
  echo "Web gate used ${web_project_commit}; current project is ${current_project_commit}." >&2
  echo "Re-run the Web validation/resume gate with the current committed code first." >&2
  exit 1
fi
observed_train_manifest_sha256="$(sha256sum "${TRAIN_MANIFEST_REL}" | awk '{print $1}')"
observed_val_manifest_sha256="$(sha256sum "${VAL_MANIFEST_REL}" | awk '{print $1}')"
if [[ "${observed_train_manifest_sha256}" != "${EXPECTED_TRAIN_MANIFEST_SHA256}" ]]; then
  echo "Unexpected training manifest SHA-256: ${observed_train_manifest_sha256}" >&2
  exit 1
fi
if [[ "${observed_val_manifest_sha256}" != "${EXPECTED_VAL_MANIFEST_SHA256}" ]]; then
  echo "Unexpected validation manifest SHA-256: ${observed_val_manifest_sha256}" >&2
  exit 1
fi

python tools/prepare_fixed_manifest.py \
  --input "${TRAIN_MANIFEST_REL}" \
  --output "${MONITOR_MANIFEST_REL}" \
  --limit 16 \
  --audit-output "${MONITOR_AUDIT_REL}"
observed_monitor_manifest_sha256="$(sha256sum "${MONITOR_MANIFEST_REL}" | awk '{print $1}')"
if [[ "${observed_monitor_manifest_sha256}" != "${EXPECTED_MONITOR_MANIFEST_SHA256}" ]]; then
  echo "Unexpected fixed monitor manifest SHA-256: ${observed_monitor_manifest_sha256}" >&2
  exit 1
fi

ruff check .
pytest
python -m compileall -q src tools

mkdir -p "${OUTPUT_REL}"
{
  echo "project_commit=$(git rev-parse HEAD)"
  echo "train_manifest_sha256=${observed_train_manifest_sha256}"
  echo "val_manifest_sha256=${observed_val_manifest_sha256}"
  echo "fixed_monitor_manifest_sha256=${observed_monitor_manifest_sha256}"
  echo "config=${CONFIG_REL}"
  echo "web_gate=${WEB_OUTPUT_REL}/verification_report.json"
  echo "phase_1=stop_after_step_50"
  echo "phase_2=resume_step_50_to_100"
} > "${OUTPUT_REL}/preflight.txt"

python tools/verify_training_run.py \
  --output "${WEB_OUTPUT_REL}" \
  --expected-steps 100 \
  --expected-train-manifest-sha256 "${EXPECTED_TRAIN_MANIFEST_SHA256}" \
  --expected-dinov3-commit "${EXPECTED_DINOV3_COMMIT}" \
  --expected-final-queue-size 4096 \
  --require-completed \
  --required-checkpoint-step 50 \
  --required-checkpoint-step 100 \
  --require-in-batch-loss \
  --require-fixed-monitor \
  --expected-fixed-monitor-manifest-sha256 "${EXPECTED_MONITOR_MANIFEST_SHA256}" \
  --fixed-monitor-every 10 \
  --require-validation \
  --expected-val-manifest-sha256 "${EXPECTED_VAL_MANIFEST_SHA256}" \
  --validation-every 50 \
  --require-best-checkpoint \
  --required-resume-step 50 | tee "${OUTPUT_REL}/web_gate_report.json"

dinotxt-rs-smoke --config "${CONFIG_REL}" 2>&1 | tee "${OUTPUT_REL}/smoke.log"
dinotxt-rs-train --config "${CONFIG_REL}" --stop-after-step 50 \
  2>&1 | tee "${OUTPUT_REL}/train_phase_1.log"
dinotxt-rs-train --config "${CONFIG_REL}" --resume "${OUTPUT_REL}/step_0000050.pt" \
  2>&1 | tee "${OUTPUT_REL}/train_phase_2.log"

python tools/verify_training_run.py \
  --output "${OUTPUT_REL}" \
  --expected-steps 100 \
  --expected-train-manifest-sha256 "${EXPECTED_TRAIN_MANIFEST_SHA256}" \
  --expected-dinov3-commit "${EXPECTED_DINOV3_COMMIT}" \
  --expected-final-queue-size 4096 \
  --require-completed \
  --required-checkpoint-step 50 \
  --required-checkpoint-step 100 \
  --require-in-batch-loss \
  --require-fixed-monitor \
  --expected-fixed-monitor-manifest-sha256 "${EXPECTED_MONITOR_MANIFEST_SHA256}" \
  --fixed-monitor-every 10 \
  --require-validation \
  --expected-val-manifest-sha256 "${EXPECTED_VAL_MANIFEST_SHA256}" \
  --validation-every 50 \
  --require-best-checkpoint \
  --required-resume-step 50 | tee "${OUTPUT_REL}/verification_report.json"
