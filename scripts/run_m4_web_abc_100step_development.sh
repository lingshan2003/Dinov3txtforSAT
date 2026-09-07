#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_A_REL="outputs/gate_a_step0_parity/verification_report.json"
TRAIN_MANIFEST_REL="assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77.jsonl"
VAL_MANIFEST_REL="assets/data/manifests/chatearthnet_35_val_no_nodata_global77.jsonl"
MONITOR_MANIFEST_REL="assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77_fixed16.jsonl"
RSICD_ROOT_REL="assets/data/raw/rsicd"
RSICD_ANNOTATION_REL="assets/data/raw/rsicd/dataset_rsicd.json"
RSICD_VAL_MANIFEST_REL="assets/data/manifests/rsicd_val_retrieval_v1.jsonl"
RSICD_VAL_AUDIT_REL="assets/data/manifests/rsicd_val_retrieval_v1.audit.json"
DEVELOPMENT_OUTPUT_REL="outputs/m4_web_abc_100step_development_seed11"
CONFIG_A_REL="configs/m4_web_a_fullscope_lr5e6_100step.toml"
CONFIG_B_REL="configs/m4_web_b_visionhead_lr5e5_100step.toml"
CONFIG_C_REL="configs/m4_web_c_visionhead_lr5e6_100step.toml"
OUTPUT_A_REL="outputs/m4_web_a_fullscope_lr5e6_100step_seed11"
OUTPUT_B_REL="outputs/m4_web_b_visionhead_lr5e5_100step_seed11"
OUTPUT_C_REL="outputs/m4_web_c_visionhead_lr5e6_100step_seed11"
EXPECTED_TRAIN_MANIFEST_SHA256="78abc613fbc8d98ea4617770473b30662d9eda31c0deb0dd06b51b1965d9fc0b"
EXPECTED_VAL_MANIFEST_SHA256="1040ccf2ec07100ceb81ad665e28527d38b948cc9e1e547eb76e23a265c25f88"
EXPECTED_MONITOR_MANIFEST_SHA256="983964ad9622daa05b67b71bd2b5a9fafa06866eff7c815f2abdd3adbc3237a8"
EXPECTED_RSICD_ANNOTATION_SHA256="5e342037d469d074711676bdb9c02b6942a624530b1959d24d2734e68af9cede"
EXPECTED_DINOV3_COMMIT="6876159a11b4df116f30f667f8c9888617df0751"
RETENTION_MARGIN="0.01"

cd "${PROJECT_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv. Run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi
for path in \
  "${GATE_A_REL}" \
  "${TRAIN_MANIFEST_REL}" \
  "${VAL_MANIFEST_REL}" \
  "${MONITOR_MANIFEST_REL}" \
  "${RSICD_ANNOTATION_REL}" \
  "${CONFIG_A_REL}" \
  "${CONFIG_B_REL}" \
  "${CONFIG_C_REL}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required M4 input is missing: ${path}" >&2
    exit 1
  fi
done
if [[ ! -d "${RSICD_ROOT_REL}" ]]; then
  echo "Missing RSICD image directory: ${RSICD_ROOT_REL}" >&2
  exit 1
fi
for output in \
  "${DEVELOPMENT_OUTPUT_REL}" \
  "${OUTPUT_A_REL}" \
  "${OUTPUT_B_REL}" \
  "${OUTPUT_C_REL}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing M4 output: ${output}" >&2
    exit 1
  fi
done
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run with uncommitted or untracked project changes." >&2
  echo "Commit and push the exact M4 code/config first." >&2
  exit 1
fi

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

current_project_commit="$(git rev-parse HEAD)"
gate_status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${GATE_A_REL}")"
gate_web_commit="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"]["web"]["project_commit"])' "${GATE_A_REL}")"
gate_sat_commit="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"]["sat"]["project_commit"])' "${GATE_A_REL}")"
if [[ "${gate_status}" != "complete" ]]; then
  echo "Gate A is not complete: ${gate_status}" >&2
  exit 1
fi
if [[ "${gate_web_commit}" != "${current_project_commit}" || "${gate_sat_commit}" != "${current_project_commit}" ]]; then
  echo "Gate A was not run at current commit ${current_project_commit}." >&2
  echo "Preserve the old Gate A output, then rerun scripts/run_step0_parity_checks.sh." >&2
  exit 1
fi

observed_train_sha256="$(sha256sum "${TRAIN_MANIFEST_REL}" | awk '{print $1}')"
observed_val_sha256="$(sha256sum "${VAL_MANIFEST_REL}" | awk '{print $1}')"
observed_monitor_sha256="$(sha256sum "${MONITOR_MANIFEST_REL}" | awk '{print $1}')"
observed_annotation_sha256="$(sha256sum "${RSICD_ANNOTATION_REL}" | awk '{print $1}')"
if [[ "${observed_train_sha256}" != "${EXPECTED_TRAIN_MANIFEST_SHA256}" ]]; then
  echo "Unexpected training manifest SHA-256: ${observed_train_sha256}" >&2
  exit 1
fi
if [[ "${observed_val_sha256}" != "${EXPECTED_VAL_MANIFEST_SHA256}" ]]; then
  echo "Unexpected validation manifest SHA-256: ${observed_val_sha256}" >&2
  exit 1
fi
if [[ "${observed_monitor_sha256}" != "${EXPECTED_MONITOR_MANIFEST_SHA256}" ]]; then
  echo "Unexpected fixed monitor manifest SHA-256: ${observed_monitor_sha256}" >&2
  exit 1
fi
if [[ "${observed_annotation_sha256}" != "${EXPECTED_RSICD_ANNOTATION_SHA256}" ]]; then
  echo "Unexpected RSICD annotation SHA-256: ${observed_annotation_sha256}" >&2
  exit 1
fi

ruff check .
pytest
python -m compileall -q src tools

python tools/prepare_rsicd_retrieval_manifest.py \
  --annotations "${RSICD_ANNOTATION_REL}" \
  --images-root "${RSICD_ROOT_REL}" \
  --split val \
  --output "${RSICD_VAL_MANIFEST_REL}" \
  --audit-output "${RSICD_VAL_AUDIT_REL}"
rsicd_val_sha256="$(sha256sum "${RSICD_VAL_MANIFEST_REL}" | awk '{print $1}')"

mkdir -p "${DEVELOPMENT_OUTPUT_REL}"
python tools/audit_manifest_image_overlap.py \
  --left-manifest "${TRAIN_MANIFEST_REL}" \
  --right-manifest "${RSICD_VAL_MANIFEST_REL}" \
  --output "${DEVELOPMENT_OUTPUT_REL}/train_vs_rsicd_val_overlap.json" \
  --require-zero-overlap \
  | tee "${DEVELOPMENT_OUTPUT_REL}/overlap_audit.log"

{
  echo "project_commit=${current_project_commit}"
  echo "gate_a=${GATE_A_REL}"
  echo "train_manifest_sha256=${observed_train_sha256}"
  echo "val_manifest_sha256=${observed_val_sha256}"
  echo "fixed_monitor_manifest_sha256=${observed_monitor_sha256}"
  echo "rsicd_annotation_sha256=${observed_annotation_sha256}"
  echo "rsicd_val_manifest_sha256=${rsicd_val_sha256}"
  echo "retention_margin_absolute_mean_recall=${RETENTION_MARGIN}"
  echo "configured_target_steps=5000"
  echo "execution_cap_steps=100"
  echo "resume_step=50"
} > "${DEVELOPMENT_OUTPUT_REL}/preflight.txt"

run_training() {
  local label="$1"
  local config="$2"
  local output="$3"

  mkdir -p "${output}"
  dinotxt-rs-smoke --config "${config}" --batch-size 64 \
    2>&1 | tee "${DEVELOPMENT_OUTPUT_REL}/${label}_smoke.log"
  dinotxt-rs-train --config "${config}" --stop-after-step 50 \
    2>&1 | tee "${output}/train_phase_1.log"
  dinotxt-rs-train --config "${config}" --resume "${output}/step_0000050.pt" \
    --stop-after-step 100 2>&1 | tee "${output}/train_phase_2.log"
  python tools/verify_training_run.py \
    --output "${output}" \
    --expected-steps 100 \
    --expected-target-steps 5000 \
    --require-incomplete \
    --expected-train-manifest-sha256 "${EXPECTED_TRAIN_MANIFEST_SHA256}" \
    --expected-dinov3-commit "${EXPECTED_DINOV3_COMMIT}" \
    --expected-final-queue-size 4096 \
    --required-checkpoint-step 0 \
    --required-checkpoint-step 50 \
    --required-checkpoint-step 100 \
    --require-in-batch-loss \
    --require-fixed-monitor \
    --expected-fixed-monitor-manifest-sha256 "${EXPECTED_MONITOR_MANIFEST_SHA256}" \
    --fixed-monitor-every 10 \
    --require-validation \
    --expected-val-manifest-sha256 "${EXPECTED_VAL_MANIFEST_SHA256}" \
    --validation-every 50 \
    --expected-validation-loss-batch-size 16 \
    --expected-validation-forward-batch-size 64 \
    --require-best-checkpoint \
    --required-resume-step 50 \
    | tee "${output}/verification_report.json"
}

run_training "a" "${CONFIG_A_REL}" "${OUTPUT_A_REL}"
run_training "b" "${CONFIG_B_REL}" "${OUTPUT_B_REL}"
run_training "c" "${CONFIG_C_REL}" "${OUTPUT_C_REL}"

evaluate_val() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local training_output="$4"
  local args=(
    -m dinotxt_rs.cli.evaluate_rsicd
    --config "${config}"
    --manifest "${RSICD_VAL_MANIFEST_REL}"
    --split val
    --output "${DEVELOPMENT_OUTPUT_REL}/rsicd_val_${label}.json"
    --batch-size 64
    --num-workers 4
    --retrieval-chunk-size 256
  )
  if [[ -n "${checkpoint}" ]]; then
    args+=(--checkpoint "${checkpoint}" --training-output "${training_output}")
  fi
  python "${args[@]}" 2>&1 | tee "${DEVELOPMENT_OUTPUT_REL}/rsicd_val_${label}.log"
}

evaluate_val "official" "${CONFIG_A_REL}" "" ""
for step in 50 100; do
  checkpoint_name="step_$(printf '%07d' "${step}").pt"
  evaluate_val "a_step${step}" "${CONFIG_A_REL}" "${OUTPUT_A_REL}/${checkpoint_name}" "${OUTPUT_A_REL}"
  evaluate_val "b_step${step}" "${CONFIG_B_REL}" "${OUTPUT_B_REL}/${checkpoint_name}" "${OUTPUT_B_REL}"
  evaluate_val "c_step${step}" "${CONFIG_C_REL}" "${OUTPUT_C_REL}/${checkpoint_name}" "${OUTPUT_C_REL}"
done

python tools/verify_m4_abc_development.py \
  --development-output "${DEVELOPMENT_OUTPUT_REL}" \
  --run-a "${OUTPUT_A_REL}" \
  --run-b "${OUTPUT_B_REL}" \
  --run-c "${OUTPUT_C_REL}" \
  --retention-margin "${RETENTION_MARGIN}" \
  | tee "${DEVELOPMENT_OUTPUT_REL}/verification_report.json"
