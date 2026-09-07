#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M4_OUTPUT_REL="outputs/m4_web_abc_100step_development_seed11"
M5_OUTPUT_REL="outputs/m5_web_ac_250step_development_seed11"
CONFIG_A_REL="configs/m4_web_a_fullscope_lr5e6_100step.toml"
CONFIG_C_REL="configs/m4_web_c_visionhead_lr5e6_100step.toml"
OUTPUT_A_REL="outputs/m4_web_a_fullscope_lr5e6_100step_seed11"
OUTPUT_C_REL="outputs/m4_web_c_visionhead_lr5e6_100step_seed11"
EXPECTED_TRAIN_MANIFEST_SHA256="78abc613fbc8d98ea4617770473b30662d9eda31c0deb0dd06b51b1965d9fc0b"
EXPECTED_VAL_MANIFEST_SHA256="1040ccf2ec07100ceb81ad665e28527d38b948cc9e1e547eb76e23a265c25f88"
EXPECTED_MONITOR_MANIFEST_SHA256="983964ad9622daa05b67b71bd2b5a9fafa06866eff7c815f2abdd3adbc3237a8"
EXPECTED_DINOV3_COMMIT="6876159a11b4df116f30f667f8c9888617df0751"
RETENTION_MARGIN="0.01"

cd "${PROJECT_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv. Run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi
for path in \
  "${M4_OUTPUT_REL}/verification_report.json" \
  "${M4_OUTPUT_REL}/train_vs_rsicd_val_overlap.json" \
  "${M4_OUTPUT_REL}/rsicd_val_official.json" \
  "${M4_OUTPUT_REL}/rsicd_val_a_step100.json" \
  "${M4_OUTPUT_REL}/rsicd_val_c_step100.json" \
  "${OUTPUT_A_REL}/step_0000100.pt" \
  "${OUTPUT_A_REL}/provenance.json" \
  "${OUTPUT_C_REL}/step_0000100.pt" \
  "${OUTPUT_C_REL}/provenance.json" \
  "${CONFIG_A_REL}" \
  "${CONFIG_C_REL}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required M5 input is missing: ${path}" >&2
    exit 1
  fi
done
if [[ -e "${M5_OUTPUT_REL}" ]]; then
  echo "Refusing to overwrite existing M5 output: ${M5_OUTPUT_REL}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run with uncommitted or untracked project changes." >&2
  echo "Commit and push the exact M5 verifier and protocol first." >&2
  exit 1
fi

m4_status="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${M4_OUTPUT_REL}/verification_report.json")"
m4_recommendation="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended_for_250step"])' "${M4_OUTPUT_REL}/verification_report.json")"
if [[ "${m4_status}" != "complete" || "${m4_recommendation}" != "a" ]]; then
  echo "M4 must be complete and recommend A before this registered A/C continuation." >&2
  exit 1
fi

training_commit_a="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_commit"])' "${OUTPUT_A_REL}/provenance.json")"
training_commit_c="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_commit"])' "${OUTPUT_C_REL}/provenance.json")"
if [[ -z "${training_commit_a}" || "${training_commit_a}" != "${training_commit_c}" ]]; then
  echo "A and C do not record the same source project commit." >&2
  exit 1
fi
if ! git cat-file -e "${training_commit_a}^{commit}"; then
  echo "The M4 source commit is unavailable locally: ${training_commit_a}" >&2
  exit 1
fi
if ! git merge-base --is-ancestor "${training_commit_a}" HEAD; then
  echo "Current HEAD does not descend from the M4 source commit ${training_commit_a}." >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/ruff check .
.venv/bin/pytest
.venv/bin/python -m compileall -q src tools

mkdir -p "${M5_OUTPUT_REL}"
cp "${M4_OUTPUT_REL}/verification_report.json" "${M5_OUTPUT_REL}/source_m4_verification_report.json"
cp "${M4_OUTPUT_REL}/train_vs_rsicd_val_overlap.json" "${M5_OUTPUT_REL}/train_vs_rsicd_val_overlap.json"
cp "${M4_OUTPUT_REL}/rsicd_val_official.json" "${M5_OUTPUT_REL}/rsicd_val_official.json"
cp "${M4_OUTPUT_REL}/rsicd_val_a_step100.json" "${M5_OUTPUT_REL}/rsicd_val_a_step100.json"
cp "${M4_OUTPUT_REL}/rsicd_val_c_step100.json" "${M5_OUTPUT_REL}/rsicd_val_c_step100.json"
cp "${OUTPUT_A_REL}/verification_report.json" "${M5_OUTPUT_REL}/a_100step_training_verification.json"
cp "${OUTPUT_C_REL}/verification_report.json" "${M5_OUTPUT_REL}/c_100step_training_verification.json"

current_project_commit="$(git rev-parse HEAD)"
{
  echo "runner_commit=${current_project_commit}"
  echo "training_commit=${training_commit_a}"
  echo "source_m4_report=${M4_OUTPUT_REL}/verification_report.json"
  echo "retention_margin_absolute_mean_recall=${RETENTION_MARGIN}"
  echo "configured_target_steps=5000"
  echo "execution_cap_steps=250"
  echo "resume_step=100"
} > "${M5_OUTPUT_REL}/preflight.txt"

temporary_parent="$(mktemp -d)"
historical_root="${temporary_parent}/m4-source"
cleanup_historical_worktree() {
  if [[ -d "${historical_root}" ]]; then
    git worktree remove --force "${historical_root}" >/dev/null 2>&1 || true
  fi
  rmdir "${temporary_parent}" >/dev/null 2>&1 || true
}
trap cleanup_historical_worktree EXIT
git worktree add --detach "${historical_root}" "${training_commit_a}"

run_training() {
  local label="$1"
  local config="$2"
  local output="$3"

  PYTHONPATH="${historical_root}/src" "${PROJECT_ROOT}/.venv/bin/python" \
    -m dinotxt_rs.cli.train \
    --config "${historical_root}/${config}" \
    --resume "${PROJECT_ROOT}/${output}/step_0000100.pt" \
    --stop-after-step 250 \
    2>&1 | tee "${output}/train_phase_3_step100_to250.log"
  "${PROJECT_ROOT}/.venv/bin/python" tools/verify_training_run.py \
    --output "${output}" \
    --expected-steps 250 \
    --expected-target-steps 5000 \
    --require-incomplete \
    --expected-train-manifest-sha256 "${EXPECTED_TRAIN_MANIFEST_SHA256}" \
    --expected-dinov3-commit "${EXPECTED_DINOV3_COMMIT}" \
    --expected-final-queue-size 4096 \
    --required-checkpoint-step 0 \
    --required-checkpoint-step 50 \
    --required-checkpoint-step 100 \
    --required-checkpoint-step 150 \
    --required-checkpoint-step 200 \
    --required-checkpoint-step 250 \
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
    --required-resume-step 100 \
    | tee "${output}/verification_report.json"
  echo "continued_run=${label} step=250"
}

run_training "a" "${CONFIG_A_REL}" "${OUTPUT_A_REL}"
run_training "c" "${CONFIG_C_REL}" "${OUTPUT_C_REL}"

evaluate_val() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local training_output="$4"
  "${PROJECT_ROOT}/.venv/bin/python" -m dinotxt_rs.cli.evaluate_rsicd \
    --config "${config}" \
    --manifest "assets/data/manifests/rsicd_val_retrieval_v1.jsonl" \
    --split val \
    --output "${M5_OUTPUT_REL}/rsicd_val_${label}.json" \
    --batch-size 64 \
    --num-workers 4 \
    --retrieval-chunk-size 256 \
    --checkpoint "${checkpoint}" \
    --training-output "${training_output}" \
    2>&1 | tee "${M5_OUTPUT_REL}/rsicd_val_${label}.log"
}

for step in 150 200 250; do
  checkpoint_name="step_$(printf '%07d' "${step}").pt"
  evaluate_val "a_step${step}" "${CONFIG_A_REL}" "${OUTPUT_A_REL}/${checkpoint_name}" "${OUTPUT_A_REL}"
  evaluate_val "c_step${step}" "${CONFIG_C_REL}" "${OUTPUT_C_REL}/${checkpoint_name}" "${OUTPUT_C_REL}"
done

"${PROJECT_ROOT}/.venv/bin/python" tools/verify_m5_ac_250step_development.py \
  --development-output "${M5_OUTPUT_REL}" \
  --run-a "${OUTPUT_A_REL}" \
  --run-c "${OUTPUT_C_REL}" \
  --retention-margin "${RETENTION_MARGIN}" \
  | tee "${M5_OUTPUT_REL}/verification_report.json"
