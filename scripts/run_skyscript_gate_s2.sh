#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run: bash scripts/run_skyscript_gate_s2.sh" >&2
  return 2
fi

set -Eeuo pipefail

on_error() {
  local status=$?
  echo "Gate S2 stopped at line ${BASH_LINENO[0]} (exit=${status})." >&2
  echo "The parent terminal remains open; inspect the saved attempt log before retrying." >&2
  exit "${status}"
}
trap on_error ERR

usage() {
  cat <<'EOF'
Usage: bash scripts/run_skyscript_gate_s2.sh [options]

Options:
  --stop-after-stage N  Stop after passing stage 100, 250, or 500 (default: 500).
  --rsicd-root PATH     RSICD root (default: assets/data/raw/rsicd).
  --batch-size N        Evaluation forward batch size (default: 64).
  --num-workers N       Evaluation image workers (default: 4).
  --chunk-size N        Retrieval score chunk size (default: 256).
  -h, --help            Show this help.

This script creates a new 500-step seed-11 run from official initialization.
It intentionally stops training at steps 100 and 250, verifies and evaluates
each boundary, and resumes only after the current stage passes. Existing valid
stage artifacts are reused. It never resumes a completed 100-step S0/S1 run.

Run with bash; never source it. For a long SSH job, run it inside tmux.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOP_AFTER_STAGE=500
RSICD_ROOT="assets/data/raw/rsicd"
BATCH_SIZE=64
NUM_WORKERS=4
CHUNK_SIZE=256

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-after-stage)
      STOP_AFTER_STAGE="$2"
      shift 2
      ;;
    --rsicd-root)
      RSICD_ROOT="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${STOP_AFTER_STAGE}" != "100" && "${STOP_AFTER_STAGE}" != "250" \
  && "${STOP_AFTER_STAGE}" != "500" ]]; then
  echo "--stop-after-stage must be 100, 250, or 500." >&2
  exit 2
fi
for value in "${BATCH_SIZE}" "${NUM_WORKERS}" "${CHUNK_SIZE}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "Batch size, worker count, and chunk size must be nonnegative integers." >&2
    exit 2
  fi
done
if (( BATCH_SIZE == 0 || CHUNK_SIZE == 0 )); then
  echo "Batch size and chunk size must be positive." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv; run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing Gate S2 with uncommitted or untracked project changes:" >&2
  git status --short >&2
  echo "Commit and push the exact S2 code/config before starting the experiment." >&2
  exit 1
fi

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

BASELINE_CONFIG="configs/skyscript_web_adapter_100step.toml"
CONFIG="configs/skyscript_web_adapter_500step_seed11.toml"
RUN_DIR="outputs/skyscript_images23_top30raw_imageadapter256_36495_500step_seed11"
GATE_DIR="outputs/skyscript_gate_s2_seed11"
S1_SUMMARY="outputs/skyscript_gate_s1/summary.json"
TRAIN_MANIFEST="assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl"
VAL_MANIFEST="assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
RSICD_ANNOTATION="${RSICD_ROOT}/dataset_rsicd.json"
RSICD_VAL_MANIFEST="assets/data/manifests/rsicd_val_retrieval_v1.jsonl"
RSICD_VAL_AUDIT="assets/data/manifests/rsicd_val_retrieval_v1.audit.json"
TRAIN_VAL_SHARED="outputs/skyscript_s0_train_val_overlap.json"
TRAIN_VAL_REPORT="${GATE_DIR}/train_vs_val_overlap.json"
TRAIN_RSICD_REPORT="${GATE_DIR}/train_vs_rsicd_val_overlap.json"

EXPECTED_TRAIN_SHA="4264d884d564631816baf3e339e7ca12be2958966d99a60f6a75ae3d5d4dbaec"
EXPECTED_VAL_SHA="062238716b8fddad890b4f97354588ad7524f1099667d3c0baa932002845b7f6"
EXPECTED_DINOV3_COMMIT="6876159a11b4df116f30f667f8c9888617df0751"
EXPECTED_RSICD_ANNOTATION_SHA="5e342037d469d074711676bdb9c02b6942a624530b1959d24d2734e68af9cede"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_equal() {
  local observed="$1"
  local expected="$2"
  local label="$3"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "Unexpected ${label}: expected ${expected}, got ${observed}" >&2
    exit 1
  fi
}

copy_immutable() {
  local source="$1"
  local destination="$2"
  if [[ -e "${destination}" ]]; then
    if ! cmp -s "${source}" "${destination}"; then
      echo "Existing artifact differs from its source: ${destination}" >&2
      exit 1
    fi
    return
  fi
  local temporary
  temporary="$(mktemp "$(dirname "${destination}")/.copy.XXXXXX")"
  cp "${source}" "${temporary}"
  mv "${temporary}" "${destination}"
}

for path in \
  "${BASELINE_CONFIG}" \
  "${CONFIG}" \
  "${S1_SUMMARY}" \
  "${TRAIN_MANIFEST}" \
  "${VAL_MANIFEST}" \
  "${RSICD_ANNOTATION}"; do
  require_file "${path}"
done
require_equal "$(git -C external/dinov3 rev-parse HEAD)" "${EXPECTED_DINOV3_COMMIT}" \
  "DINOv3 commit"
require_equal "$(sha256_file "${TRAIN_MANIFEST}")" "${EXPECTED_TRAIN_SHA}" \
  "training manifest SHA-256"
require_equal "$(sha256_file "${VAL_MANIFEST}")" "${EXPECTED_VAL_SHA}" \
  "validation manifest SHA-256"
require_equal "$(sha256_file "${RSICD_ANNOTATION}")" "${EXPECTED_RSICD_ANNOTATION_SHA}" \
  "RSICD annotation SHA-256"

python - "${S1_SUMMARY}" "${BASELINE_CONFIG}" "${CONFIG}" <<'PY'
import copy
import json
import sys
import tomllib
from pathlib import Path

s1_path, baseline_path, s2_path = map(Path, sys.argv[1:])
s1 = json.loads(s1_path.read_text(encoding="utf-8"))
assert s1["gate"] == "SkyScript_S1" and s1["status"] == "pass"
assert s1["seeds"] == [11, 23, 47]
with baseline_path.open("rb") as handle:
    baseline = tomllib.load(handle)
with s2_path.open("rb") as handle:
    observed = tomllib.load(handle)
expected = copy.deepcopy(baseline)
expected["experiment"] = {
    "name": "skyscript_images23_top30raw_imageadapter256_36495_500step_seed11",
    "seed": 11,
    "output_dir": "outputs/skyscript_images23_top30raw_imageadapter256_36495_500step_seed11",
}
expected["train"]["max_steps"] = 500
expected["train"]["warmup_steps"] = 50
expected["train"]["validation_every"] = 50
expected["train"]["checkpoint_every"] = 50
if observed != expected:
    raise RuntimeError("S2 config differs from the frozen protocol outside approved fields")
print("s1_prerequisite=passed s2_config_protocol=verified")
PY

echo "Running code checks once before GPU work..."
ruff check .
pytest
python -m compileall -q src tools

mkdir -p "${GATE_DIR}" "${GATE_DIR}/quarantine"

if [[ ! -e "${RSICD_VAL_MANIFEST}" && ! -e "${RSICD_VAL_AUDIT}" ]]; then
  echo "Preparing RSICD-val retrieval manifest..."
  python tools/prepare_rsicd_retrieval_manifest.py \
    --annotations "${RSICD_ANNOTATION}" \
    --images-root "${RSICD_ROOT}" \
    --split val \
    --output "${RSICD_VAL_MANIFEST}" \
    --audit-output "${RSICD_VAL_AUDIT}"
elif [[ ! -f "${RSICD_VAL_MANIFEST}" || ! -f "${RSICD_VAL_AUDIT}" ]]; then
  echo "RSICD-val manifest and audit must either both exist or both be absent." >&2
  exit 1
fi
RSICD_VAL_SHA="$(sha256_file "${RSICD_VAL_MANIFEST}")"

python - "${RSICD_VAL_AUDIT}" "${RSICD_VAL_SHA}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["dataset"] == "RSICD" and report["split"] == "val"
assert report["manifest"]["sha256"] == sys.argv[2]
assert report["images"] > 0 and report["captions"] > 0
print("rsicd_val_manifest=verified")
PY

write_preflight() {
  local destination="${GATE_DIR}/preflight.txt"
  local temporary="${GATE_DIR}/.preflight.expected"
  {
    echo "project_commit=$(git rev-parse HEAD)"
    echo "s1_summary=${S1_SUMMARY}"
    echo "s1_summary_sha256=$(sha256_file "${S1_SUMMARY}")"
    echo "config=${CONFIG}"
    echo "config_sha256=$(sha256_file "${CONFIG}")"
    echo "training_output=${RUN_DIR}"
    echo "train_manifest_sha256=${EXPECTED_TRAIN_SHA}"
    echo "skyscript_val_manifest_sha256=${EXPECTED_VAL_SHA}"
    echo "rsicd_val_manifest_sha256=${RSICD_VAL_SHA}"
    echo "stages=100,250,500"
    echo "validation_every=50"
    echo "warmup_steps=50"
    echo "batch_size=${BATCH_SIZE}"
    echo "num_workers=${NUM_WORKERS}"
    echo "retrieval_chunk_size=${CHUNK_SIZE}"
    echo "retention_margin_absolute_mean_recall=0.01"
  } > "${temporary}"
  if [[ -e "${destination}" ]]; then
    if ! cmp -s "${temporary}" "${destination}"; then
      echo "Existing S2 artifacts belong to a different preflight identity:" >&2
      diff -u "${destination}" "${temporary}" >&2 || true
      mv "${temporary}" "${GATE_DIR}/preflight.mismatch.$(date +%Y%m%dT%H%M%S)"
      exit 1
    fi
    rm "${temporary}"
  else
    mv "${temporary}" "${destination}"
  fi
}
write_preflight

if [[ ! -e "${TRAIN_VAL_REPORT}" ]]; then
  if [[ ! -e "${TRAIN_VAL_SHARED}" ]]; then
    echo "Auditing SkyScript train/validation overlap..."
    python tools/audit_manifest_image_overlap.py \
      --left-manifest "${TRAIN_MANIFEST}" \
      --right-manifest "${VAL_MANIFEST}" \
      --output "${TRAIN_VAL_SHARED}" \
      --require-zero-overlap \
      2>&1 | tee "${GATE_DIR}/train_vs_val_overlap.log"
  fi
  copy_immutable "${TRAIN_VAL_SHARED}" "${TRAIN_VAL_REPORT}"
fi

if [[ ! -e "${TRAIN_RSICD_REPORT}" ]]; then
  echo "Auditing training/RSICD-val image overlap..."
  python tools/audit_manifest_image_overlap.py \
    --left-manifest "${TRAIN_MANIFEST}" \
    --right-manifest "${RSICD_VAL_MANIFEST}" \
    --output "${TRAIN_RSICD_REPORT}" \
    --require-zero-overlap \
    2>&1 | tee "${GATE_DIR}/train_vs_rsicd_val_overlap.log"
fi

current_step() {
  if [[ ! -f "${RUN_DIR}/training_summary.json" ]]; then
    if [[ -e "${RUN_DIR}/config.toml" || -e "${RUN_DIR}/provenance.json" \
      || -e "${RUN_DIR}/metrics.jsonl" ]] \
      || compgen -G "${RUN_DIR}/step_*.pt" > /dev/null; then
      echo "The S2 run has partial artifacts without a training summary: ${RUN_DIR}" >&2
      echo "Automatic recovery is unsafe; preserve and inspect the directory first." >&2
      exit 1
    fi
    echo 0
    return
  fi
  python - "${RUN_DIR}/training_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = summary.get("steps")
completed = summary.get("completed")
valid = (step, completed) in {(100, False), (250, False), (500, True)}
if not valid or summary.get("target_steps") != 500:
    raise ValueError(f"Unexpected S2 training state: step={step}, completed={completed}")
print(step)
PY
}

validate_resume_tail() {
  local step="$1"
  python - "${CONFIG}" "${RUN_DIR}" "${step}" <<'PY'
import json
import sys
from pathlib import Path

config = Path(sys.argv[1]).resolve()
run_dir = Path(sys.argv[2]).resolve()
step = int(sys.argv[3])
checkpoint = run_dir / f"step_{step:07d}.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint}")
partials = sorted(run_dir.glob("*.part"))
if partials:
    raise ValueError(f"Incomplete atomic outputs prevent safe resume: {partials}")
if (run_dir / "config.toml").read_bytes() != config.read_bytes():
    raise ValueError("Run config snapshot does not exactly match the immutable S2 config")
metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
if [record.get("step") for record in metrics] != list(range(1, step + 1)):
    raise ValueError("metrics.jsonl does not end exactly at the resume checkpoint")
validation = [
    json.loads(line) for line in (run_dir / "validation.jsonl").read_text().splitlines()
]
if [record.get("step") for record in validation] != list(range(0, step + 1, 50)):
    raise ValueError("validation.jsonl does not end exactly at the resume checkpoint")
summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
if summary.get("steps") != step or summary.get("completed") is not False:
    raise ValueError("training_summary.json does not describe the resume checkpoint")
resume_path = run_dir / "resume_history.jsonl"
resume_history = (
    [json.loads(line) for line in resume_path.read_text().splitlines()]
    if resume_path.is_file()
    else []
)
observed_resumes = [record.get("checkpoint_step") for record in resume_history]
expected_resumes = [] if step == 100 else [100]
if observed_resumes != expected_resumes:
    raise ValueError(
        f"Resume history is {observed_resumes}, expected {expected_resumes}; retry is unsafe"
    )
print(f"resume_tail=verified step={step}")
PY
}

train_to_stage() {
  local target="$1"
  local observed
  observed="$(current_step)"
  if (( observed >= target )); then
    echo "Reusing S2 training state through step ${observed}."
    return
  fi
  if [[ "${observed}" == "0" && "${target}" == "100" ]]; then
    mkdir -p "${RUN_DIR}"
    echo "Running S2 smoke check..."
    python -m dinotxt_rs.cli.smoke_model --config "${CONFIG}" --batch-size 16 \
      2>&1 | tee "${RUN_DIR}/smoke.log"
    echo "Training the new S2 schedule from official initialization to step 100..."
    python -m dinotxt_rs.cli.train --config "${CONFIG}" --stop-after-step 100 \
      2>&1 | tee "${RUN_DIR}/train_phase_000_to_100.log"
  elif [[ "${observed}" == "100" && "${target}" == "250" ]]; then
    validate_resume_tail 100
    echo "Strictly resuming S2 from step 100 to step 250..."
    python -m dinotxt_rs.cli.train \
      --config "${CONFIG}" \
      --resume "${RUN_DIR}/step_0000100.pt" \
      --stop-after-step 250 \
      2>&1 | tee "${RUN_DIR}/train_phase_100_to_250.log"
  elif [[ "${observed}" == "250" && "${target}" == "500" ]]; then
    validate_resume_tail 250
    echo "Strictly resuming S2 from step 250 to step 500..."
    python -m dinotxt_rs.cli.train \
      --config "${CONFIG}" \
      --resume "${RUN_DIR}/step_0000250.pt" \
      2>&1 | tee "${RUN_DIR}/train_phase_250_to_500.log"
  else
    echo "Cannot progress from S2 step ${observed} to requested stage ${target}." >&2
    exit 1
  fi
  require_equal "$(current_step)" "${target}" "completed S2 phase boundary"
}

ensure_training_report() {
  local stage="$1"
  local report="${GATE_DIR}/training_step${stage}.json"
  if [[ -e "${report}" ]]; then
    echo "Reusing training verification for stage ${stage}."
    if [[ "${stage}" == "500" && ! -e "${RUN_DIR}/verification_report.json" ]]; then
      copy_immutable "${report}" "${RUN_DIR}/verification_report.json"
    fi
    return
  fi
  local observed
  observed="$(current_step)"
  if [[ "${observed}" != "${stage}" ]]; then
    echo "Cannot create stage ${stage} verification after training advanced to ${observed}." >&2
    exit 1
  fi
  local temporary
  temporary="$(mktemp "${GATE_DIR}/.training-step${stage}.XXXXXX")"
  local args=(
    tools/verify_training_run.py
    --output "${RUN_DIR}"
    --expected-steps "${stage}"
    --expected-target-steps 500
    --expected-train-manifest-sha256 "${EXPECTED_TRAIN_SHA}"
    --expected-val-manifest-sha256 "${EXPECTED_VAL_SHA}"
    --expected-dinov3-commit "${EXPECTED_DINOV3_COMMIT}"
    --expected-final-queue-size 0
    --require-validation
    --validation-every 50
    --require-best-checkpoint
    --expected-validation-loss-batch-size 16
    --expected-validation-forward-batch-size 64
  )
  local checkpoint_step
  for (( checkpoint_step=0; checkpoint_step<=stage; checkpoint_step+=50 )); do
    args+=(--required-checkpoint-step "${checkpoint_step}")
  done
  if [[ "${stage}" == "100" ]]; then
    args+=(--require-incomplete --forbid-resume)
  elif [[ "${stage}" == "250" ]]; then
    args+=(--require-incomplete --required-resume-step 100)
  else
    args+=(--require-completed --required-resume-step 100 --required-resume-step 250)
  fi
  echo "Verifying training artifacts at step ${stage}..."
  if python "${args[@]}" > "${temporary}"; then
    mv "${temporary}" "${report}"
  else
    status=$?
    echo "Training verification failed; stdout remains at ${temporary}" >&2
    exit "${status}"
  fi
  if [[ "${stage}" == "500" ]]; then
    if [[ -e "${RUN_DIR}/verification_report.json" ]]; then
      cmp -s "${report}" "${RUN_DIR}/verification_report.json" || {
        echo "Existing final verification report differs from S2 stage 500 report." >&2
        exit 1
      }
    else
      copy_immutable "${report}" "${RUN_DIR}/verification_report.json"
    fi
  fi
}

ensure_common_evidence() {
  local parity="${RUN_DIR}/skyscript_step0_parity.json"
  if [[ ! -e "${parity}" ]]; then
    echo "Running S2 step-0 parity..."
    python -m dinotxt_rs.cli.check_step0_parity \
      --config "${RUN_DIR}/config.toml" \
      --checkpoint "${RUN_DIR}/step_0000000.pt" \
      --training-output "${RUN_DIR}" \
      --input-manifest "${VAL_MANIFEST}" \
      --batch-size 16 \
      --output "${parity}" \
      2>&1 | tee "${GATE_DIR}/parity.attempt-$(date +%Y%m%dT%H%M%S).log"
  fi
}

evaluate_skyscript_step() {
  local step="$1"
  local output="${GATE_DIR}/skyscript_val_step$(printf '%03d' "${step}").json"
  if [[ ! -e "${output}" ]]; then
    echo "Evaluating SkyScript retrieval at S2 step ${step}..."
    python -m dinotxt_rs.cli.evaluate_skyscript \
      --config "${RUN_DIR}/config.toml" \
      --manifest "${VAL_MANIFEST}" \
      --split val \
      --checkpoint "${RUN_DIR}/step_$(printf '%07d' "${step}").pt" \
      --training-output "${RUN_DIR}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --retrieval-chunk-size "${CHUNK_SIZE}" \
      --output "${output}" \
      2>&1 | tee "${GATE_DIR}/skyscript_step${step}.attempt-$(date +%Y%m%dT%H%M%S).log"
  fi
}

evaluate_rsicd_official() {
  local output="${GATE_DIR}/rsicd_val_official.json"
  if [[ ! -e "${output}" ]]; then
    echo "Evaluating official initialization on RSICD-val..."
    python -m dinotxt_rs.cli.evaluate_rsicd \
      --config "${RUN_DIR}/config.toml" \
      --manifest "${RSICD_VAL_MANIFEST}" \
      --split val \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --retrieval-chunk-size "${CHUNK_SIZE}" \
      --output "${output}" \
      2>&1 | tee "${GATE_DIR}/rsicd_official.attempt-$(date +%Y%m%dT%H%M%S).log"
  fi
}

evaluate_rsicd_step() {
  local step="$1"
  local output="${GATE_DIR}/rsicd_val_step$(printf '%03d' "${step}").json"
  if [[ ! -e "${output}" ]]; then
    echo "Evaluating RSICD-val at S2 step ${step}..."
    python -m dinotxt_rs.cli.evaluate_rsicd \
      --config "${RUN_DIR}/config.toml" \
      --manifest "${RSICD_VAL_MANIFEST}" \
      --split val \
      --checkpoint "${RUN_DIR}/step_$(printf '%07d' "${step}").pt" \
      --training-output "${RUN_DIR}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --retrieval-chunk-size "${CHUNK_SIZE}" \
      --output "${output}" \
      2>&1 | tee "${GATE_DIR}/rsicd_step${step}.attempt-$(date +%Y%m%dT%H%M%S).log"
  fi
}

run_stage_gate() {
  local stage="$1"
  ensure_training_report "${stage}"
  ensure_common_evidence
  evaluate_skyscript_step 0
  evaluate_rsicd_official
  evaluate_rsicd_step 0
  evaluate_skyscript_step "${stage}"
  evaluate_rsicd_step "${stage}"

  local summary="${GATE_DIR}/stage_${stage}_summary.json"
  if python tools/summarize_skyscript_gate_s2.py stage \
      --run-dir "${RUN_DIR}" \
      --gate-dir "${GATE_DIR}" \
      --stage "${stage}" \
      --training-report "${GATE_DIR}/training_step${stage}.json" \
      --output "${summary}"; then
    echo "Gate S2 stage ${stage} PASSED."
  else
    local status=$?
    if [[ "${status}" -eq 2 && -f "${summary}" ]]; then
      echo "Gate S2 stage ${stage} FAILED its numerical criteria." >&2
      echo "Training will not continue beyond this boundary." >&2
    fi
    exit "${status}"
  fi
}

for stage in 100 250 500; do
  train_to_stage "${stage}"
  run_stage_gate "${stage}"
  if [[ "${STOP_AFTER_STAGE}" == "${stage}" ]]; then
    if [[ "${stage}" == "500" ]]; then
      if python tools/summarize_skyscript_gate_s2.py final \
          --s1-summary "${S1_SUMMARY}" \
          --gate-dir "${GATE_DIR}" \
          --output "${GATE_DIR}/summary.json"; then
        echo "Gate S2 PASSED. Summary: ${GATE_DIR}/summary.json"
      else
        status=$?
        echo "Gate S2 completed but did not pass all stages." >&2
        exit "${status}"
      fi
    else
      echo "Stopped after passing requested S2 stage ${stage}."
      echo "Run the same script with a later --stop-after-stage value to continue."
    fi
    exit 0
  fi
done
