#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run: bash scripts/run_skyscript_gate_s0.sh" >&2
  return 2
fi

set -Eeuo pipefail

on_error() {
  local status=$?
  echo "Gate S0 stopped at line ${BASH_LINENO[0]} (exit=${status})." >&2
  echo "The parent terminal remains open; inspect the last error and saved attempt log." >&2
  exit "${status}"
}
trap on_error ERR

usage() {
  cat <<'EOF'
Usage: bash scripts/run_skyscript_gate_s0.sh [options]

Options:
  --run-dir PATH       Existing SkyScript 100-step training output.
  --s0-dir PATH        Directory for Gate S0 evaluation reports.
  --rsicd-root PATH    RSICD root containing dataset_rsicd.json and images.
  --batch-size N       Evaluation forward batch size (default: 64).
  --num-workers N      Evaluation image workers (default: 4).
  --chunk-size N       Retrieval score chunk size (default: 256).
  -h, --help           Show this help.

Run this script with bash; do not source it into the current terminal.
Existing final reports are validated and reused, never overwritten.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed11"
S0_DIR="outputs/skyscript_gate_s0_seed11"
RSICD_ROOT="assets/data/raw/rsicd"
BATCH_SIZE=64
NUM_WORKERS=4
CHUNK_SIZE=256

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --s0-dir)
      S0_DIR="$2"
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

cd "${PROJECT_ROOT}"

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

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv; run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TRAIN_MANIFEST="assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl"
VAL_MANIFEST="assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
RSICD_ANNOTATION="${RSICD_ROOT}/dataset_rsicd.json"
RSICD_VAL_MANIFEST="assets/data/manifests/rsicd_val_retrieval_v1.jsonl"
RSICD_VAL_AUDIT="assets/data/manifests/rsicd_val_retrieval_v1.audit.json"

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

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing Gate S0 with uncommitted or untracked project changes:" >&2
  git status --short >&2
  exit 1
fi

require_file "${RUN_DIR}/config.toml"
require_file "${RUN_DIR}/provenance.json"
require_file "${TRAIN_MANIFEST}"
require_file "${VAL_MANIFEST}"
require_file "${RSICD_ANNOTATION}"
for step in 0 25 50 75 100; do
  require_file "${RUN_DIR}/step_$(printf '%07d' "${step}").pt"
done

require_equal "$(git -C external/dinov3 rev-parse HEAD)" "${EXPECTED_DINOV3_COMMIT}" \
  "DINOv3 commit"
require_equal "$(sha256_file "${TRAIN_MANIFEST}")" "${EXPECTED_TRAIN_SHA}" \
  "training manifest SHA-256"
require_equal "$(sha256_file "${VAL_MANIFEST}")" "${EXPECTED_VAL_SHA}" \
  "validation manifest SHA-256"
require_equal "$(sha256_file "${RSICD_ANNOTATION}")" "${EXPECTED_RSICD_ANNOTATION_SHA}" \
  "RSICD annotation SHA-256"

echo "Running code checks..."
ruff check .
pytest
python -m compileall -q src tools

mkdir -p "${S0_DIR}" "${S0_DIR}/quarantine"

shopt -s nullglob
for partial in "${RUN_DIR}"/*.part; do
  destination="${S0_DIR}/quarantine/$(basename "${partial}").$(date +%Y%m%dT%H%M%S)"
  echo "Moving incomplete artifact out of the training directory: ${partial} -> ${destination}"
  mv "${partial}" "${destination}"
done
shopt -u nullglob

TRAINING_REPORT="${RUN_DIR}/verification_report.json"
if [[ ! -e "${TRAINING_REPORT}" ]]; then
  verification_tmp="$(mktemp "${S0_DIR}/.training-verification.XXXXXX")"
  echo "Verifying the completed training run..."
  if python tools/verify_training_run.py \
    --output "${RUN_DIR}" \
    --expected-steps 100 \
    --expected-target-steps 100 \
    --require-completed \
    --expected-train-manifest-sha256 "${EXPECTED_TRAIN_SHA}" \
    --expected-val-manifest-sha256 "${EXPECTED_VAL_SHA}" \
    --expected-dinov3-commit "${EXPECTED_DINOV3_COMMIT}" \
    --expected-final-queue-size 0 \
    --required-checkpoint-step 0 \
    --required-checkpoint-step 25 \
    --required-checkpoint-step 50 \
    --required-checkpoint-step 75 \
    --required-checkpoint-step 100 \
    --require-validation \
    --validation-every 25 \
    --require-best-checkpoint \
    --expected-validation-loss-batch-size 16 \
    --expected-validation-forward-batch-size 64 \
    > "${verification_tmp}"; then
    mv "${verification_tmp}" "${TRAINING_REPORT}"
  else
    status=$?
    echo "Training verification failed; stdout was preserved at ${verification_tmp}" >&2
    exit "${status}"
  fi
else
  echo "Reusing existing training verification: ${TRAINING_REPORT}"
fi

python - "${TRAINING_REPORT}" "${EXPECTED_TRAIN_SHA}" "${EXPECTED_VAL_SHA}" \
  "${EXPECTED_DINOV3_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["steps"] == 100
assert report["completed"] is True
assert report["final_queue_size"] == 0
assert report["train_manifest_sha256"] == sys.argv[2]
assert report["val_manifest_sha256"] == sys.argv[3]
assert report["dinov3_commit"] == sys.argv[4]
assert report["best_validation_step"] == 100
print("training_artifacts=verified")
PY

OVERLAP_REPORT="outputs/skyscript_s0_train_val_overlap.json"
if [[ ! -e "${OVERLAP_REPORT}" ]]; then
  echo "Auditing SkyScript train/validation image overlap..."
  python tools/audit_manifest_image_overlap.py \
    --left-manifest "${TRAIN_MANIFEST}" \
    --right-manifest "${VAL_MANIFEST}" \
    --output "${OVERLAP_REPORT}" \
    --require-zero-overlap \
    2>&1 | tee "${S0_DIR}/train_vs_val_overlap.log"
else
  echo "Reusing existing overlap report: ${OVERLAP_REPORT}"
fi

python - "${OVERLAP_REPORT}" "${EXPECTED_TRAIN_SHA}" "${EXPECTED_VAL_SHA}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "clear"
assert report["exact_file_overlap_count"] == 0
assert report["exact_decoded_pixel_overlap_count"] == 0
assert report["left_manifest"]["sha256"] == sys.argv[2]
assert report["right_manifest"]["sha256"] == sys.argv[3]
print("skyscript_train_val_overlap=clear")
PY

PARITY_REPORT="${RUN_DIR}/skyscript_step0_parity.json"
if [[ ! -e "${PARITY_REPORT}" ]]; then
  parity_log="${S0_DIR}/parity.attempt-$(date +%Y%m%dT%H%M%S).log"
  echo "Running adapter-aware step-0 parity..."
  python -m dinotxt_rs.cli.check_step0_parity \
    --config "${RUN_DIR}/config.toml" \
    --checkpoint "${RUN_DIR}/step_0000000.pt" \
    --training-output "${RUN_DIR}" \
    --input-manifest "${VAL_MANIFEST}" \
    --batch-size 16 \
    --output "${PARITY_REPORT}" \
    2>&1 | tee "${parity_log}"
else
  echo "Reusing existing parity report: ${PARITY_REPORT}"
fi

python - "${PARITY_REPORT}" "${EXPECTED_VAL_SHA}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["format_version"] == 2
assert report["status"] == "pass"
assert report["checkpoint_step"] == 0
assert report["input"]["manifest_sha256"] == sys.argv[2]
assert report["tokens"]["passed"] is True
assert report["trainable_parameters"]["passed"] is True
assert all(item["passed"] is True for item in report["comparisons"].values())
print("step0_parity=passed")
PY

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
else
  echo "Reusing existing RSICD-val manifest: ${RSICD_VAL_MANIFEST}"
fi

RSICD_VAL_SHA="$(sha256_file "${RSICD_VAL_MANIFEST}")"
python - "${RSICD_VAL_AUDIT}" "${RSICD_VAL_SHA}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["dataset"] == "RSICD"
assert report["split"] == "val"
assert report["manifest"]["sha256"] == sys.argv[2]
assert report["images"] > 0 and report["captions"] > 0
print(f"rsicd_val_manifest=verified images={report['images']} captions={report['captions']}")
PY

RSICD_OVERLAP_REPORT="${S0_DIR}/train_vs_rsicd_val_overlap.json"
if [[ ! -e "${RSICD_OVERLAP_REPORT}" ]]; then
  echo "Auditing training/RSICD-val image overlap..."
  python tools/audit_manifest_image_overlap.py \
    --left-manifest "${TRAIN_MANIFEST}" \
    --right-manifest "${RSICD_VAL_MANIFEST}" \
    --output "${RSICD_OVERLAP_REPORT}" \
    --require-zero-overlap \
    2>&1 | tee "${S0_DIR}/train_vs_rsicd_val_overlap.log"
else
  echo "Reusing existing training/RSICD-val overlap report."
fi

python - "${RSICD_OVERLAP_REPORT}" "${EXPECTED_TRAIN_SHA}" "${RSICD_VAL_SHA}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "clear"
assert report["exact_file_overlap_count"] == 0
assert report["exact_decoded_pixel_overlap_count"] == 0
assert report["left_manifest"]["sha256"] == sys.argv[2]
assert report["right_manifest"]["sha256"] == sys.argv[3]
print("training_vs_rsicd_val_overlap=clear")
PY

write_preflight() {
  local destination="${S0_DIR}/preflight.txt"
  local temporary="${S0_DIR}/.preflight.expected"
  {
    echo "project_commit=$(git rev-parse HEAD)"
    echo "training_output=${RUN_DIR}"
    echo "train_manifest_sha256=${EXPECTED_TRAIN_SHA}"
    echo "skyscript_val_manifest_sha256=${EXPECTED_VAL_SHA}"
    echo "rsicd_val_manifest_sha256=${RSICD_VAL_SHA}"
    echo "batch_size=${BATCH_SIZE}"
    echo "num_workers=${NUM_WORKERS}"
    echo "retrieval_chunk_size=${CHUNK_SIZE}"
    echo "retention_margin_absolute_mean_recall=0.01"
  } > "${temporary}"
  if [[ -e "${destination}" ]]; then
    if ! cmp -s "${temporary}" "${destination}"; then
      echo "Existing S0 results belong to a different preflight identity:" >&2
      diff -u "${destination}" "${temporary}" >&2 || true
      mv "${temporary}" "${S0_DIR}/preflight.mismatch.$(date +%Y%m%dT%H%M%S)"
      exit 1
    fi
    rm "${temporary}"
    echo "Reusing matching preflight record: ${destination}"
  else
    shopt -s nullglob
    local orphaned_reports=(
      "${S0_DIR}"/skyscript_val_step*.json
      "${S0_DIR}"/rsicd_val_*.json
    )
    shopt -u nullglob
    if (( ${#orphaned_reports[@]} > 0 )); then
      echo "Evaluation reports exist without a preflight identity:" >&2
      printf '  %s\n' "${orphaned_reports[@]}" >&2
      mv "${temporary}" "${S0_DIR}/preflight.unpublished.$(date +%Y%m%dT%H%M%S)"
      exit 1
    fi
    mv "${temporary}" "${destination}"
  fi
}
write_preflight

validate_evaluation_report() {
  local report="$1"
  local task="$2"
  local manifest_sha="$3"
  local expected_step="$4"
  local expected_source="$5"
  python - "${report}" "${task}" "${manifest_sha}" "${expected_step}" \
    "${expected_source}" <<'PY'
import json
import math
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["task"] == sys.argv[2]
assert report["manifest"]["sha256"] == sys.argv[3]
assert report["split"] == "val"
checkpoint = report["model"]["checkpoint"]
if sys.argv[4] == "official":
    assert checkpoint is None
else:
    assert checkpoint["step"] == int(sys.argv[4])
if sys.argv[5]:
    assert report["sources"] == [sys.argv[5]]
metrics = report["metrics"]
for direction in ("image_to_text", "text_to_image"):
    for key in ("r1", "r5", "r10", "median_rank", "mean_rank"):
        assert math.isfinite(float(metrics[direction][key]))
assert math.isfinite(float(metrics["mean_recall"]))
print(f"evaluation_report=verified path={sys.argv[1]}")
PY
}

evaluate_skyscript_step() {
  local step="$1"
  local tag
  local checkpoint
  local output
  local log
  tag="$(printf '%03d' "${step}")"
  checkpoint="${RUN_DIR}/step_$(printf '%07d' "${step}").pt"
  output="${S0_DIR}/skyscript_val_step${tag}.json"
  if [[ ! -e "${output}" ]]; then
    log="${S0_DIR}/skyscript_val_step${tag}.attempt-$(date +%Y%m%dT%H%M%S).log"
    echo "Evaluating SkyScript validation retrieval at step ${step}..."
    python -m dinotxt_rs.cli.evaluate_skyscript \
      --config "${RUN_DIR}/config.toml" \
      --manifest "${VAL_MANIFEST}" \
      --split val \
      --checkpoint "${checkpoint}" \
      --training-output "${RUN_DIR}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --retrieval-chunk-size "${CHUNK_SIZE}" \
      --output "${output}" \
      2>&1 | tee "${log}"
  else
    echo "Reusing existing SkyScript step ${step} report."
  fi
  validate_evaluation_report "${output}" "paired_image_text_global_retrieval" \
    "${EXPECTED_VAL_SHA}" "${step}" "SkyScript"
}

evaluate_rsicd_official() {
  local output="${S0_DIR}/rsicd_val_official.json"
  local log
  if [[ ! -e "${output}" ]]; then
    log="${S0_DIR}/rsicd_val_official.attempt-$(date +%Y%m%dT%H%M%S).log"
    echo "Evaluating official initialization on RSICD-val..."
    python -m dinotxt_rs.cli.evaluate_rsicd \
      --config "${RUN_DIR}/config.toml" \
      --manifest "${RSICD_VAL_MANIFEST}" \
      --split val \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --retrieval-chunk-size "${CHUNK_SIZE}" \
      --output "${output}" \
      2>&1 | tee "${log}"
  else
    echo "Reusing existing RSICD-val official report."
  fi
  validate_evaluation_report "${output}" "rsicd_image_text_retrieval" \
    "${RSICD_VAL_SHA}" "official" ""
}

evaluate_rsicd_step() {
  local step="$1"
  local tag
  local checkpoint
  local output
  local log
  tag="$(printf '%03d' "${step}")"
  checkpoint="${RUN_DIR}/step_$(printf '%07d' "${step}").pt"
  output="${S0_DIR}/rsicd_val_step${tag}.json"
  if [[ ! -e "${output}" ]]; then
    log="${S0_DIR}/rsicd_val_step${tag}.attempt-$(date +%Y%m%dT%H%M%S).log"
    echo "Evaluating RSICD-val retention at step ${step}..."
    python -m dinotxt_rs.cli.evaluate_rsicd \
      --config "${RUN_DIR}/config.toml" \
      --manifest "${RSICD_VAL_MANIFEST}" \
      --split val \
      --checkpoint "${checkpoint}" \
      --training-output "${RUN_DIR}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --retrieval-chunk-size "${CHUNK_SIZE}" \
      --output "${output}" \
      2>&1 | tee "${log}"
  else
    echo "Reusing existing RSICD-val step ${step} report."
  fi
  validate_evaluation_report "${output}" "rsicd_image_text_retrieval" \
    "${RSICD_VAL_SHA}" "${step}" ""
}

for step in 0 25 50 75 100; do
  evaluate_skyscript_step "${step}"
done

evaluate_rsicd_official
for step in 0 25 50 75 100; do
  evaluate_rsicd_step "${step}"
done

if python - "${S0_DIR}" "${TRAINING_REPORT}" "${OVERLAP_REPORT}" \
  "${PARITY_REPORT}" "${RSICD_OVERLAP_REPORT}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
steps = (0, 25, 50, 75, 100)

def read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

training = read(sys.argv[2])
overlap = read(sys.argv[3])
parity = read(sys.argv[4])
rsicd_overlap = read(sys.argv[5])
sky = {step: read(root / f"skyscript_val_step{step:03d}.json")["metrics"] for step in steps}
rsicd_official = read(root / "rsicd_val_official.json")["metrics"]
rsicd = {step: read(root / f"rsicd_val_step{step:03d}.json")["metrics"] for step in steps}

sky_delta = sky[100]["mean_recall"] - sky[0]["mean_recall"]
rsicd_delta = rsicd[100]["mean_recall"] - rsicd_official["mean_recall"]
checks = {
    "training_artifacts": training["completed"] is True and training["steps"] == 100,
    "train_validation_overlap": overlap["status"] == "clear",
    "step0_parity": parity["status"] == "pass",
    "training_rsicd_val_overlap": rsicd_overlap["status"] == "clear",
    "skyscript_step100_mean_recall_above_step0": sky_delta > 0,
    "rsicd_step100_mean_recall_drop_within_0.01": rsicd_delta >= -0.01,
}
summary = {
    "format_version": 1,
    "gate": "SkyScript_S0",
    "status": "pass" if all(checks.values()) else "fail",
    "checks": checks,
    "criteria": {
        "skyscript_primary": "step100 mean_recall > step0 mean_recall",
        "rsicd_retention": "step100 mean_recall - official mean_recall >= -0.01",
    },
    "skyscript": {
        "steps": {str(step): sky[step] for step in steps},
        "step100_minus_step0_mean_recall": sky_delta,
    },
    "rsicd_val": {
        "official": rsicd_official,
        "steps": {str(step): rsicd[step] for step in steps},
        "step100_minus_official_mean_recall": rsicd_delta,
    },
}
destination = root / "summary.json"
temporary = destination.with_suffix(".json.part")
temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, destination)
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if summary["status"] == "pass" else 2)
PY
then
  echo "Gate S0 PASSED. Summary: ${S0_DIR}/summary.json"
else
  status=$?
  echo "Gate S0 completed but did not pass its numerical criteria." >&2
  echo "Review ${S0_DIR}/summary.json; do not start Gate S1." >&2
  exit "${status}"
fi
