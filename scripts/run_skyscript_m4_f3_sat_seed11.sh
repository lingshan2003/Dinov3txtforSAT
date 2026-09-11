#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it with bash." >&2
  return 2
fi

set -Eeuo pipefail

on_error() {
  local status=$?
  echo "SAT F3 text-side mechanism screen stopped at line ${BASH_LINENO[0]} (exit=${status})." >&2
  echo "Completed boundaries and reports are reusable; rerun after fixing the cause." >&2
  exit "${status}"
}
trap on_error ERR

usage() {
  cat <<'EOF'
Usage: bash scripts/run_skyscript_m4_f3_sat_seed11.sh [options]

Options:
  --rsicd-root PATH  RSICD root (default: assets/data/raw/rsicd)
  --batch-size N     Retrieval/text-drift forward batch size (default: 64)
  --num-workers N    Retrieval image workers (default: 4)
  --chunk-size N     Retrieval score chunk size (default: 256)
  -h, --help         Show this help.

Runs both SAT F3 seed11 learning-rate candidates all the way through 500 steps,
then evaluates steps 0/100/250/500 and writes one final summary. F0 is reused and
never retrained. The SAT visual backbone and official dino.txt vision head remain
frozen; only the image adapter and text projection are trainable. Run inside tmux.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RSICD_ROOT="assets/data/raw/rsicd"
BATCH_SIZE=64
NUM_WORKERS=4
CHUNK_SIZE=256

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rsicd-root) RSICD_ROOT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "${BATCH_SIZE}" "${NUM_WORKERS}" "${CHUNK_SIZE}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Numeric options must be integers." >&2; exit 2; }
done
(( BATCH_SIZE > 0 && CHUNK_SIZE > 0 )) || {
  echo "Batch and chunk sizes must be positive." >&2
  exit 2
}

cd "${PROJECT_ROOT}"
[[ -x .venv/bin/python ]] || {
  echo "Missing .venv; run scripts/bootstrap_autodl.sh first." >&2
  exit 1
}
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to mix the experiment with uncommitted or untracked project changes:" >&2
  git status --short >&2
  exit 1
fi

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

F0_SUMMARY="outputs/skyscript_gate_m4_sat_seed11/summary.json"
STUDY_DIR="outputs/skyscript_m4_f3_sat_seed11"
TRAIN_MANIFEST="assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl"
VAL_MANIFEST="assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
RSICD_ANNOTATION="${RSICD_ROOT}/dataset_rsicd.json"
RSICD_MANIFEST="assets/data/manifests/rsicd_val_retrieval_v1.jsonl"
RSICD_AUDIT="assets/data/manifests/rsicd_val_retrieval_v1.audit.json"
SAT_WEIGHTS="assets/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
DINOTXT_WEIGHTS="assets/checkpoints/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
BPE_VOCAB="assets/checkpoints/bpe_simple_vocab_16e6.txt.gz"

EXPECTED_F0_SHA="7723b4cd6a4851e2c766ebd18aae0a97102b5cfee37f465df2d9309ddf55f2be"
EXPECTED_TRAIN_SHA="4264d884d564631816baf3e339e7ca12be2958966d99a60f6a75ae3d5d4dbaec"
EXPECTED_VAL_SHA="062238716b8fddad890b4f97354588ad7524f1099667d3c0baa932002845b7f6"
EXPECTED_DINOV3_COMMIT="6876159a11b4df116f30f667f8c9888617df0751"
EXPECTED_RSICD_ANNOTATION_SHA="5e342037d469d074711676bdb9c02b6942a624530b1959d24d2734e68af9cede"
EXPECTED_SAT_WEIGHTS_SHA="eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48"
EXPECTED_DINOTXT_WEIGHTS_SHA="a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0"
EXPECTED_BPE_VOCAB_SHA="924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"

declare -a CANDIDATES=(
  "f3_adapter256_textproj_lr1e5"
  "f3_adapter256_textproj_lr5e6"
)
declare -A CONFIGS=(
  [f3_adapter256_textproj_lr1e5]="configs/skyscript_sat_f3_adapter256_textproj_lr1e5_500step_seed11.toml"
  [f3_adapter256_textproj_lr5e6]="configs/skyscript_sat_f3_adapter256_textproj_lr5e6_500step_seed11.toml"
)

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }; }
require_hash() {
  local observed
  observed="$(sha256_file "$1")"
  [[ "${observed}" == "$2" ]] || {
    echo "Unexpected $3: expected $2, got ${observed}" >&2
    exit 1
  }
}

for path in \
  "${F0_SUMMARY}" \
  "${TRAIN_MANIFEST}" \
  "${VAL_MANIFEST}" \
  "${RSICD_ANNOTATION}" \
  "${SAT_WEIGHTS}" \
  "${DINOTXT_WEIGHTS}" \
  "${BPE_VOCAB}"; do
  require_file "${path}"
done
for candidate in "${CANDIDATES[@]}"; do require_file "${CONFIGS[${candidate}]}"; done
require_hash "${F0_SUMMARY}" "${EXPECTED_F0_SHA}" "F0 summary SHA-256"
require_hash "${TRAIN_MANIFEST}" "${EXPECTED_TRAIN_SHA}" "training manifest SHA-256"
require_hash "${VAL_MANIFEST}" "${EXPECTED_VAL_SHA}" "validation manifest SHA-256"
require_hash "${RSICD_ANNOTATION}" "${EXPECTED_RSICD_ANNOTATION_SHA}" "RSICD annotation SHA-256"
require_hash "${SAT_WEIGHTS}" "${EXPECTED_SAT_WEIGHTS_SHA}" "SAT weights SHA-256"
require_hash "${DINOTXT_WEIGHTS}" "${EXPECTED_DINOTXT_WEIGHTS_SHA}" "dino.txt weights SHA-256"
require_hash "${BPE_VOCAB}" "${EXPECTED_BPE_VOCAB_SHA}" "BPE vocabulary SHA-256"
[[ "$(git -C external/dinov3 rev-parse HEAD)" == "${EXPECTED_DINOV3_COMMIT}" ]] || {
  echo "Unexpected DINOv3 upstream commit." >&2
  exit 1
}

python tools/verify_skyscript_m4_f3_configs.py
python - "${F0_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["gate"] == "SkyScript_M4_SAT"
assert report["status"] == "pass"
assert report["stages"] == [100, 250, 500]
print("f0=verified_reused_not_retrained")
PY

echo "Running code checks once before GPU work..."
ruff check .
pytest
python -m compileall -q src tools

mkdir -p "${STUDY_DIR}"
if [[ ! -e "${RSICD_MANIFEST}" && ! -e "${RSICD_AUDIT}" ]]; then
  python tools/prepare_rsicd_retrieval_manifest.py \
    --annotations "${RSICD_ANNOTATION}" \
    --images-root "${RSICD_ROOT}" \
    --split val \
    --output "${RSICD_MANIFEST}" \
    --audit-output "${RSICD_AUDIT}"
elif [[ ! -f "${RSICD_MANIFEST}" || ! -f "${RSICD_AUDIT}" ]]; then
  echo "RSICD manifest and audit must exist together." >&2
  exit 1
fi

current_step() {
  local run_dir="$1"
  if [[ ! -f "${run_dir}/training_summary.json" ]]; then
    if [[ -e "${run_dir}/metrics.jsonl" || -e "${run_dir}/config.toml" ]] \
      || compgen -G "${run_dir}/step_*.pt" > /dev/null; then
      echo "Partial run lacks training_summary.json: ${run_dir}" >&2
      exit 1
    fi
    echo 0
    return
  fi
  python - "${run_dir}/training_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
state = (summary.get("steps"), summary.get("completed"), summary.get("target_steps"))
if state not in {(100, False, 500), (250, False, 500), (500, True, 500)}:
    raise ValueError(f"Unexpected staged training state: {state}")
assert summary.get("visual_backbone_permanently_frozen") is True
print(summary["steps"])
PY
}

validate_resume_tail() {
  local config="$1"
  local run_dir="$2"
  local step="$3"
  python - "${config}" "${run_dir}" "${step}" <<'PY'
import json
import sys
from pathlib import Path

config = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
step = int(sys.argv[3])
checkpoint = run_dir / f"step_{step:07d}.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint}")
if list(run_dir.glob("*.part")):
    raise ValueError("Incomplete atomic output prevents safe resume")
if (run_dir / "config.toml").read_bytes() != config.read_bytes():
    raise ValueError("Run config snapshot differs from the registered candidate config")
metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
if [record.get("step") for record in metrics] != list(range(1, step + 1)):
    raise ValueError("metrics.jsonl contains an interrupted tail; preserve and inspect it")
validation = [
    json.loads(line) for line in (run_dir / "validation.jsonl").read_text().splitlines()
]
if [record.get("step") for record in validation] != list(range(0, step + 1, 50)):
    raise ValueError("validation.jsonl contains an interrupted tail; preserve and inspect it")
history_path = run_dir / "resume_history.jsonl"
history = (
    [json.loads(line) for line in history_path.read_text().splitlines()]
    if history_path.is_file()
    else []
)
expected = [] if step == 100 else [100]
if [record.get("checkpoint_step") for record in history] != expected:
    raise ValueError("Resume history differs from the registered staged protocol")
print(f"resume_tail=verified step={step}")
PY
}

train_candidate() {
  local candidate="$1"
  local config="${CONFIGS[${candidate}]}"
  local run_dir="outputs/skyscript_sat_${candidate}_36495_500step_seed11"
  local observed
  observed="$(current_step "${run_dir}")"
  mkdir -p "${run_dir}"
  if [[ "${observed}" == "0" ]]; then
    python -m dinotxt_rs.cli.smoke_model --config "${config}" --batch-size 16 \
      2>&1 | tee "${run_dir}/smoke.log"
    python -m dinotxt_rs.cli.train --config "${config}" --stop-after-step 100 \
      2>&1 | tee "${run_dir}/train_phase_000_to_100.log"
    observed=100
  fi
  if [[ "${observed}" == "100" ]]; then
    validate_resume_tail "${config}" "${run_dir}" 100
    python -m dinotxt_rs.cli.train --config "${config}" \
      --resume "${run_dir}/step_0000100.pt" --stop-after-step 250 \
      2>&1 | tee "${run_dir}/train_phase_100_to_250.log"
    observed=250
  fi
  if [[ "${observed}" == "250" ]]; then
    validate_resume_tail "${config}" "${run_dir}" 250
    python -m dinotxt_rs.cli.train --config "${config}" \
      --resume "${run_dir}/step_0000250.pt" \
      2>&1 | tee "${run_dir}/train_phase_250_to_500.log"
  fi
  [[ "$(current_step "${run_dir}")" == "500" ]] || {
    echo "Candidate did not reach step500: ${candidate}" >&2
    exit 1
  }
}

verify_candidate() {
  local candidate="$1"
  local run_dir="outputs/skyscript_sat_${candidate}_36495_500step_seed11"
  local gate_dir="outputs/skyscript_gate_m4_${candidate}_sat_seed11"
  local report="${gate_dir}/training_step500.json"
  mkdir -p "${gate_dir}"
  if [[ ! -e "${report}" ]]; then
    local args=(
      tools/verify_training_run.py
      --output "${run_dir}"
      --expected-steps 500
      --expected-target-steps 500
      --expected-train-manifest-sha256 "${EXPECTED_TRAIN_SHA}"
      --expected-val-manifest-sha256 "${EXPECTED_VAL_SHA}"
      --expected-dinov3-commit "${EXPECTED_DINOV3_COMMIT}"
      --expected-final-queue-size 0
      --require-completed
      --require-validation
      --validation-every 50
      --require-best-checkpoint
      --expected-validation-loss-batch-size 16
      --expected-validation-forward-batch-size 64
      --required-resume-step 100
      --required-resume-step 250
      --require-visual-backbone-frozen
      --required-optimizer-group image_adapter
      --required-optimizer-group text_projection
    )
    local checkpoint
    for (( checkpoint=0; checkpoint<=500; checkpoint+=50 )); do
      args+=(--required-checkpoint-step "${checkpoint}")
    done
    python "${args[@]}" > "${report}.part"
    mv "${report}.part" "${report}"
  fi
}

evaluate_candidate() {
  local candidate="$1"
  local config="${CONFIGS[${candidate}]}"
  local run_dir="outputs/skyscript_sat_${candidate}_36495_500step_seed11"
  local gate_dir="outputs/skyscript_gate_m4_${candidate}_sat_seed11"
  local parity="${run_dir}/skyscript_step0_parity.json"
  if [[ ! -e "${parity}" ]]; then
    python -m dinotxt_rs.cli.check_step0_parity \
      --config "${run_dir}/config.toml" \
      --checkpoint "${run_dir}/step_0000000.pt" \
      --training-output "${run_dir}" \
      --input-manifest "${VAL_MANIFEST}" \
      --batch-size 16 \
      --output "${parity}"
  fi
  for step in 0 100 250 500; do
    local sky="${gate_dir}/skyscript_val_step$(printf '%03d' "${step}").json"
    local rsicd="${gate_dir}/rsicd_val_step$(printf '%03d' "${step}").json"
    if [[ ! -e "${sky}" ]]; then
      python -m dinotxt_rs.cli.evaluate_skyscript \
        --config "${run_dir}/config.toml" --manifest "${VAL_MANIFEST}" --split val \
        --checkpoint "${run_dir}/step_$(printf '%07d' "${step}").pt" \
        --training-output "${run_dir}" --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" --retrieval-chunk-size "${CHUNK_SIZE}" \
        --output "${sky}"
    fi
    if [[ ! -e "${rsicd}" ]]; then
      python -m dinotxt_rs.cli.evaluate_rsicd \
        --config "${run_dir}/config.toml" --manifest "${RSICD_MANIFEST}" --split val \
        --checkpoint "${run_dir}/step_$(printf '%07d' "${step}").pt" \
        --training-output "${run_dir}" --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" --retrieval-chunk-size "${CHUNK_SIZE}" \
        --output "${rsicd}"
    fi
  done
  local official="${gate_dir}/rsicd_val_official.json"
  if [[ ! -e "${official}" ]]; then
    python -m dinotxt_rs.cli.evaluate_rsicd \
      --config "${config}" --manifest "${RSICD_MANIFEST}" --split val \
      --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
      --retrieval-chunk-size "${CHUNK_SIZE}" --output "${official}"
  fi
  local drift="${gate_dir}/text_embedding_drift.json"
  if [[ ! -e "${drift}" ]]; then
    python -m dinotxt_rs.cli.evaluate_text_drift \
      --config "${run_dir}/config.toml" \
      --training-output "${run_dir}" \
      --manifest "skyscript_val=${VAL_MANIFEST}" \
      --manifest "rsicd_val=${RSICD_MANIFEST}" \
      --checkpoint "0=${run_dir}/step_0000000.pt" \
      --checkpoint "100=${run_dir}/step_0000100.pt" \
      --checkpoint "250=${run_dir}/step_0000250.pt" \
      --checkpoint "500=${run_dir}/step_0000500.pt" \
      --batch-size "${BATCH_SIZE}" \
      --output "${drift}"
  fi
}

for candidate in "${CANDIDATES[@]}"; do
  echo "===== SAT text-side candidate: ${candidate} ====="
  train_candidate "${candidate}"
  verify_candidate "${candidate}"
  evaluate_candidate "${candidate}"
done

python tools/summarize_skyscript_m4_f3_sat.py \
  --baseline "${F0_SUMMARY}" \
  --candidate-root outputs \
  --output-root outputs \
  --output "${STUDY_DIR}/summary.json"

echo "SAT F3 text-projection mechanism screen complete."
echo "Summary: ${STUDY_DIR}/summary.json"
