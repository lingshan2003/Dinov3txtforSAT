#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run: bash scripts/run_skyscript_gate_s1.sh" >&2
  return 2
fi

set -Eeuo pipefail

on_error() {
  local status=$?
  echo "Gate S1 stopped at line ${BASH_LINENO[0]} (exit=${status})." >&2
  echo "Your parent terminal remains open. Fix the reported issue and run the script again." >&2
  exit "${status}"
}
trap on_error ERR

usage() {
  cat <<'EOF'
Usage: bash scripts/run_skyscript_gate_s1.sh [options]

Options:
  --seed 23|47|all       Seed(s) to train/evaluate (default: all).
  --skip-training        Only verify/evaluate existing completed runs.
  --resume-seed23 PATH   Strictly resume seed 23 from a consistent checkpoint.
  --resume-seed47 PATH   Strictly resume seed 47 from a consistent checkpoint.
  --rsicd-root PATH      RSICD root (default: assets/data/raw/rsicd).
  --batch-size N         Evaluation forward batch size (default: 64).
  --num-workers N        Evaluation image workers (default: 4).
  --chunk-size N         Retrieval score chunk size (default: 256).
  -h, --help             Show this help.

The script runs code/protocol checks once, trains from official initialization,
reuses complete verified artifacts, runs the per-seed S0-equivalent checks, and
writes the three-seed S1 summary once seeds 11, 23, and 47 are all available.

Run with bash; never source it. For a long SSH job, run it inside tmux.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_SEED="all"
SKIP_TRAINING=false
RESUME_SEED23=""
RESUME_SEED47=""
RSICD_ROOT="assets/data/raw/rsicd"
BATCH_SIZE=64
NUM_WORKERS=4
CHUNK_SIZE=256

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      TARGET_SEED="$2"
      shift 2
      ;;
    --skip-training)
      SKIP_TRAINING=true
      shift
      ;;
    --resume-seed23)
      RESUME_SEED23="$2"
      shift 2
      ;;
    --resume-seed47)
      RESUME_SEED47="$2"
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

if [[ "${TARGET_SEED}" != "23" && "${TARGET_SEED}" != "47" && "${TARGET_SEED}" != "all" ]]; then
  echo "--seed must be 23, 47, or all." >&2
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
if [[ "${TARGET_SEED}" == "23" && -n "${RESUME_SEED47}" ]]; then
  echo "--resume-seed47 cannot be used with --seed 23." >&2
  exit 2
fi
if [[ "${TARGET_SEED}" == "47" && -n "${RESUME_SEED23}" ]]; then
  echo "--resume-seed23 cannot be used with --seed 47." >&2
  exit 2
fi
if [[ "${SKIP_TRAINING}" == true && ( -n "${RESUME_SEED23}" || -n "${RESUME_SEED47}" ) ]]; then
  echo "Resume options cannot be combined with --skip-training." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv; run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing Gate S1 with uncommitted or untracked project changes:" >&2
  git status --short >&2
  echo "Commit and push the exact S1 code/config before starting the experiment." >&2
  exit 1
fi

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONFIG_11="configs/skyscript_web_adapter_100step.toml"
CONFIG_23="configs/skyscript_web_adapter_100step_seed23.toml"
CONFIG_47="configs/skyscript_web_adapter_100step_seed47.toml"
RUN_11="outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed11"
RUN_23="outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed23"
RUN_47="outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed47"
GATE_11="outputs/skyscript_gate_s0_seed11"
GATE_23="outputs/skyscript_gate_s1_seed23"
GATE_47="outputs/skyscript_gate_s1_seed47"
S1_DIR="outputs/skyscript_gate_s1"

for path in "${CONFIG_11}" "${CONFIG_23}" "${CONFIG_47}" "${GATE_11}/summary.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required S1 input: ${path}" >&2
    exit 1
  fi
done

python - "${CONFIG_11}" "${CONFIG_23}" "${CONFIG_47}" <<'PY'
import copy
import sys
import tomllib
from pathlib import Path

expected = {
    11: (
        "skyscript_images23_top30raw_imageadapter256_36495_100step_seed11",
        "outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed11",
    ),
    23: (
        "skyscript_images23_top30raw_imageadapter256_36495_100step_seed23",
        "outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed23",
    ),
    47: (
        "skyscript_images23_top30raw_imageadapter256_36495_100step_seed47",
        "outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed47",
    ),
}
configs = []
for seed, raw_path in zip((11, 23, 47), sys.argv[1:], strict=True):
    path = Path(raw_path)
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    experiment = config["experiment"]
    name, output_dir = expected[seed]
    assert experiment == {"name": name, "seed": seed, "output_dir": output_dir}, (
        f"Unexpected experiment identity in {path}: {experiment}"
    )
    normalized = copy.deepcopy(config)
    for field in ("name", "seed", "output_dir"):
        del normalized["experiment"][field]
    configs.append(normalized)
if configs[1:] != configs[:1] * 2:
    raise RuntimeError(
        "S1 configs differ outside experiment.name, experiment.seed, experiment.output_dir"
    )
print("s1_config_protocol=verified seeds=11,23,47")
PY

python - "${GATE_11}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["status"] == "pass"
assert all(summary["checks"].values())
print("seed11_gate_s0=verified")
PY

echo "Running code checks once before GPU work..."
ruff check .
pytest
python -m compileall -q src tools

is_complete_run() {
  local run_dir="$1"
  python - "${run_dir}/training_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
summary = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if summary.get("completed") is True and summary.get("steps") == 100 else 1)
PY
}

validate_resume_tail() {
  local config="$1"
  local run_dir="$2"
  local checkpoint="$3"
  python - "${config}" "${run_dir}" "${checkpoint}" <<'PY'
import json
import re
import sys
from pathlib import Path

config = Path(sys.argv[1]).resolve()
run_dir = Path(sys.argv[2]).resolve()
checkpoint = Path(sys.argv[3]).resolve()
if checkpoint.parent != run_dir or not checkpoint.is_file():
    raise ValueError(f"Resume checkpoint must be an existing file directly under {run_dir}")
match = re.fullmatch(r"step_(\d{7})\.pt", checkpoint.name)
if match is None:
    raise ValueError(f"Unexpected checkpoint name: {checkpoint.name}")
step = int(match.group(1))
if step not in (25, 50, 75):
    raise ValueError("S1 resume checkpoint must be step 25, 50, or 75")
snapshot = run_dir / "config.toml"
if not snapshot.is_file() or snapshot.read_bytes() != config.read_bytes():
    raise ValueError("Run config snapshot does not exactly match the selected immutable config")
metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
if [item.get("step") for item in metrics] != list(range(1, step + 1)):
    raise ValueError(
        "metrics.jsonl extends before/after the requested checkpoint; automatic resume is unsafe"
    )
validation = [
    json.loads(line) for line in (run_dir / "validation.jsonl").read_text().splitlines()
]
expected_validation = list(range(0, step + 1, 25))
if [item.get("step") for item in validation] != expected_validation:
    raise ValueError(
        "validation.jsonl does not end exactly at the requested checkpoint; resume is unsafe"
    )
print(f"resume_tail=verified step={step}")
PY
}

run_seed() {
  local seed="$1"
  local config="$2"
  local run_dir="$3"
  local gate_dir="$4"
  local resume_checkpoint="$5"

  if is_complete_run "${run_dir}"; then
    if [[ -n "${resume_checkpoint}" ]]; then
      echo "Seed ${seed} is already complete; remove its resume option." >&2
      exit 2
    fi
    echo "Reusing completed seed ${seed} training run: ${run_dir}"
  elif [[ "${SKIP_TRAINING}" == true ]]; then
    echo "Seed ${seed} has no completed 100-step run, but --skip-training was requested." >&2
    exit 1
  elif [[ -n "${resume_checkpoint}" ]]; then
    validate_resume_tail "${config}" "${run_dir}" "${resume_checkpoint}"
    local log
    log="${run_dir}/train.resume-$(date +%Y%m%dT%H%M%S).log"
    echo "Strictly resuming seed ${seed} from ${resume_checkpoint}..."
    python -m dinotxt_rs.cli.train --config "${config}" --resume "${resume_checkpoint}" \
      2>&1 | tee "${log}"
  else
    if [[ -e "${run_dir}/config.toml" || -e "${run_dir}/provenance.json" \
      || -e "${run_dir}/metrics.jsonl" ]] \
      || compgen -G "${run_dir}/step_*.pt" > /dev/null; then
      echo "Seed ${seed} has an incomplete existing run: ${run_dir}" >&2
      echo "Inspect it first. Resume only with --resume-seed${seed} and an exact safe tail." >&2
      exit 1
    fi
    mkdir -p "${run_dir}"
    echo "Running seed ${seed} smoke check..."
    python -m dinotxt_rs.cli.smoke_model --config "${config}" --batch-size 16 \
      2>&1 | tee "${run_dir}/smoke.log"
    echo "Training seed ${seed} from official initialization for 100 steps..."
    python -m dinotxt_rs.cli.train --config "${config}" \
      2>&1 | tee "${run_dir}/train.log"
  fi

  if ! is_complete_run "${run_dir}"; then
    echo "Seed ${seed} training did not produce a completed 100-step summary." >&2
    exit 1
  fi

  if bash scripts/run_skyscript_gate_s0.sh \
      --run-dir "${run_dir}" \
      --s0-dir "${gate_dir}" \
      --rsicd-root "${RSICD_ROOT}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --chunk-size "${CHUNK_SIZE}" \
      --skip-code-checks; then
    echo "Seed ${seed} gate passed."
  else
    local gate_status=$?
    if [[ "${gate_status}" -eq 2 && -f "${gate_dir}/summary.json" ]]; then
      echo "Seed ${seed} completed but failed a numerical gate; continuing other seeds." >&2
    else
      echo "Seed ${seed} gate stopped before producing a valid summary." >&2
      exit "${gate_status}"
    fi
  fi
}

if [[ "${TARGET_SEED}" == "23" || "${TARGET_SEED}" == "all" ]]; then
  run_seed 23 "${CONFIG_23}" "${RUN_23}" "${GATE_23}" "${RESUME_SEED23}"
fi
if [[ "${TARGET_SEED}" == "47" || "${TARGET_SEED}" == "all" ]]; then
  run_seed 47 "${CONFIG_47}" "${RUN_47}" "${GATE_47}" "${RESUME_SEED47}"
fi

if [[ -f "${GATE_23}/summary.json" && -f "${GATE_47}/summary.json" ]]; then
  mkdir -p "${S1_DIR}"
  if python tools/summarize_skyscript_gate_s1.py \
      --seed-result 11 "${RUN_11}" "${GATE_11}/summary.json" \
      --seed-result 23 "${RUN_23}" "${GATE_23}/summary.json" \
      --seed-result 47 "${RUN_47}" "${GATE_47}/summary.json" \
      --output "${S1_DIR}/summary.json"; then
    echo "Gate S1 PASSED. Summary: ${S1_DIR}/summary.json"
  else
    summary_status=$?
    if [[ "${summary_status}" -eq 2 && -f "${S1_DIR}/summary.json" ]]; then
      echo "Gate S1 completed but FAILED one or more numerical criteria." >&2
      echo "Keep all seed reports and do not start S2." >&2
      exit 2
    fi
    exit "${summary_status}"
  fi
else
  echo "Selected seed work passed, but both seed 23 and 47 reports are required for S1 aggregation."
fi
