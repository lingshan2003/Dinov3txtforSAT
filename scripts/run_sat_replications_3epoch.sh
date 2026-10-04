#!/usr/bin/env bash
# Reproduce three SAT candidates at seeds 23/47, then evaluate and package reports.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run with bash; do not source this script." >&2
  return 2
fi
set -Eeuo pipefail

EVALUATE=1
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/run_sat_replications_3epoch.sh [--train-only]"
  echo "Train adapter+LoRA, adapter-only, head+LoRA at seeds 23 and 47."
  echo "Default: also evaluate step0/best on SkyScript-val/RSICD-val and package reports."
  echo "Completed training is skipped; interrupted training resumes latest.pt."
  exit 0
fi
if [[ $# -eq 1 && "$1" == "--train-only" ]]; then
  EVALUATE=0
elif [[ $# -ne 0 ]]; then
  echo "Unknown arguments. Use --help." >&2
  exit 2
fi

trap 'echo "Replication stopped at line $LINENO. Inspect logs before rerunning." >&2' ERR
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing .venv/bin/python" >&2; exit 1; }
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

TRIALS=()
RUN_DIRS=()
for seed in 23 47; do
  for method in adapter_textlora adapter visionhead_textlora; do
    trial="skyscript_sat_${method}_3epoch_seed${seed}"
    TRIALS+=("${trial}")
    RUN_DIRS+=("outputs/${trial}")
  done
done

# Read-only preflight; use existing provenance checks in the trainer/evaluator.
"${PYTHON}" - "${EVALUATE}" "${TRIALS[@]}" <<'PY'
import sys
from pathlib import Path
from dinotxt_rs.config import load_config, required_paths

for trial in sys.argv[2:]:
    config = load_config(f"configs/{trial}.toml")
    if str(config.experiment.output_dir) != f"outputs/{trial}":
        raise ValueError(f"Unexpected output directory: {trial}")
    missing = [str(path) for path in required_paths(config) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    snapshot = config.experiment.output_dir / "config.toml"
    if snapshot.exists() and snapshot.read_bytes() != config.source.read_bytes():
        raise ValueError(f"Saved configuration differs: {snapshot}")
    print(f"preflight={trial} paths=present", flush=True)
if sys.argv[1] == "1":
    for manifest in (
        "skyscript_images23_val_raw_unique4055_seed23_global77.jsonl",
        "rsicd_val_retrieval_v1.jsonl",
    ):
        path = Path("assets/data/manifests") / manifest
        if not path.is_file():
            raise FileNotFoundError(path)
PY

for trial in "${TRIALS[@]}"; do
  config="configs/${trial}.toml"
  run_dir="outputs/${trial}"
  state="$("${PYTHON}" - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = root / "training_summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    if summary.get("completed") is True:
        if summary.get("steps") != 1710 or summary.get("target_steps") != 1710:
            raise ValueError(f"Unexpected completed training budget: {root}")
        for name in ("step_0000000.pt", "best.pt", "latest.pt"):
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
        print("complete")
        sys.exit(0)
if (root / "latest.pt").is_file():
    print("resume")
elif root.exists() and any(root.iterdir()):
    raise RuntimeError(
        f"Existing run has no latest.pt: {root}. Inspect it; if only step0 was saved, "
        "resume that checkpoint manually with the saved config.toml."
    )
else:
    print("fresh")
PY
)"
  if [[ "${state}" == "complete" ]]; then
    echo "Already completed: ${trial}"
    continue
  fi
  mkdir -p "${run_dir}"
  args=(--config "${config}")
  if [[ "${state}" == "resume" ]]; then
    args+=(--resume "${run_dir}/latest.pt")
  fi
  echo "Training ${trial} (${state})"
  "${PYTHON}" -u -m dinotxt_rs.cli.train "${args[@]}" 2>&1 | tee -a "${run_dir}/train.log"
  "${PYTHON}" - "${run_dir}/training_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if summary.get("completed") is not True or summary.get("steps") != 1710:
    raise RuntimeError(f"Training did not complete: {sys.argv[1]}")
validation = summary["validation"]
if validation["last_step"] != 1710:
    raise RuntimeError(f"Missing terminal validation: {sys.argv[1]}")
print(f"completed={sys.argv[1]} best_step={validation['best_step']} "
      f"best_validation_loss={validation['best_loss']}", flush=True)
PY
done

# Check existing/new reports before treating them as completed; do not overwrite.
validate_report() {
  "${PYTHON}" - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import math
import sys
from pathlib import Path

report, run_dir, dataset, tag, manifest = sys.argv[1:]
root = Path(run_dir)
r = json.loads(Path(report).read_text())
summary = json.loads((root / "training_summary.json").read_text())
expected_step = 0 if tag == "step_0000000" else summary["validation"]["best_step"]
expected_counts = (4055, 4055) if dataset == "skyscript" else (1094, 5470)
expected_task = (
    "paired_image_text_global_retrieval" if dataset == "skyscript"
    else "rsicd_image_text_retrieval"
)
if r["split"] != "val" or r["task"] != expected_task:
    raise ValueError(f"Unexpected retrieval task/split: {report}")
if (r["counts"]["images"], r["counts"]["captions"]) != expected_counts:
    raise ValueError(f"Unexpected retrieval candidate counts: {report}")
if Path(r["manifest"]["path"]).resolve() != Path(manifest).resolve():
    raise ValueError(f"Unexpected retrieval manifest: {report}")
checkpoint = r["model"]["checkpoint"]
if checkpoint["step"] != expected_step:
    raise ValueError(f"Unexpected checkpoint step: {report}")
if Path(checkpoint["path"]).resolve() != (root / f"{tag}.pt").resolve():
    raise ValueError(f"Unexpected checkpoint path: {report}")
if Path(r["model"]["model_config"]).resolve() != (root / "config.toml").resolve():
    raise ValueError(f"Unexpected model configuration: {report}")
identity = checkpoint["identity_check"]
if identity["status"] not in ("match", "warning") or identity["blocking_mismatches"]:
    raise ValueError(f"Checkpoint identity mismatch: {report}")
if r["tie_policy"] != "score_descending_then_candidate_index_ascending":
    raise ValueError(f"Unexpected retrieval tie policy: {report}")
metrics = r["metrics"]
recalls = [metrics[d][k] for d in ("image_to_text", "text_to_image")
           for k in ("r1", "r5", "r10")]
if any(not math.isfinite(x) or not 0 <= x <= 1 for x in recalls):
    raise ValueError(f"Invalid Recall: {report}")
if not math.isclose(metrics["mean_recall"], sum(recalls) / 6, abs_tol=1e-12):
    raise ValueError(f"Invalid mean Recall: {report}")
print(f"verified={report} mean_recall={metrics['mean_recall']}", flush=True)
PY
}

if [[ "${EVALUATE}" == 1 ]]; then
  for run_dir in "${RUN_DIRS[@]}"; do
    for dataset in skyscript rsicd; do
      if [[ "${dataset}" == skyscript ]]; then
        manifest="assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
      else
        manifest="assets/data/manifests/rsicd_val_retrieval_v1.jsonl"
      fi
      for tag in step_0000000 best; do
        report="${run_dir}/${dataset}_val_${tag}.json"
        if [[ -f "${report}" ]]; then
          validate_report "${report}" "${run_dir}" "${dataset}" "${tag}" "${manifest}"
          echo "Already evaluated: ${report}"
          continue
        fi
        "${PYTHON}" -u -m "dinotxt_rs.cli.evaluate_${dataset}" \
          --config "${run_dir}/config.toml" --manifest "${manifest}" \
          --checkpoint "${run_dir}/${tag}.pt" --training-output "${run_dir}" \
          --split val --output "${report}" 2>&1 | tee -a "${run_dir}/retrieval.log"
        validate_report "${report}" "${run_dir}" "${dataset}" "${tag}" "${manifest}"
      done
    done
  done
fi

archive="outputs/sat_replications_3epoch_reports.tar.gz"
tar --exclude='*.pt' --exclude='*.part' -czf "${archive}.part" "${RUN_DIRS[@]}"
mv -f "${archive}.part" "${archive}"
echo "Reports packaged: ${archive}"
if [[ "${EVALUATE}" == 1 ]]; then
  echo "All six runs and 24 retrieval reports completed. Keep existing seed11 reports for comparison."
else
  echo "Training completed. Rerun without --train-only to add retrieval and refresh the archive."
fi
