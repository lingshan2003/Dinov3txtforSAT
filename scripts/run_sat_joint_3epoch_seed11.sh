#!/usr/bin/env bash
# Run the four joint visual/text experiments sequentially in tmux.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run with bash; do not source this script." >&2
  return 2
fi
set -Eeuo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/run_sat_joint_3epoch_seed11.sh"
  echo "Train adapter/head crossed with text projection/whole-text LoRA."
  echo "Completed runs are skipped; interrupted runs resume from latest.pt."
  echo "Validation loss is recorded; full retrieval evaluation is run separately."
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Unknown arguments. Use --help." >&2
  exit 2
fi

trap 'echo "Joint training stopped at line $LINENO. Check the current train.log before rerunning." >&2' ERR
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing .venv/bin/python" >&2; exit 1; }
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

TRIALS=(
  skyscript_sat_adapter_textlora_3epoch_seed11
  skyscript_sat_visionhead_textlora_3epoch_seed11
  skyscript_sat_adapter_textproj_3epoch_seed11
  skyscript_sat_visionhead_textproj_3epoch_seed11
)

# Check every configuration and all required paths before spending GPU time.
"${PYTHON}" - "${TRIALS[@]}" <<'PY'
import sys
from dinotxt_rs.config import load_config, required_paths

for trial in sys.argv[1:]:
    config = load_config(f"configs/{trial}.toml")
    missing = [str(path) for path in required_paths(config) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    snapshot = config.experiment.output_dir / "config.toml"
    if snapshot.exists() and snapshot.read_bytes() != config.source.read_bytes():
        raise ValueError(f"Saved configuration differs: {snapshot}")
    print(f"preflight={trial} paths=present", flush=True)
PY

for trial in "${TRIALS[@]}"; do
  config="configs/${trial}.toml"
  run_dir="outputs/${trial}"
  # A summary is published only on clean exit; interrupted runs may lack it.
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
if not summary.get("completed") or summary.get("steps") != 1710:
    raise RuntimeError(f"Training did not complete: {sys.argv[1]}")
validation = summary["validation"]
print(f"completed={sys.argv[1]} best_step={validation['best_step']} "
      f"best_validation_loss={validation['best_loss']}", flush=True)
PY
done
echo "All four joint runs completed. Full retrieval evaluation is still pending."
