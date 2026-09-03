#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_REL="configs/verify_web_100step.toml"
OUTPUT_REL="outputs/m3_web_global77_100step_seed11"
MANIFEST_REL="assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77.jsonl"
EXPECTED_MANIFEST_SHA256="78abc613fbc8d98ea4617770473b30662d9eda31c0deb0dd06b51b1965d9fc0b"
EXPECTED_DINOV3_COMMIT="6876159a11b4df116f30f667f8c9888617df0751"

cd "${PROJECT_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${PROJECT_ROOT}/.venv. Run scripts/bootstrap_autodl.sh first." >&2
  exit 1
fi
if [[ ! -f "${CONFIG_REL}" || ! -f "${MANIFEST_REL}" ]]; then
  echo "Missing the 100-step config or its canonical training manifest." >&2
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

observed_manifest_sha256="$(sha256sum "${MANIFEST_REL}" | awk '{print $1}')"
if [[ "${observed_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "Unexpected training manifest SHA-256: ${observed_manifest_sha256}" >&2
  exit 1
fi

source .venv/bin/activate
ruff check .
pytest
python -m compileall -q src tools

mkdir -p "${OUTPUT_REL}"
{
  echo "project_commit=$(git rev-parse HEAD)"
  echo "manifest_sha256=${observed_manifest_sha256}"
  echo "config=${CONFIG_REL}"
} > "${OUTPUT_REL}/preflight.txt"

dinotxt-rs-smoke --config "${CONFIG_REL}" 2>&1 | tee "${OUTPUT_REL}/smoke.log"
dinotxt-rs-train --config "${CONFIG_REL}" 2>&1 | tee "${OUTPUT_REL}/train.log"

python tools/verify_training_run.py \
  --output "${OUTPUT_REL}" \
  --expected-steps 100 \
  --expected-train-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --expected-dinov3-commit "${EXPECTED_DINOV3_COMMIT}" \
  --expected-final-queue-size 4096 \
  --required-checkpoint-step 50 \
  --required-checkpoint-step 100 \
  --require-in-batch-loss | tee "${OUTPUT_REL}/verification_report.json"
