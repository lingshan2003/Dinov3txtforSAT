#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
DINOV3_GIT_URL="${DINOV3_GIT_URL:-https://github.com/facebookresearch/dinov3.git}"
DINOV3_REF="${DINOV3_REF:-6876159a11b4df116f30f667f8c9888617df0751}"
DINOV3_DIR="${PROJECT_ROOT}/external/dinov3"

command -v "${PYTHON_BIN}" >/dev/null || {
  echo "Missing ${PYTHON_BIN}. AutoDL image must provide Python 3.12." >&2
  exit 1
}

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel --index-url "${PIP_INDEX_URL}"
"${VENV_DIR}/bin/python" -m pip install \
  -r "${PROJECT_ROOT}/requirements/torch-cu128.txt" \
  --index-url "${PYTORCH_INDEX_URL}"
"${VENV_DIR}/bin/python" -m pip install -e "${PROJECT_ROOT}[dev]" --index-url "${PIP_INDEX_URL}"

if [[ -e "${DINOV3_DIR}" && ! -d "${DINOV3_DIR}/.git" ]]; then
  echo "${DINOV3_DIR} exists but is not a git checkout; refusing to overwrite it." >&2
  exit 1
fi
if [[ ! -d "${DINOV3_DIR}/.git" ]]; then
  git clone "${DINOV3_GIT_URL}" "${DINOV3_DIR}"
fi
if ! git -C "${DINOV3_DIR}" cat-file -e "${DINOV3_REF}^{commit}" 2>/dev/null; then
  git -C "${DINOV3_DIR}" fetch origin "${DINOV3_REF}"
fi
git -C "${DINOV3_DIR}" checkout --detach "${DINOV3_REF}"

"${VENV_DIR}/bin/python" -m pip install -e "${DINOV3_DIR}" --no-deps --index-url "${PIP_INDEX_URL}"
mkdir -p "${PROJECT_ROOT}/assets/checkpoints" "${PROJECT_ROOT}/assets/data" "${PROJECT_ROOT}/outputs"

"${VENV_DIR}/bin/python" - <<'PY'
import platform
import torch
import torchvision
print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
PY

echo "Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
