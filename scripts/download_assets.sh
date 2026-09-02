#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/assets/checkpoints}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/assets/data/raw/chatearthnet}"
ZENODO_RECORD_BASE="${ZENODO_RECORD_BASE:-https://zenodo.org/records/11003436/files}"

DINOV3_WEB_URL="${DINOV3_WEB_URL:-https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
DINOV3_SAT_URL="${DINOV3_SAT_URL:-https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth}"
DINOTXT_HEAD_URL="${DINOTXT_HEAD_URL:-https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth}"
BPE_URL="${BPE_URL:-https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz}"

mkdir -p "${CHECKPOINT_DIR}" "${DATA_DIR}"

hash_file() {
  local algorithm="$1"
  local path="$2"
  if command -v "${algorithm}sum" >/dev/null; then
    "${algorithm}sum" "${path}" | awk '{print $1}'
  elif [[ "${algorithm}" == "sha256" ]] && command -v shasum >/dev/null; then
    shasum -a 256 "${path}" | awk '{print $1}'
  elif [[ "${algorithm}" == "md5" ]] && command -v md5 >/dev/null; then
    md5 -q "${path}"
  else
    echo "No ${algorithm} tool found" >&2
    return 1
  fi
}

verify_hash() {
  local path="$1"
  local algorithm="$2"
  local expected="$3"
  local actual
  actual="$(hash_file "${algorithm}" "${path}")"
  [[ "${actual:0:${#expected}}" == "${expected}" ]] || {
    echo "Hash mismatch for ${path}: expected ${expected}, got ${actual}" >&2
    return 1
  }
}

download() {
  local url="$1"
  local destination="$2"
  local algorithm="${3:-}"
  local expected="${4:-}"
  if [[ -f "${destination}" ]]; then
    if [[ -z "${algorithm}" ]] || verify_hash "${destination}" "${algorithm}" "${expected}"; then
      echo "Already verified: ${destination}"
      return
    fi
    echo "Existing file failed verification; move it aside before retrying: ${destination}" >&2
    return 1
  fi
  local partial="${destination}.part"
  echo "Downloading ${url}"
  if command -v aria2c >/dev/null; then
    aria2c --continue=true --max-connection-per-server=8 --split=8 \
      --dir "$(dirname "${partial}")" --out "$(basename "${partial}")" "${url}"
  elif command -v wget >/dev/null; then
    wget --continue --tries=20 --timeout=30 --output-document "${partial}" "${url}"
  else
    curl --location --fail --retry 20 --retry-all-errors --continue-at - --output "${partial}" "${url}"
  fi
  if [[ -n "${algorithm}" ]]; then
    verify_hash "${partial}" "${algorithm}" "${expected}"
  fi
  mv "${partial}" "${destination}"
}

download_weights() {
  download "${DINOV3_WEB_URL}" "${CHECKPOINT_DIR}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" sha256 8aa4cbdd
  download "${DINOV3_SAT_URL}" "${CHECKPOINT_DIR}/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" sha256 eadcf0ff
  download "${DINOTXT_HEAD_URL}" "${CHECKPOINT_DIR}/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth" sha256 a442d8f5
  download "${BPE_URL}" "${CHECKPOINT_DIR}/bpe_simple_vocab_16e6.txt.gz"
}

download_metadata() {
  download "${ZENODO_RECORD_BASE}/json_files.zip?download=1" "${DATA_DIR}/json_files.zip" md5 ac365a328d645641ca28315bee9e1d25
}

download_rgb() {
  download "${ZENODO_RECORD_BASE}/s2_rgb_images.zip?download=1" "${DATA_DIR}/s2_rgb_images.zip" md5 5425015e2537271044341403e8d90965
}

case "${1:-all}" in
  weights) download_weights ;;
  data-metadata) download_metadata ;;
  data-rgb) download_rgb ;;
  all) download_weights; download_metadata; download_rgb ;;
  *) echo "Usage: $0 [weights|data-metadata|data-rgb|all]" >&2; exit 2 ;;
esac

