from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

from dinotxt_rs.config import Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_provenance(config: Config) -> Path:
    files = {
        "config": config.source,
        "backbone_weights": config.model.backbone_weights,
        "dinotxt_weights": config.model.dinotxt_weights,
        "bpe_vocab": config.model.bpe_vocab,
        "train_manifest": config.data.train_manifest,
    }
    if config.data.val_manifest is not None:
        files["val_manifest"] = config.data.val_manifest
    if config.data.fixed_monitor_manifest is not None:
        files["fixed_monitor_manifest"] = config.data.fixed_monitor_manifest
    payload = {
        "project_commit": git_commit(config.source.parent),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dinov3_commit": git_commit(config.model.dinov3_repo),
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
    }
    config.experiment.output_dir.mkdir(parents=True, exist_ok=True)
    destination = config.experiment.output_dir / "provenance.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination
