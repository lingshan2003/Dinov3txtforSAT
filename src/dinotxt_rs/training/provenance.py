from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_provenance(config: Config) -> dict[str, Any]:
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
    return {
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


def run_identity(config_text: str, provenance: dict[str, Any]) -> dict[str, Any]:
    """Fields compared before checkpoint resume or evaluation.

    Paths are deliberately excluded: moving a fully verified project tree must
    not invalidate a run, whereas any input content or source revision change
    must. ``project_commit`` is retained for audit but handled as advisory by
    :func:`compare_run_identities`; the other fields are blocking.
    """
    files = provenance.get("files")
    if not isinstance(files, dict):
        raise ValueError("Provenance files are invalid")
    file_hashes = {
        name: record["sha256"]
        for name, record in files.items()
        if isinstance(record, dict) and isinstance(record.get("sha256"), str)
    }
    if len(file_hashes) != len(files):
        raise ValueError("Provenance file hashes are invalid")
    return {
        "format_version": 1,
        "config_sha256": sha256_text(config_text),
        "project_commit": provenance.get("project_commit"),
        "dinov3_commit": provenance.get("dinov3_commit"),
        "files": file_hashes,
    }


def compare_run_identities(
    expected: dict[str, Any], observed: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Split identity differences into blocking and provenance-only fields.

    The project commit remains recorded, but changing it does not by itself
    prove that an experiment's effective protocol changed. Exact config and
    input identities, plus the upstream DINOv3 commit, remain blocking.
    """
    blocking = [
        name
        for name in ("format_version", "config_sha256", "dinov3_commit")
        if observed.get(name) != expected.get(name)
    ]
    if observed.get("files") != expected.get("files"):
        blocking.append("files")
    advisory = (
        ["project_commit"]
        if observed.get("project_commit") != expected.get("project_commit")
        else []
    )
    return blocking, advisory


def write_provenance(config: Config, payload: dict[str, Any] | None = None) -> Path:
    if payload is None:
        payload = build_provenance(config)
    config.experiment.output_dir.mkdir(parents=True, exist_ok=True)
    destination = config.experiment.output_dir / "provenance.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination
