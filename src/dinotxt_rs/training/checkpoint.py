from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dinotxt_rs.models.official_dinotxt import trainable_state_dict


def save_checkpoint(
    output_dir: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    queue: Any,
    sampler_state: dict[str, Any],
    loader_generator_state: torch.Tensor,
    run_state: dict[str, Any],
    run_identity: dict[str, Any],
    step: int,
    config_text: str,
    name: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (name or f"step_{step:07d}.pt")
    temporary = destination.with_suffix(".pt.part")
    payload = {
        "format_version": 2,
        "step": step,
        "trainable_model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "queue": queue.state_dict(),
        "sampler": sampler_state,
        "loader_generator_state": loader_generator_state,
        "rng": capture_rng_state(),
        "run_state": run_state,
        "run_identity": run_identity,
        "config_toml": config_text,
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def save_best_checkpoint(output_dir: Path, source: Path) -> Path:
    """Atomically make best.pt reference an already-verified step checkpoint."""
    destination = output_dir / "best.pt"
    temporary = destination.with_suffix(".pt.part")
    try:
        os.link(source, temporary)
    except OSError:
        # A copied best checkpoint is still safe on filesystems without hard links.
        with source.open("rb") as read_handle, temporary.open("wb") as write_handle:
            while chunk := read_handle.read(8 * 1024 * 1024):
                write_handle.write(chunk)
    os.replace(temporary, destination)
    return destination


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _identity_mismatches(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    mismatches = [
        name
        for name in ("format_version", "config_sha256", "project_commit", "dinov3_commit")
        if observed.get(name) != expected.get(name)
    ]
    if observed.get("files") != expected.get("files"):
        mismatches.append("files")
    return mismatches


def _restore_trainable_model(model: Any, state: Any) -> None:
    if not isinstance(state, dict):
        raise ValueError("Checkpoint trainable_model is invalid")
    expected_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected_names:
        raise ValueError("Checkpoint trainable parameter names do not match the configured model")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError("Checkpoint has unexpected trainable model keys")


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        raise ValueError("Checkpoint RNG state is invalid")
    torch_state = state.get("torch")
    numpy_state = state.get("numpy")
    if not isinstance(torch_state, torch.Tensor) or numpy_state is None or "python" not in state:
        raise ValueError("Checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if torch.cuda.is_available():
        if not isinstance(cuda_state, list):
            raise ValueError("Checkpoint CUDA RNG state is invalid")
        torch.cuda.set_rng_state_all(cuda_state)


def load_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    queue: Any,
    sampler: Any,
    loader_generator: torch.Generator,
    device: torch.device,
    expected_identity: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise ValueError("Resume requires a format_version=2 checkpoint")
    observed_identity = payload.get("run_identity")
    if not isinstance(observed_identity, dict):
        raise ValueError("Resume checkpoint has no run identity")
    mismatches = _identity_mismatches(expected_identity, observed_identity)
    if mismatches:
        raise ValueError(
            "Refusing to resume because checkpoint identity differs: " + ", ".join(mismatches)
        )
    if payload.get("config_toml") is None:
        raise ValueError("Resume checkpoint is missing its config snapshot")
    step = payload.get("step")
    if not isinstance(step, int) or step < 0:
        raise ValueError("Resume checkpoint step is invalid")
    _restore_trainable_model(model, payload.get("trainable_model"))
    optimizer.load_state_dict(payload.get("optimizer"))
    scheduler.load_state_dict(payload.get("scheduler"))
    scaler.load_state_dict(payload.get("scaler"))
    queue.load_state_dict(payload.get("queue"), device)
    sampler.load_state_dict(payload.get("sampler"))
    loader_generator_state = payload.get("loader_generator_state")
    if not isinstance(loader_generator_state, torch.Tensor):
        raise ValueError("Checkpoint DataLoader generator state is invalid")
    loader_generator.set_state(loader_generator_state)
    _restore_rng_state(payload.get("rng"))
    run_state = payload.get("run_state")
    if not isinstance(run_state, dict):
        raise ValueError("Checkpoint run state is invalid")
    return step, run_state
