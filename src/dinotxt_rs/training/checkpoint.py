from __future__ import annotations

import json
import os
import pickle
import random
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dinotxt_rs.models.official_dinotxt import trainable_state_dict
from dinotxt_rs.training.provenance import compare_run_identities


def _temporary_file(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _stage_bytes(destination: Path, content: bytes) -> Path:
    temporary = _temporary_file(destination)
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    optimizer_parameter_groups: dict[str, Any],
    step: int,
    config_text: str,
    name: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (name or f"step_{step:07d}.pt")
    payload = {
        "format_version": 2,
        "step": step,
        "trainable_model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "optimizer_parameter_groups": optimizer_parameter_groups,
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
    temporary = _temporary_file(destination)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_best_checkpoint(output_dir: Path, source: Path) -> Path:
    """Atomically make best.pt reference an already-verified step checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "best.pt"
    temporary = _temporary_file(destination)
    temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError:
            # Never open a preexisting .part file: it may be a hard link to an old checkpoint.
            with source.open("rb") as read_handle, temporary.open("xb") as write_handle:
                while chunk := read_handle.read(8 * 1024 * 1024):
                    write_handle.write(chunk)
                write_handle.flush()
                os.fsync(write_handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _matching_best_checkpoint(path: Path, step: int, identity: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        # mmap avoids materializing large optimizer tensors just to inspect metadata.
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
        return False
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != 2
        or type(payload.get("step")) is not int
        or payload["step"] != step
        or not isinstance(payload.get("run_identity"), dict)
    ):
        return False
    blocking, _ = compare_run_identities(identity, payload["run_identity"])
    return not blocking


def prepare_resume_artifacts(
    output_dir: Path,
    step: int,
    run_state: dict[str, Any],
    resume_path: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate best state and archive logs beyond the restored optimizer step.

    All input validation and best-source selection precede any artifact mutation. The
    discarded records are published before log truncation, so an interrupted rollback
    always leaves a recoverable copy. Individual replacements are atomic; this is not
    a filesystem-wide transaction.
    """
    if type(step) is not int or step < 0:
        raise ValueError("Resume artifact step must be a nonnegative integer")
    logs: dict[str, dict[str, Any]] = {}
    for filename in ("metrics.jsonl", "validation.jsonl", "fixed_monitor.jsonl"):
        path = output_dir / filename
        if not path.is_file():
            continue
        retained: list[str] = []
        discarded: list[dict[str, Any]] = []
        previous_step = -1
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid resume log JSON: {filename}:{line_number}") from error
            log_step = record.get("step") if isinstance(record, dict) else None
            if type(log_step) is not int or log_step < 0 or log_step <= previous_step:
                raise ValueError(
                    f"Resume log steps must be nonnegative, strictly increasing integers: "
                    f"{filename}:{line_number}"
                )
            previous_step = log_step
            if log_step <= step:
                retained.append(line)
            else:
                discarded.append(record)
        logs[filename] = {"retained": retained, "discarded": discarded}

    best_step = run_state.get("best_validation_step")
    best_source: Path | None = None
    best_path = output_dir / "best.pt"
    if best_step is None and best_path.exists():
        raise ValueError("Restored run state has no best_validation_step but best.pt exists")
    if best_step is not None:
        if type(best_step) is not int or not 0 <= best_step <= step:
            raise ValueError("Restored best_validation_step is invalid")
        if not _matching_best_checkpoint(best_path, best_step, expected_identity):
            for candidate in (resume_path, output_dir / f"step_{best_step:07d}.pt"):
                if _matching_best_checkpoint(candidate, best_step, expected_identity):
                    best_source = candidate
                    break
            if best_source is None:
                raise ValueError(
                    f"Cannot recover best.pt at restored best step {best_step} "
                    "with matching identity"
                )

    discarded_logs = [
        {"filename": filename, "records": contents["discarded"]}
        for filename, contents in logs.items()
        if contents["discarded"]
    ]
    archive_path = output_dir / "resume_discarded.jsonl"
    staged: list[tuple[Path, Path]] = []
    try:
        if discarded_logs:
            archive_content = archive_path.read_bytes() if archive_path.is_file() else b""
            for line_number, line in enumerate(archive_content.splitlines(), 1):
                try:
                    archived_record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid resume archive JSON at line {line_number}"
                    ) from error
                if not isinstance(archived_record, dict):
                    raise ValueError(f"Invalid resume archive record at line {line_number}")
            archive_record = {
                "format_version": 1,
                "resume_step": step,
                "resume_checkpoint": str(resume_path.resolve()),
                "logs": discarded_logs,
            }
            if archive_content and not archive_content.endswith(b"\n"):
                archive_content += b"\n"
            archive_content += (json.dumps(archive_record, ensure_ascii=False) + "\n").encode()
            staged.append((_stage_bytes(archive_path, archive_content), archive_path))
            for filename, contents in logs.items():
                if contents["discarded"]:
                    content = "".join(line + "\n" for line in contents["retained"]).encode()
                    path = output_dir / filename
                    staged.append((_stage_bytes(path, content), path))
        # A failed best recovery must not truncate the live logs.
        if best_source is not None:
            save_best_checkpoint(output_dir, best_source)
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return {
        "restored_step": step,
        "best_validation_step": best_step,
        "best_recovered_from": None if best_source is None else str(best_source.resolve()),
        "discarded_records": {
            filename: len(contents["discarded"]) for filename, contents in logs.items()
        },
        "discarded_archive": str(archive_path) if discarded_logs else None,
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


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
    expected_optimizer_parameter_groups: dict[str, Any],
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise ValueError("Resume requires a format_version=2 checkpoint")
    observed_identity = payload.get("run_identity")
    if not isinstance(observed_identity, dict):
        raise ValueError("Resume checkpoint has no run identity")
    blocking, advisory = compare_run_identities(expected_identity, observed_identity)
    if blocking:
        raise ValueError(
            "Refusing to resume because checkpoint identity differs: " + ", ".join(blocking)
        )
    identity_check = {
        "status": "warning" if advisory else "match",
        "blocking_mismatches": blocking,
        "advisory_mismatches": advisory,
        "checkpoint_project_commit": observed_identity.get("project_commit"),
        "current_project_commit": expected_identity.get("project_commit"),
    }
    if advisory:
        warnings.warn(
            "Project commit changed across checkpoint resume: "
            f"checkpoint={observed_identity.get('project_commit')!r}, "
            f"current={expected_identity.get('project_commit')!r}. Continuing because the "
            "exact config, input files, upstream DINOv3 commit, and checkpoint structure match; "
            "the transition is recorded in resume_history.jsonl.",
            RuntimeWarning,
            stacklevel=2,
        )
    if payload.get("config_toml") is None:
        raise ValueError("Resume checkpoint is missing its config snapshot")
    if payload.get("optimizer_parameter_groups") != expected_optimizer_parameter_groups:
        raise ValueError("Checkpoint optimizer parameter groups do not match the configured model")
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
    return step, run_state, identity_check
