#!/usr/bin/env python3
"""Inspect a training output after artifacts have been retained or pruned.

This tool intentionally does not validate a run as successful.  It makes the
remaining evidence explicit, including whether a retained ``best.pt`` is a
byte-identical hard-link/copy candidate for a missing step checkpoint.  It
never writes or replaces a checkpoint; the only optional write is its small
JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect retained training artifacts without changing checkpoints"
    )
    parser.add_argument("--output", required=True, type=Path, help="Experiment output directory")
    parser.add_argument(
        "--expected-checkpoint-step",
        action="append",
        type=int,
        default=[],
        help="A checkpoint step that the experiment protocol required (repeatable)",
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Path for the small JSON inspection report",
    )
    return parser.parse_args()


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {exc}"
    if not isinstance(value, dict):
        return None, "expected a JSON object"
    return value, None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                return None, f"line {line_number} is blank"
            value = json.loads(line)
            if not isinstance(value, dict):
                return None, f"line {line_number} is not a JSON object"
            records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {exc}"
    return records, None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _checkpoint_metadata(path: Path, loader: Callable[..., Any]) -> dict[str, Any]:
    """Load only reportable checkpoint metadata (torch must still read the archive)."""
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }
    try:
        payload = loader(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - error wording depends on torch version.
        record["load_error"] = f"{type(exc).__name__}: {exc}"
        return record
    if not isinstance(payload, dict):
        record["load_error"] = "checkpoint payload is not a dictionary"
        return record

    record["format_version"] = payload.get("format_version")
    step = payload.get("step")
    if isinstance(step, int) and step >= 0:
        record["checkpoint_step"] = step
    else:
        record["load_error"] = "checkpoint step is absent or invalid"
        return record

    run_state = payload.get("run_state")
    if isinstance(run_state, dict):
        record["run_state"] = {
            name: run_state.get(name)
            for name in (
                "micro_step",
                "validation_evaluations",
                "best_validation_loss",
                "best_validation_step",
            )
        }
    run_identity = payload.get("run_identity")
    if isinstance(run_identity, dict):
        record["run_identity"] = {
            name: run_identity.get(name)
            for name in ("format_version", "config_sha256", "project_commit", "dinov3_commit")
        }
    return record


def _checkpoint_paths(output_dir: Path) -> list[Path]:
    paths = sorted(output_dir.glob("step_*.pt"))
    best = output_dir / "best.pt"
    if best.is_file():
        paths.append(best)
    return paths


def _validation_summary(records: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if records is None:
        return None
    points: list[dict[str, float | int]] = []
    for record in records:
        step = record.get("step")
        loss = _finite_number(record.get("loss"))
        if isinstance(step, int) and step >= 0 and loss is not None:
            points.append({"step": step, "loss": loss})
    if not points:
        return {"points": [], "error": "no finite (step, loss) records"}
    best = min(points, key=lambda point: (float(point["loss"]), int(point["step"])))
    initial = next((point for point in points if point["step"] == 0), None)
    final = points[-1]
    summary: dict[str, Any] = {
        "points": points,
        "best_step_from_records": best["step"],
        "best_loss_from_records": best["loss"],
        "final_step_from_records": final["step"],
        "final_loss_from_records": final["loss"],
    }
    if initial is not None:
        delta = float(best["loss"]) - float(initial["loss"])
        summary["initial_step_zero_loss"] = initial["loss"]
        summary["best_minus_step_zero"] = delta
        summary["best_relative_change_from_step_zero"] = delta / float(initial["loss"])
    return summary


def _summary_final_checkpoint_name(summary: dict[str, Any] | None) -> str | None:
    if summary is None:
        return None
    value = summary.get("final_checkpoint")
    if not isinstance(value, str) or not value:
        return None
    return Path(value).name


def inspect_training_artifacts(
    *,
    output_dir: Path,
    expected_checkpoint_steps: tuple[int, ...] = (),
    checkpoint_loader: Callable[..., Any] = torch.load,
) -> dict[str, Any]:
    """Return a conservative report about what evidence remains in ``output_dir``."""
    output_dir = output_dir.resolve()
    if any(step < 0 for step in expected_checkpoint_steps):
        raise ValueError("Expected checkpoint steps must be nonnegative")

    summary, summary_error = _read_json(output_dir / "training_summary.json")
    validation_records, validation_error = _read_jsonl(output_dir / "validation.jsonl")
    resume_records, resume_error = _read_jsonl(output_dir / "resume_history.jsonl")

    loaded_by_inode: dict[tuple[int, int], dict[str, Any]] = {}
    checkpoints: list[dict[str, Any]] = []
    for path in _checkpoint_paths(output_dir):
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in loaded_by_inode:
            record = dict(loaded_by_inode[inode])
            record["path"] = str(path)
            record["hard_link_or_duplicate_of"] = loaded_by_inode[inode]["path"]
        else:
            record = _checkpoint_metadata(path, checkpoint_loader)
            loaded_by_inode[inode] = record
        checkpoints.append(record)

    available_by_step: dict[int, list[str]] = {}
    for checkpoint in checkpoints:
        step = checkpoint.get("checkpoint_step")
        if isinstance(step, int):
            available_by_step.setdefault(step, []).append(str(checkpoint["path"]))

    expected_steps = sorted(set(expected_checkpoint_steps))
    required_checkpoints = []
    recovery_options = []
    for step in expected_steps:
        expected_path = output_dir / f"step_{step:07d}.pt"
        exists = expected_path.is_file() and expected_path.stat().st_size > 0
        matching_sources = available_by_step.get(step, [])
        checkpoint_record: dict[str, Any] = {
            "step": step,
            "path": str(expected_path),
            "present": exists,
        }
        if not exists:
            checkpoint_record["matching_retained_checkpoint_payloads"] = matching_sources
            if matching_sources:
                recovery_options.append(
                    {
                        "missing_step": step,
                        "status": "recoverable_from_retained_same_step_payload",
                        "sources": matching_sources,
                        "note": (
                            "A retained checkpoint declares this exact step. Do not use a "
                            "checkpoint from a different step as a replacement."
                        ),
                    }
                )
            else:
                recovery_options.append(
                    {
                        "missing_step": step,
                        "status": "not_recoverable_from_remaining_checkpoints",
                        "note": "No retained checkpoint declares this step.",
                    }
                )
        required_checkpoints.append(checkpoint_record)

    validation = _validation_summary(validation_records)
    best_consistency: dict[str, Any] = {}
    if (
        summary is not None
        and isinstance(summary.get("validation"), dict)
        and validation is not None
    ):
        declared = summary["validation"]
        declared_step = declared.get("best_step")
        declared_loss = _finite_number(declared.get("best_loss"))
        best_consistency = {
            "summary_best_step": declared_step,
            "summary_best_loss": declared_loss,
            "best_step_matches_validation_records": declared_step
            == validation.get("best_step_from_records"),
            "best_loss_matches_validation_records": declared_loss
            == validation.get("best_loss_from_records"),
        }
        best_file = next(
            (
                checkpoint
                for checkpoint in checkpoints
                if Path(str(checkpoint["path"])).name == "best.pt"
            ),
            None,
        )
        if best_file is None:
            best_consistency["best_file_present"] = False
        else:
            best_consistency["best_file_present"] = True
            best_consistency["best_file_checkpoint_step"] = best_file.get("checkpoint_step")
            best_consistency["best_file_matches_summary_best_step"] = (
                best_file.get("checkpoint_step") == declared_step
            )

    final_name = _summary_final_checkpoint_name(summary)
    final_checkpoint = {
        "declared_in_summary": None if summary is None else summary.get("final_checkpoint"),
        "present": False if final_name is None else (output_dir / final_name).is_file(),
    }
    if final_name is not None:
        final_checkpoint["expected_path"] = str(output_dir / final_name)

    recorded_resume_steps: list[int] = []
    if resume_records is not None:
        recorded_resume_steps = sorted(
            {
                record["checkpoint_step"]
                for record in resume_records
                if isinstance(record.get("checkpoint_step"), int)
            }
        )
    resume_evidence = {
        "recorded_resume_steps": recorded_resume_steps,
        "replayable_from_retained_checkpoint_steps": sorted(
            step for step in recorded_resume_steps if step in available_by_step
        ),
        "not_replayable_from_retained_checkpoint_steps": sorted(
            step for step in recorded_resume_steps if step not in available_by_step
        ),
    }

    missing_required = [item["step"] for item in required_checkpoints if not item["present"]]
    errors = {
        "training_summary": summary_error,
        "validation": validation_error,
        "resume_history": resume_error,
    }
    report: dict[str, Any] = {
        "format_version": 1,
        "output_dir": str(output_dir),
        "inspection_is_read_only_for_checkpoints": True,
        "training_summary": summary,
        "training_summary_error": summary_error,
        "validation": validation,
        "validation_error": validation_error,
        "best_checkpoint_consistency": best_consistency,
        "final_checkpoint": final_checkpoint,
        "checkpoints_found": checkpoints,
        "required_checkpoints": required_checkpoints,
        "missing_required_checkpoint_steps": missing_required,
        "resume_evidence": resume_evidence,
        "recovery_options": recovery_options,
        "artifact_read_errors": {name: error for name, error in errors.items() if error},
    }
    report["artifact_evidence_status"] = (
        "degraded" if missing_required or report["artifact_read_errors"] else "complete"
    )
    return report


def main() -> None:
    args = parse_args()
    report = inspect_training_artifacts(
        output_dir=args.output,
        expected_checkpoint_steps=tuple(args.expected_checkpoint_step),
    )
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report={report_path}")
    print(f"artifact_evidence_status={report['artifact_evidence_status']}")
    print(f"missing_required_checkpoint_steps={report['missing_required_checkpoint_steps']}")
    print(f"recovery_options={len(report['recovery_options'])}")


if __name__ == "__main__":
    main()
