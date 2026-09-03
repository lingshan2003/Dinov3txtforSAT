#!/usr/bin/env python3
"""Validate the durable artifacts from a bounded training run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate metrics, provenance, and checkpoints")
    parser.add_argument("--output", required=True, type=Path, help="Experiment output directory")
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--expected-train-manifest-sha256", required=True)
    parser.add_argument("--expected-dinov3-commit")
    parser.add_argument("--expected-final-queue-size", type=int)
    parser.add_argument("--required-checkpoint-step", action="append", type=int, default=[])
    parser.add_argument("--require-in-batch-loss", action="store_true")
    parser.add_argument("--require-fixed-monitor", action="store_true")
    parser.add_argument("--expected-fixed-monitor-manifest-sha256")
    parser.add_argument("--fixed-monitor-every", type=int)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank metric record")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _series_summary(
    values: list[float], *, preferred_window_size: int = 10
) -> dict[str, float | int]:
    window = min(preferred_window_size, len(values))
    return {
        "window_size": window,
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean_first_window": statistics.fmean(values[:window]),
        "mean_last_window": statistics.fmean(values[-window:]),
    }


def verify_training_run(
    *,
    output_dir: Path,
    expected_steps: int,
    expected_train_manifest_sha256: str,
    expected_dinov3_commit: str | None = None,
    expected_final_queue_size: int | None = None,
    required_checkpoint_steps: tuple[int, ...] = (),
    require_in_batch_loss: bool = False,
    require_fixed_monitor: bool = False,
    expected_fixed_monitor_manifest_sha256: str | None = None,
    fixed_monitor_every: int | None = None,
) -> dict[str, Any]:
    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    if require_fixed_monitor and (
        expected_fixed_monitor_manifest_sha256 is None or fixed_monitor_every is None
    ):
        raise ValueError(
            "A required fixed monitor needs its expected manifest SHA-256 and interval"
        )
    if fixed_monitor_every is not None and fixed_monitor_every <= 0:
        raise ValueError("fixed_monitor_every must be positive")
    output_dir = output_dir.resolve()
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "training_summary.json"
    provenance_path = output_dir / "provenance.json"
    config_path = output_dir / "config.toml"
    required_paths = (metrics_path, summary_path, provenance_path, config_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required output artifact(s):\n" + "\n".join(missing))
    partials = sorted(output_dir.glob("*.part"))
    if partials:
        raise RuntimeError(f"Incomplete atomic output(s): {partials}")

    metrics = _read_metrics(metrics_path)
    if len(metrics) != expected_steps:
        raise ValueError(f"Expected {expected_steps} metric records, found {len(metrics)}")
    expected_step_numbers = list(range(1, expected_steps + 1))
    step_numbers = [record.get("step") for record in metrics]
    if step_numbers != expected_step_numbers:
        raise ValueError(f"Unexpected metric steps: {step_numbers[:5]} ... {step_numbers[-5:]}")

    losses: list[float] = []
    in_batch_losses: list[float] = []
    gradient_norms: list[float] = []
    peak_bytes: list[int] = []
    for step, record in enumerate(metrics, start=1):
        losses.append(_finite(record.get("loss"), f"metrics step {step} loss"))
        gradient_norms.append(
            _finite(record.get("gradient_norm"), f"metrics step {step} gradient_norm")
        )
        _finite(record.get("logit_scale"), f"metrics step {step} logit_scale")
        peak = record.get("peak_cuda_allocated_bytes")
        if not isinstance(peak, int) or peak <= 0:
            raise ValueError(
                f"metrics step {step} peak_cuda_allocated_bytes must be a positive int"
            )
        peak_bytes.append(peak)
        if require_in_batch_loss:
            in_batch_losses.append(
                _finite(record.get("in_batch_loss"), f"metrics step {step} in_batch_loss")
            )

    summary = _read_json(summary_path)
    if summary.get("steps") != expected_steps:
        raise ValueError(f"Summary steps must be {expected_steps}, got {summary.get('steps')!r}")
    for field in ("all_losses_finite", "all_gradients_finite"):
        if summary.get(field) is not True:
            raise ValueError(f"Summary field {field} must be true")
    for field in ("initial_loss", "final_loss", "last_gradient_norm"):
        _finite(summary.get(field), f"summary {field}")
    if require_in_batch_loss:
        for field in ("initial_in_batch_loss", "final_in_batch_loss"):
            _finite(summary.get(field), f"summary {field}")
    fixed_monitor_losses: list[float] = []
    if require_fixed_monitor:
        fixed_monitor_path = output_dir / "fixed_monitor.jsonl"
        if not fixed_monitor_path.is_file():
            raise FileNotFoundError(f"Missing fixed monitor metrics: {fixed_monitor_path}")
        fixed_monitor_records = _read_metrics(fixed_monitor_path)
        expected_monitor_steps = list(range(0, expected_steps + 1, fixed_monitor_every))
        monitor_steps = [record.get("step") for record in fixed_monitor_records]
        if monitor_steps != expected_monitor_steps:
            raise ValueError(
                "Unexpected fixed monitor steps: "
                f"expected {expected_monitor_steps}, got {monitor_steps}"
            )
        for step, record in zip(monitor_steps, fixed_monitor_records, strict=True):
            fixed_monitor_losses.append(_finite(record.get("loss"), f"monitor step {step} loss"))
            _finite(record.get("logit_scale"), f"monitor step {step} logit_scale")
        fixed_monitor_summary = summary.get("fixed_monitor")
        if not isinstance(fixed_monitor_summary, dict):
            raise ValueError("Summary has no fixed_monitor section")
        if fixed_monitor_summary.get("every") != fixed_monitor_every:
            raise ValueError(
                "Unexpected fixed monitor interval: "
                f"expected {fixed_monitor_every}, got {fixed_monitor_summary.get('every')!r}"
            )
        for field in ("initial_loss", "final_loss"):
            _finite(fixed_monitor_summary.get(field), f"summary fixed_monitor {field}")
    checkpoint = Path(str(summary.get("final_checkpoint", "")))
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Final checkpoint is missing or empty: {checkpoint}")
    required_checkpoints: list[str] = []
    for step in required_checkpoint_steps:
        if step <= 0:
            raise ValueError(f"Required checkpoint step must be positive, got {step}")
        checkpoint_path = output_dir / f"step_{step:07d}.pt"
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise FileNotFoundError(f"Required checkpoint is missing or empty: {checkpoint_path}")
        required_checkpoints.append(str(checkpoint_path))

    provenance = _read_json(provenance_path)
    observed_sha = provenance.get("files", {}).get("train_manifest", {}).get("sha256")
    if observed_sha != expected_train_manifest_sha256:
        raise ValueError(
            "Unexpected training manifest SHA-256: "
            f"expected {expected_train_manifest_sha256}, got {observed_sha!r}"
        )
    if require_fixed_monitor:
        observed_fixed_monitor_sha = (
            provenance.get("files", {}).get("fixed_monitor_manifest", {}).get("sha256")
        )
        if observed_fixed_monitor_sha != expected_fixed_monitor_manifest_sha256:
            raise ValueError(
                "Unexpected fixed monitor manifest SHA-256: "
                f"expected {expected_fixed_monitor_manifest_sha256}, "
                f"got {observed_fixed_monitor_sha!r}"
            )
    if expected_dinov3_commit is not None:
        observed_dinov3_commit = provenance.get("dinov3_commit")
        if observed_dinov3_commit != expected_dinov3_commit:
            raise ValueError(
                "Unexpected DINOv3 commit: "
                f"expected {expected_dinov3_commit}, got {observed_dinov3_commit!r}"
            )
    if expected_final_queue_size is not None:
        if metrics[-1].get("queue_size") != expected_final_queue_size:
            raise ValueError(
                "Unexpected final queue size: "
                f"expected {expected_final_queue_size}, got {metrics[-1].get('queue_size')!r}"
            )
        if summary.get("queue_size") != expected_final_queue_size:
            raise ValueError(
                "Unexpected configured queue size: "
                f"expected {expected_final_queue_size}, got {summary.get('queue_size')!r}"
            )

    report: dict[str, Any] = {
        "output_dir": str(output_dir),
        "steps": expected_steps,
        "loss": _series_summary(losses),
        "gradient_norm": _series_summary(gradient_norms),
        "peak_cuda_allocated_bytes": max(peak_bytes),
        "final_queue_size": metrics[-1].get("queue_size"),
        "final_checkpoint": str(checkpoint),
        "required_checkpoints": required_checkpoints,
        "project_commit": provenance.get("project_commit"),
        "dinov3_commit": provenance.get("dinov3_commit"),
        "train_manifest_sha256": observed_sha,
    }
    if require_in_batch_loss:
        report["in_batch_loss"] = _series_summary(in_batch_losses)
    if require_fixed_monitor:
        report["fixed_monitor_loss"] = _series_summary(
            fixed_monitor_losses, preferred_window_size=3
        )
    return report


def main() -> None:
    args = parse_args()
    report = verify_training_run(
        output_dir=args.output,
        expected_steps=args.expected_steps,
        expected_train_manifest_sha256=args.expected_train_manifest_sha256,
        expected_dinov3_commit=args.expected_dinov3_commit,
        expected_final_queue_size=args.expected_final_queue_size,
        required_checkpoint_steps=tuple(args.required_checkpoint_step),
        require_in_batch_loss=args.require_in_batch_loss,
        require_fixed_monitor=args.require_fixed_monitor,
        expected_fixed_monitor_manifest_sha256=args.expected_fixed_monitor_manifest_sha256,
        fixed_monitor_every=args.fixed_monitor_every,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
