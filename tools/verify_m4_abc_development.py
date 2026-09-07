#!/usr/bin/env python3
"""Verify and summarize the M4 Web A/B/C 100-step development experiment."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path
from typing import Any

STEPS = (50, 100)
DESIGNS = {
    "a": {
        "text_last_k": 4,
        "train_vision_head": True,
        "train_text_projection": True,
        "train_logit_scale": True,
        "learning_rate": 5e-6,
    },
    "b": {
        "text_last_k": 0,
        "train_vision_head": True,
        "train_text_projection": False,
        "train_logit_scale": False,
        "learning_rate": 5e-5,
    },
    "c": {
        "text_last_k": 0,
        "train_vision_head": True,
        "train_text_projection": False,
        "train_logit_scale": False,
        "learning_rate": 5e-6,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validation_losses(run: Path) -> dict[int, float]:
    path = run / "validation.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    losses: dict[int, float] = {}
    for record in records:
        step = record.get("step")
        if not isinstance(step, int) or step in losses:
            raise ValueError(f"Invalid or duplicate validation step in {path}: {step!r}")
        losses[step] = _finite(record.get("loss"), f"{run.name} validation step {step}")
    if set(losses) != {0, *STEPS}:
        raise ValueError(f"Unexpected validation steps in {path}: {sorted(losses)}")
    return losses


def _verify_design(label: str, run: Path) -> None:
    config_path = run / "config.toml"
    verification_path = run / "verification_report.json"
    if not config_path.is_file() or not verification_path.is_file():
        raise FileNotFoundError(f"Run {label} lacks config or training verification: {run}")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    observed = {
        "text_last_k": config.get("model", {}).get("text_last_k"),
        "train_vision_head": config.get("model", {}).get("train_vision_head"),
        "train_text_projection": config.get("model", {}).get("train_text_projection"),
        "train_logit_scale": config.get("model", {}).get("train_logit_scale"),
        "learning_rate": config.get("train", {}).get("learning_rate"),
    }
    if observed != DESIGNS[label]:
        raise ValueError(f"Run {label} config does not match the preregistered design: {observed}")
    verification = _read_json(verification_path)
    if verification.get("steps") != 100 or verification.get("completed") is not False:
        raise ValueError(f"Run {label} is not a verified bounded 100-step experiment")


def _retrieval_report(path: Path, expected_step: int | None) -> dict[str, Any]:
    report = _read_json(path)
    if report.get("task") != "rsicd_image_text_retrieval" or report.get("split") != "val":
        raise ValueError(f"Expected an RSICD val retrieval report: {path}")
    manifest = report.get("manifest")
    metrics = report.get("metrics")
    model = report.get("model")
    if (
        not isinstance(manifest, dict)
        or not isinstance(metrics, dict)
        or not isinstance(model, dict)
    ):
        raise ValueError(f"Incomplete retrieval report: {path}")
    checkpoint = model.get("checkpoint")
    if expected_step is None:
        if checkpoint is not None:
            raise ValueError("Official development baseline unexpectedly used a checkpoint")
    elif not isinstance(checkpoint, dict) or checkpoint.get("step") != expected_step:
        raise ValueError(f"Retrieval report {path} did not use step {expected_step}")
    mean_recall = _finite(metrics.get("mean_recall"), f"{path} mean recall")
    if not 0 <= mean_recall <= 1:
        raise ValueError(f"Mean recall outside [0, 1]: {path}")
    counts = report.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(f"Retrieval report has no counts: {path}")
    return {
        "manifest_sha256": manifest.get("sha256"),
        "mean_recall": mean_recall,
        "counts": counts,
        "trainable_parameters": model.get("trainable_parameters"),
        "checkpoint_sha256": checkpoint.get("sha256") if isinstance(checkpoint, dict) else None,
    }


def verify_m4_abc_development(
    development_output: Path,
    runs: dict[str, Path],
    *,
    retention_margin: float = 0.01,
) -> dict[str, Any]:
    if set(runs) != set(DESIGNS):
        raise ValueError(f"Runs must contain exactly {sorted(DESIGNS)}")
    if not 0 <= retention_margin <= 1:
        raise ValueError("retention_margin must be in [0, 1]")
    overlap = _read_json(development_output / "train_vs_rsicd_val_overlap.json")
    if overlap.get("status") != "clear":
        raise ValueError("Training and RSICD val manifests did not pass the overlap audit")

    baseline = _retrieval_report(development_output / "rsicd_val_official.json", None)
    baseline_hash = baseline["manifest_sha256"]
    if not isinstance(baseline_hash, str):
        raise ValueError("Official development report has no manifest SHA-256")
    baseline_mean_recall = baseline["mean_recall"]
    threshold = max(0.0, baseline_mean_recall - retention_margin)

    summary: dict[str, Any] = {}
    eligible: list[str] = []
    head_only_counts: list[int] = []
    full_scope_count: int | None = None
    for label in sorted(DESIGNS):
        run = runs[label]
        _verify_design(label, run)
        validation = _validation_losses(run)
        retrieval: dict[int, dict[str, Any]] = {}
        for step in STEPS:
            report = _retrieval_report(
                development_output / f"rsicd_val_{label}_step{step}.json", step
            )
            if report["manifest_sha256"] != baseline_hash or report["counts"] != baseline["counts"]:
                raise ValueError(f"Run {label} step {step} used a different RSICD val input")
            retrieval[step] = report
        trainable = retrieval[100]["trainable_parameters"]
        if not isinstance(trainable, dict) or not isinstance(trainable.get("trainable"), int):
            raise ValueError(f"Run {label} report has no trainable parameter count")
        if label == "a":
            full_scope_count = trainable["trainable"]
        else:
            head_only_counts.append(trainable["trainable"])
        final_recall = retrieval[100]["mean_recall"]
        retention_pass = final_recall >= threshold
        same_domain_improved = validation[100] < validation[0]
        # Step 100 is still inside the 250-step warmup and is used as a safety screen.
        # Same-domain improvement is reported but becomes mandatory only at the longer gate.
        promotion_eligible = retention_pass
        if promotion_eligible:
            eligible.append(label)
        summary[label] = {
            "design": DESIGNS[label],
            "run": str(run.resolve()),
            "trainable_parameters": trainable["trainable"],
            "chatearthnet_validation": {
                "step0": validation[0],
                "step50": validation[50],
                "step100": validation[100],
                "step100_change_from_step0": validation[100] - validation[0],
                "improved_at_step100": same_domain_improved,
            },
            "rsicd_val_mean_recall": {
                "official": baseline_mean_recall,
                "step50": retrieval[50]["mean_recall"],
                "step100": final_recall,
                "step100_change_from_official": final_recall - baseline_mean_recall,
                "retention_threshold": threshold,
                "retention_pass": retention_pass,
            },
            "promotion_eligible": promotion_eligible,
        }
    if full_scope_count is None or len(set(head_only_counts)) != 1:
        raise ValueError("A/B/C trainable parameter counts are inconsistent")
    if full_scope_count <= head_only_counts[0]:
        raise ValueError("Full-scope run must train more parameters than head-only runs")
    eligible.sort(
        key=lambda label: summary[label]["rsicd_val_mean_recall"]["step100"], reverse=True
    )
    return {
        "format_version": 1,
        "status": "complete",
        "experiment": "M4_web_ABC_100step_development",
        "selection_policy": {
            "rsicd_split": "val",
            "retention_margin_absolute_mean_recall": retention_margin,
            "requires_chatearthnet_step100_improvement": False,
            "requires_rsicd_val_noninferiority": True,
        },
        "rsicd_val": {
            "manifest_sha256": baseline_hash,
            "counts": baseline["counts"],
            "official_mean_recall": baseline_mean_recall,
        },
        "overlap_audit": overlap,
        "runs": summary,
        "promotion_eligible": eligible,
        "recommended_for_250step": eligible[0] if eligible else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-output", required=True, type=Path)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument("--run-c", required=True, type=Path)
    parser.add_argument("--retention-margin", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_m4_abc_development(
        args.development_output,
        {"a": args.run_a, "b": args.run_b, "c": args.run_c},
        retention_margin=args.retention_margin,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
