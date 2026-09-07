#!/usr/bin/env python3
"""Verify and summarize the M5 Web A/C continuation through step 250."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path
from typing import Any

LABELS = ("a", "c")
VALIDATION_STEPS = (0, 50, 100, 150, 200, 250)
RETRIEVAL_STEPS = (100, 150, 200, 250)
DESIGNS = {
    "a": {
        "text_last_k": 4,
        "train_vision_head": True,
        "train_text_projection": True,
        "train_logit_scale": True,
        "learning_rate": 5e-6,
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
    if tuple(sorted(losses)) != VALIDATION_STEPS:
        raise ValueError(f"Unexpected validation steps in {path}: {sorted(losses)}")
    return losses


def _verify_run(label: str, run: Path) -> tuple[dict[int, float], str]:
    config_path = run / "config.toml"
    verification_path = run / "verification_report.json"
    if not config_path.is_file() or not verification_path.is_file():
        raise FileNotFoundError(f"Run {label} lacks config or training verification: {run}")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    observed_design = {
        "text_last_k": config.get("model", {}).get("text_last_k"),
        "train_vision_head": config.get("model", {}).get("train_vision_head"),
        "train_text_projection": config.get("model", {}).get("train_text_projection"),
        "train_logit_scale": config.get("model", {}).get("train_logit_scale"),
        "learning_rate": config.get("train", {}).get("learning_rate"),
    }
    if observed_design != DESIGNS[label]:
        raise ValueError(f"Run {label} config does not match its M4 design: {observed_design}")
    verification = _read_json(verification_path)
    if verification.get("steps") != 250 or verification.get("completed") is not False:
        raise ValueError(f"Run {label} is not a verified bounded 250-step experiment")
    project_commit = verification.get("project_commit")
    if not isinstance(project_commit, str) or not project_commit:
        raise ValueError(f"Run {label} training verification has no project commit")
    resume_steps = {
        record.get("checkpoint_step")
        for record in verification.get("resume_history", [])
        if isinstance(record, dict)
    }
    if not {50, 100}.issubset(resume_steps):
        raise ValueError(f"Run {label} lacks required resume evidence at steps 50 and 100")
    return _validation_losses(run), project_commit


def _retrieval_report(path: Path, expected_step: int | None) -> dict[str, Any]:
    report = _read_json(path)
    if report.get("task") != "rsicd_image_text_retrieval" or report.get("split") != "val":
        raise ValueError(f"Expected an RSICD val retrieval report: {path}")
    manifest = report.get("manifest")
    metrics = report.get("metrics")
    model = report.get("model")
    counts = report.get("counts")
    if not all(isinstance(value, dict) for value in (manifest, metrics, model, counts)):
        raise ValueError(f"Incomplete retrieval report: {path}")
    checkpoint = model.get("checkpoint")
    checkpoint_commit: str | None = None
    if expected_step is None:
        if checkpoint is not None:
            raise ValueError("Official M5 baseline unexpectedly used a checkpoint")
    else:
        if not isinstance(checkpoint, dict) or checkpoint.get("step") != expected_step:
            raise ValueError(f"Retrieval report {path} did not use step {expected_step}")
        identity = checkpoint.get("run_identity")
        if not isinstance(identity, dict) or not isinstance(identity.get("project_commit"), str):
            raise ValueError(f"Retrieval report {path} lacks checkpoint run identity")
        checkpoint_commit = identity["project_commit"]
    mean_recall = _finite(metrics.get("mean_recall"), f"{path} mean recall")
    if not 0 <= mean_recall <= 1:
        raise ValueError(f"Mean recall outside [0, 1]: {path}")
    trainable = model.get("trainable_parameters")
    if not isinstance(trainable, dict) or not isinstance(trainable.get("trainable"), int):
        raise ValueError(f"Retrieval report has no trainable parameter count: {path}")
    return {
        "manifest_sha256": manifest.get("sha256"),
        "mean_recall": mean_recall,
        "counts": counts,
        "trainable_parameters": trainable["trainable"],
        "checkpoint_commit": checkpoint_commit,
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_loss = left["chatearthnet_validation"]["step250"]
    right_loss = right["chatearthnet_validation"]["step250"]
    left_recall = left["rsicd_val_mean_recall"]["step250"]
    right_recall = right["rsicd_val_mean_recall"]["step250"]
    return (
        left_loss <= right_loss
        and left_recall >= right_recall
        and (left_loss < right_loss or left_recall > right_recall)
    )


def verify_m5_ac_250step_development(
    development_output: Path,
    runs: dict[str, Path],
    *,
    retention_margin: float = 0.01,
) -> dict[str, Any]:
    if tuple(sorted(runs)) != LABELS:
        raise ValueError(f"Runs must contain exactly {list(LABELS)}")
    if not 0 <= retention_margin <= 1:
        raise ValueError("retention_margin must be in [0, 1]")

    source_m4 = _read_json(development_output / "source_m4_verification_report.json")
    if source_m4.get("status") != "complete" or source_m4.get("recommended_for_250step") != "a":
        raise ValueError("Source M4 report is incomplete or did not recommend A")
    source_eligible = source_m4.get("promotion_eligible")
    if not isinstance(source_eligible, list) or not set(LABELS).issubset(source_eligible):
        raise ValueError("Source M4 report did not qualify both A and C")
    overlap = _read_json(development_output / "train_vs_rsicd_val_overlap.json")
    if overlap.get("status") != "clear":
        raise ValueError("Training and RSICD val manifests did not pass the overlap audit")

    baseline = _retrieval_report(development_output / "rsicd_val_official.json", None)
    baseline_hash = baseline["manifest_sha256"]
    if not isinstance(baseline_hash, str):
        raise ValueError("Official development report has no manifest SHA-256")
    source_rsicd = source_m4.get("rsicd_val")
    if not isinstance(source_rsicd, dict):
        raise ValueError("Source M4 report has no RSICD val identity")
    if (
        source_rsicd.get("manifest_sha256") != baseline_hash
        or source_rsicd.get("counts") != baseline["counts"]
    ):
        raise ValueError("M5 baseline differs from the source M4 RSICD val input")
    baseline_mean_recall = baseline["mean_recall"]
    if not math.isclose(
        _finite(source_rsicd.get("official_mean_recall"), "M4 official mean recall"),
        baseline_mean_recall,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("M5 official mean recall differs from the source M4 report")
    threshold = max(0.0, baseline_mean_recall - retention_margin)

    summary: dict[str, Any] = {}
    eligible: list[str] = []
    project_commits: set[str] = set()
    trainable_counts: dict[str, int] = {}
    for label in LABELS:
        validation, project_commit = _verify_run(label, runs[label])
        project_commits.add(project_commit)
        retrieval: dict[int, dict[str, Any]] = {}
        for step in RETRIEVAL_STEPS:
            report = _retrieval_report(
                development_output / f"rsicd_val_{label}_step{step}.json", step
            )
            if report["manifest_sha256"] != baseline_hash or report["counts"] != baseline["counts"]:
                raise ValueError(f"Run {label} step {step} used a different RSICD val input")
            if report["checkpoint_commit"] != project_commit:
                raise ValueError(f"Run {label} step {step} has a different training commit")
            retrieval[step] = report
        observed_counts = {report["trainable_parameters"] for report in retrieval.values()}
        if len(observed_counts) != 1:
            raise ValueError(f"Run {label} trainable parameter count changed across evaluation")
        trainable_counts[label] = observed_counts.pop()

        final_recall = retrieval[250]["mean_recall"]
        same_domain_pass = validation[250] < validation[0]
        retention_pass = final_recall >= threshold
        promotion_eligible = same_domain_pass and retention_pass
        if promotion_eligible:
            eligible.append(label)
        recall_trajectory = [retrieval[step]["mean_recall"] for step in RETRIEVAL_STEPS]
        monotonic_decline = all(
            current < previous
            for previous, current in zip(recall_trajectory, recall_trajectory[1:], strict=False)
        )
        summary[label] = {
            "design": DESIGNS[label],
            "run": str(runs[label].resolve()),
            "trainable_parameters": trainable_counts[label],
            "chatearthnet_validation": {
                **{f"step{step}": validation[step] for step in VALIDATION_STEPS},
                "step250_change_from_step0": validation[250] - validation[0],
                "improved_at_step250": same_domain_pass,
            },
            "rsicd_val_mean_recall": {
                "official": baseline_mean_recall,
                **{f"step{step}": retrieval[step]["mean_recall"] for step in RETRIEVAL_STEPS},
                "step250_change_from_official": final_recall - baseline_mean_recall,
                "step250_change_from_step100": final_recall - retrieval[100]["mean_recall"],
                "retention_threshold": threshold,
                "retention_pass": retention_pass,
                "monotonic_decline_from_step100": monotonic_decline,
            },
            "promotion_eligible": promotion_eligible,
        }

    if len(project_commits) != 1:
        raise ValueError("A and C were not continued from the same M4 project commit")
    if trainable_counts["a"] <= trainable_counts["c"]:
        raise ValueError("Full-scope A must train more parameters than vision-head-only C")

    frontier = [
        label
        for label in eligible
        if not any(
            other != label and _dominates(summary[other], summary[label]) for other in eligible
        )
    ]
    recommended = frontier[0] if len(frontier) == 1 else None
    if recommended is not None:
        decision_status = "candidate_selected"
    elif eligible:
        decision_status = "tradeoff_review_required"
    else:
        decision_status = "no_candidate_passed"

    return {
        "format_version": 1,
        "status": "complete",
        "experiment": "M5_web_AC_250step_development",
        "source_training_commit": project_commits.pop(),
        "selection_policy": {
            "rsicd_split": "val",
            "retention_margin_absolute_mean_recall": retention_margin,
            "requires_chatearthnet_step250_improvement": True,
            "requires_rsicd_val_noninferiority": True,
            "recommendation_rule": "unique Pareto-nondominated candidate at step 250",
        },
        "rsicd_val": {
            "manifest_sha256": baseline_hash,
            "counts": baseline["counts"],
            "official_mean_recall": baseline_mean_recall,
        },
        "overlap_audit": overlap,
        "runs": summary,
        "eligible_for_500step": eligible,
        "pareto_frontier": frontier,
        "recommended_for_500step": recommended,
        "decision_status": decision_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-output", required=True, type=Path)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-c", required=True, type=Path)
    parser.add_argument("--retention-margin", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_m5_ac_250step_development(
        args.development_output,
        {"a": args.run_a, "c": args.run_c},
        retention_margin=args.retention_margin,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
