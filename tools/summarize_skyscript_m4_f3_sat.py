#!/usr/bin/env python3
"""Summarize SAT F3 text-projection candidates against the archived F0 anchor."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

STAGES = (100, 250, 500)
CANDIDATES = (
    "f3_adapter256_textproj_lr1e5",
    "f3_adapter256_textproj_lr5e6",
)
F0_RSICD_STEP500_SAMPLE_STD = 0.0032465


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _evaluation(path: Path, step: int | None) -> dict[str, Any]:
    report = _read_json(path)
    model = report.get("model")
    if not isinstance(model, dict) or model.get("backbone_domain") != "sat":
        raise ValueError(f"Evaluation is not tied to the SAT backbone: {path}")
    checkpoint = model.get("checkpoint")
    if step is None:
        if checkpoint is not None:
            raise ValueError(f"Expected official initialization: {path}")
    elif not isinstance(checkpoint, dict) or checkpoint.get("step") != step:
        raise ValueError(f"Expected checkpoint step {step}: {path}")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Evaluation has no metrics: {path}")
    _finite(metrics.get("mean_recall"), f"{path} mean recall")
    return metrics


def _validation(run_dir: Path) -> dict[str, float]:
    records = [
        json.loads(line)
        for line in (run_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    values = {str(record["step"]): _finite(record["loss"], "validation loss") for record in records}
    required = {"0", *(str(stage) for stage in STAGES)}
    if not required.issubset(values):
        missing = sorted(required - values.keys())
        raise ValueError(f"Missing validation stages in {run_dir}: {missing}")
    return {step: values[step] for step in ("0", "100", "250", "500")}


def _drift(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    if report.get("task") != "text_embedding_drift_against_frozen_official_reference":
        raise ValueError(f"Unexpected text drift task: {path}")
    if report.get("model", {}).get("backbone_domain") != "sat":
        raise ValueError(f"Text drift report is not tied to the SAT config: {path}")
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError(f"Text drift report has no checkpoints: {path}")
    required = {"0", *(str(stage) for stage in STAGES)}
    if set(checkpoints) != required:
        raise ValueError(f"Text drift checkpoints differ from protocol: {path}")
    for step, checkpoint_report in checkpoints.items():
        checkpoint = checkpoint_report.get("checkpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("step") != int(step):
            raise ValueError(f"Text drift checkpoint identity mismatch at step {step}")
        aggregate = checkpoint_report.get("aggregate")
        if not isinstance(aggregate, dict):
            raise ValueError(f"Text drift aggregate missing at step {step}")
        _finite(
            aggregate.get("cosine_distance", {}).get("mean"),
            f"text drift mean at step {step}",
        )
    return report


def _candidate(candidate: str, run_dir: Path, gate_dir: Path) -> dict[str, Any]:
    training = _read_json(gate_dir / "training_step500.json")
    if training.get("steps") != 500 or training.get("completed") is not True:
        raise ValueError(f"Candidate {candidate} has no completed training verification")
    if training.get("visual_backbone_permanently_frozen") is not True:
        raise ValueError(f"Candidate {candidate} did not verify the permanent backbone freeze")
    groups = training.get("optimizer_parameter_groups", {}).get("groups")
    if not isinstance(groups, list) or [group.get("name") for group in groups] != [
        "image_adapter",
        "text_projection",
    ]:
        raise ValueError(f"Candidate {candidate} has unexpected optimizer groups")

    validation = _validation(run_dir)
    sky = {
        str(step): _evaluation(gate_dir / f"skyscript_val_step{step:03d}.json", step)
        for step in (0, *STAGES)
    }
    rsicd_official = _evaluation(gate_dir / "rsicd_val_official.json", None)
    rsicd = {
        str(step): _evaluation(gate_dir / f"rsicd_val_step{step:03d}.json", step)
        for step in (0, *STAGES)
    }
    if not math.isclose(
        _finite(rsicd["0"]["mean_recall"], "RSICD step0"),
        _finite(rsicd_official["mean_recall"], "RSICD official"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Candidate {candidate} RSICD step0 differs from official")
    drift = _drift(gate_dir / "text_embedding_drift.json")
    return {
        "run_dir": str(run_dir),
        "gate_dir": str(gate_dir),
        "optimizer_groups": groups,
        "validation_loss": validation,
        "skyscript": {
            "steps": sky,
            "minus_step0_mean_recall": {
                str(stage): sky[str(stage)]["mean_recall"] - sky["0"]["mean_recall"]
                for stage in STAGES
            },
        },
        "rsicd_val": {
            "official": rsicd_official,
            "steps": rsicd,
            "minus_official_mean_recall": {
                str(stage): rsicd[str(stage)]["mean_recall"]
                - rsicd_official["mean_recall"]
                for stage in STAGES
            },
        },
        "text_embedding_drift": {
            "report": str(gate_dir / "text_embedding_drift.json"),
            "reference_definition": drift["reference_definition"],
            "manifests": drift["manifests"],
            "checkpoints": {
                step: {
                    "aggregate": report["aggregate"],
                    "datasets": report["datasets"],
                }
                for step, report in drift["checkpoints"].items()
            },
        },
    }


def summarize(
    *, baseline_path: Path, candidate_root: Path, output_root: Path
) -> dict[str, Any]:
    baseline = _read_json(baseline_path)
    if (
        baseline.get("gate") != "SkyScript_M4_SAT"
        or baseline.get("status") != "pass"
        or baseline.get("stages") != list(STAGES)
    ):
        raise ValueError("F0 prerequisite is not the archived passing M4-A SAT summary")
    f0_validation = baseline.get("validation_loss", {}).get("steps")
    if not isinstance(f0_validation, dict):
        raise ValueError("F0 summary has no validation loss series")

    candidates: dict[str, Any] = {}
    for candidate in CANDIDATES:
        run_dir = candidate_root / f"skyscript_sat_{candidate}_36495_500step_seed11"
        gate_dir = output_root / f"skyscript_gate_m4_{candidate}_sat_seed11"
        candidates[candidate] = _candidate(candidate, run_dir, gate_dir)

    comparisons: dict[str, Any] = {}
    eligible: list[str] = []
    for name, report in candidates.items():
        by_stage: dict[str, Any] = {}
        for stage in STAGES:
            key = str(stage)
            by_stage[key] = {
                "validation_loss_minus_f0": (
                    report["validation_loss"][key] - f0_validation[key]
                ),
                "skyscript_mean_recall_minus_f0": (
                    report["skyscript"]["steps"][key]["mean_recall"]
                    - baseline["skyscript"]["steps"][key]["mean_recall"]
                ),
                "rsicd_mean_recall_minus_f0": (
                    report["rsicd_val"]["steps"][key]["mean_recall"]
                    - baseline["rsicd_val"]["steps"][key]["mean_recall"]
                ),
            }
        step500 = by_stage["500"]
        signals = {
            "validation_below_own_step0": (
                report["validation_loss"]["500"] < report["validation_loss"]["0"]
            ),
            "skyscript_above_own_step0": (
                report["skyscript"]["minus_step0_mean_recall"]["500"] > 0
            ),
            "rsicd_above_own_official": (
                report["rsicd_val"]["minus_official_mean_recall"]["500"] > 0
            ),
            "validation_below_f0_at_step500": (
                step500["validation_loss_minus_f0"] < 0
            ),
            "skyscript_above_f0_at_steps250_and500": all(
                by_stage[str(stage)]["skyscript_mean_recall_minus_f0"] > 0
                for stage in (250, 500)
            ),
            "rsicd_within_f0_seed_std_at_step500": (
                step500["rsicd_mean_recall_minus_f0"]
                >= -F0_RSICD_STEP500_SAMPLE_STD
            ),
        }
        if all(signals.values()):
            eligible.append(name)
        comparisons[name] = {
            "by_stage": by_stage,
            "signals": signals,
            "step500_aggregate_mean_cosine_distance_from_official_text": report[
                "text_embedding_drift"
            ]["checkpoints"]["500"]["aggregate"]["cosine_distance"]["mean"],
        }

    ranking_pool = eligible or list(CANDIDATES)
    ranking = sorted(
        ranking_pool,
        key=lambda name: (
            candidates[name]["skyscript"]["steps"]["500"]["mean_recall"],
            -candidates[name]["validation_loss"]["500"],
            candidates[name]["rsicd_val"]["steps"]["500"]["mean_recall"],
        ),
        reverse=True,
    )
    return {
        "format_version": 1,
        "study": "SkyScript_M4_SAT_F3_text_projection_mechanism_screen",
        "status": "complete",
        "seed": 11,
        "stages": list(STAGES),
        "visual_backbone_permanently_frozen": True,
        "vision_head_frozen": True,
        "f0": {
            "reused_not_retrained": True,
            "summary": str(baseline_path),
            "validation_loss": f0_validation,
            "skyscript": baseline["skyscript"],
            "rsicd_val": baseline["rsicd_val"],
        },
        "candidates": candidates,
        "comparisons_to_f0": comparisons,
        "screen_selection_rule": {
            "primary": "SkyScript mean recall above matched F0 at both steps 250 and 500",
            "validation": "step500 validation loss below matched F0",
            "external_retention": (
                "step500 RSICD mean recall no more than one archived F0 cross-seed sample "
                "standard deviation below matched F0"
            ),
            "f0_rsicd_step500_sample_std": F0_RSICD_STEP500_SAMPLE_STD,
            "own_trajectory": (
                "step500 validation improves and both retrieval datasets improve from their "
                "respective initialization"
            ),
            "text_drift": "diagnostic, reported but not thresholded in this seed11 screen",
        },
        "eligible_candidates": eligible,
        "ranking_scope": "eligible_candidates" if eligible else "all_candidates_diagnostic_only",
        "ranking_by_skyscript_then_validation_then_rsicd": ranking,
        "interpretation_policy": (
            "This seed11 screen selects a trainable scope and LR magnitude only. Inspect complete "
            "directional retrieval and text-drift distributions; do not claim stability, start "
            "Web controls, or advance to F4 solely because a candidate improves over its own step0."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        baseline_path=args.baseline,
        candidate_root=args.candidate_root,
        output_root=args.output_root,
    )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Refusing to overwrite different summary: {args.output}")
        print(f"summary={args.output}")
        return
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"summary={args.output}")


if __name__ == "__main__":
    main()
