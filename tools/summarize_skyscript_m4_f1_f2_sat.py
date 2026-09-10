#!/usr/bin/env python3
"""Summarize the SAT-only F1/F2 mechanism screen against the archived F0 baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

STAGES = (100, 250, 500)
CANDIDATES = (
    "f1_visionhead_lr1e5",
    "f1_visionhead_lr5e6",
    "f2_adapter256_visionhead_lr1e5",
    "f2_adapter256_visionhead_lr5e6",
)


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


def _candidate(candidate: str, run_dir: Path, gate_dir: Path) -> dict[str, Any]:
    training = _read_json(gate_dir / "training_step500.json")
    if training.get("steps") != 500 or training.get("completed") is not True:
        raise ValueError(f"Candidate {candidate} has no completed training verification")
    if training.get("visual_backbone_permanently_frozen") is not True:
        raise ValueError(f"Candidate {candidate} did not verify the permanent backbone freeze")
    groups = training.get("optimizer_parameter_groups", {}).get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Candidate {candidate} has no optimizer group evidence")
    group_names = [group.get("name") for group in groups]
    expected_groups = (
        ["vision_head"]
        if candidate.startswith("f1_")
        else ["image_adapter", "vision_head"]
    )
    if group_names != expected_groups:
        raise ValueError(
            f"Candidate {candidate} groups are {group_names}, expected {expected_groups}"
        )

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
            sky = report["skyscript"]["steps"][key]["mean_recall"]
            rsicd = report["rsicd_val"]["steps"][key]["mean_recall"]
            validation = report["validation_loss"][key]
            f0_sky = baseline["skyscript"]["steps"][key]["mean_recall"]
            f0_rsicd = baseline["rsicd_val"]["steps"][key]["mean_recall"]
            f0_val = f0_validation[key]
            by_stage[key] = {
                "validation_loss_minus_f0": validation - f0_val,
                "skyscript_mean_recall_minus_f0": sky - f0_sky,
                "rsicd_mean_recall_minus_f0": rsicd - f0_rsicd,
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
            "skyscript_above_f0_at_step500": (
                step500["skyscript_mean_recall_minus_f0"] > 0
            ),
            "rsicd_above_f0_at_step500": (
                step500["rsicd_mean_recall_minus_f0"] > 0
            ),
        }
        if all(signals.values()):
            eligible.append(name)
        comparisons[name] = {"by_stage": by_stage, "signals": signals}

    ranking = sorted(
        CANDIDATES,
        key=lambda name: (
            candidates[name]["skyscript"]["steps"]["500"]["mean_recall"],
            candidates[name]["rsicd_val"]["steps"]["500"]["mean_recall"],
        ),
        reverse=True,
    )
    return {
        "format_version": 1,
        "study": "SkyScript_M4_SAT_F1_F2_mechanism_screen",
        "status": "complete",
        "seed": 11,
        "stages": list(STAGES),
        "visual_backbone_permanently_frozen": True,
        "f0": {
            "reused_not_retrained": True,
            "summary": str(baseline_path),
            "validation_loss": f0_validation,
            "skyscript": baseline["skyscript"],
            "rsicd_val": baseline["rsicd_val"],
        },
        "candidates": candidates,
        "comparisons_to_f0": comparisons,
        "eligible_candidates": eligible,
        "step500_ranking_by_skyscript_then_rsicd": ranking,
        "interpretation_policy": (
            "This screen records mechanism signals only. Review the complete directional retrieval "
            "metrics before selecting a scope; do not start Web controls from this report alone."
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
