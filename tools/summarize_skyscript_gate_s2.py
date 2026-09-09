#!/usr/bin/env python3
"""Validate staged SkyScript Gate S2 evidence and write immutable summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

STAGES = (100, 250, 500)
VALIDATION_EVERY = 50
RETENTION_MARGIN = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage", help="Validate one 100/250/500-step boundary")
    stage.add_argument("--run-dir", required=True, type=Path)
    stage.add_argument("--gate-dir", required=True, type=Path)
    stage.add_argument("--stage", required=True, type=int, choices=STAGES)
    stage.add_argument("--training-report", required=True, type=Path)
    stage.add_argument("--output", required=True, type=Path)

    final = subparsers.add_parser("final", help="Combine the three passing stage summaries")
    final.add_argument("--s1-summary", required=True, type=Path)
    final.add_argument("--gate-dir", required=True, type=Path)
    final.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL record: {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {path}:{line_number}")
        records.append(value)
    return records


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _validate_metrics(metrics: dict[str, Any], label: str) -> None:
    for direction in ("image_to_text", "text_to_image"):
        values = metrics.get(direction)
        if not isinstance(values, dict):
            raise ValueError(f"{label} has no {direction} metrics")
        for key in ("r1", "r5", "r10", "median_rank", "mean_rank"):
            _finite(values.get(key), f"{label} {direction} {key}")
    _finite(metrics.get("mean_recall"), f"{label} mean_recall")


def _validate_evaluation(
    path: Path,
    *,
    task: str,
    manifest_sha: str,
    checkpoint_step: int | None,
    source: str | None,
) -> dict[str, Any]:
    report = _read_json(path)
    if report.get("task") != task or report.get("split") != "val":
        raise ValueError(f"Unexpected task or split in {path}")
    if report.get("manifest", {}).get("sha256") != manifest_sha:
        raise ValueError(f"Manifest identity mismatch in {path}")
    checkpoint = report.get("model", {}).get("checkpoint")
    if checkpoint_step is None:
        if checkpoint is not None:
            raise ValueError(f"Expected official initialization in {path}")
    elif not isinstance(checkpoint, dict) or checkpoint.get("step") != checkpoint_step:
        raise ValueError(f"Expected checkpoint step {checkpoint_step} in {path}")
    if source is not None and report.get("sources") != [source]:
        raise ValueError(f"Unexpected source identity in {path}")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing metrics in {path}")
    _validate_metrics(metrics, str(path))
    return metrics


def _validate_overlap(path: Path, left_sha: str, right_sha: str) -> None:
    report = _read_json(path)
    if (
        report.get("status") != "clear"
        or report.get("exact_file_overlap_count") != 0
        or report.get("exact_decoded_pixel_overlap_count") != 0
        or report.get("left_manifest", {}).get("sha256") != left_sha
        or report.get("right_manifest", {}).get("sha256") != right_sha
    ):
        raise ValueError(f"Overlap report is not clear or has the wrong identity: {path}")


def _write_atomic_immutable(destination: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Refusing to overwrite a different summary: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"Incomplete prior summary exists: {temporary}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, destination)


def summarize_stage(
    *,
    run_dir: Path,
    gate_dir: Path,
    stage: int,
    training_report_path: Path,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Stage must be one of {STAGES}, got {stage}")
    training = _read_json(training_report_path)
    expected_completed = stage == 500
    if (
        training.get("steps") != stage
        or training.get("completed") is not expected_completed
        or training.get("final_queue_size") != 0
    ):
        raise ValueError(f"Training report does not describe the expected stage {stage}")
    resume_steps = [record.get("checkpoint_step") for record in training.get("resume_history", [])]
    expected_resume_steps = [] if stage == 100 else ([100] if stage == 250 else [100, 250])
    if resume_steps != expected_resume_steps:
        raise ValueError(
            "Training report resume boundaries are "
            f"{resume_steps}, expected {expected_resume_steps}"
        )
    train_sha = training.get("train_manifest_sha256")
    val_sha = training.get("val_manifest_sha256")
    if not isinstance(train_sha, str) or not isinstance(val_sha, str):
        raise ValueError("Training report has no manifest identities")

    validation_records = [
        record
        for record in _read_jsonl(run_dir / "validation.jsonl")
        if record.get("step", 0) <= stage
    ]
    expected_steps = list(range(0, stage + 1, VALIDATION_EVERY))
    if [record.get("step") for record in validation_records] != expected_steps:
        raise ValueError(f"Validation records through stage {stage} are incomplete")
    validation = {
        str(record["step"]): _finite(record.get("loss"), f"validation step {record['step']}")
        for record in validation_records
    }
    best_step = min(expected_steps, key=lambda step: validation[str(step)])
    if training.get("best_validation_step") != best_step:
        raise ValueError("Training report best step disagrees with validation.jsonl")

    parity = _read_json(run_dir / "skyscript_step0_parity.json")
    if (
        parity.get("format_version") != 2
        or parity.get("status") != "pass"
        or parity.get("checkpoint_step") != 0
        or parity.get("input", {}).get("manifest_sha256") != val_sha
    ):
        raise ValueError("Step-0 parity report failed or has the wrong identity")

    _validate_overlap(gate_dir / "train_vs_val_overlap.json", train_sha, val_sha)
    rsicd_overlap = _read_json(gate_dir / "train_vs_rsicd_val_overlap.json")
    rsicd_sha = rsicd_overlap.get("right_manifest", {}).get("sha256")
    if not isinstance(rsicd_sha, str):
        raise ValueError("Training/RSICD overlap report has no right manifest identity")
    _validate_overlap(gate_dir / "train_vs_rsicd_val_overlap.json", train_sha, rsicd_sha)

    sky_zero = _validate_evaluation(
        gate_dir / "skyscript_val_step000.json",
        task="paired_image_text_global_retrieval",
        manifest_sha=val_sha,
        checkpoint_step=0,
        source="SkyScript",
    )
    sky_stage = _validate_evaluation(
        gate_dir / f"skyscript_val_step{stage:03d}.json",
        task="paired_image_text_global_retrieval",
        manifest_sha=val_sha,
        checkpoint_step=stage,
        source="SkyScript",
    )
    rsicd_official = _validate_evaluation(
        gate_dir / "rsicd_val_official.json",
        task="rsicd_image_text_retrieval",
        manifest_sha=rsicd_sha,
        checkpoint_step=None,
        source=None,
    )
    rsicd_zero = _validate_evaluation(
        gate_dir / "rsicd_val_step000.json",
        task="rsicd_image_text_retrieval",
        manifest_sha=rsicd_sha,
        checkpoint_step=0,
        source=None,
    )
    rsicd_stage = _validate_evaluation(
        gate_dir / f"rsicd_val_step{stage:03d}.json",
        task="rsicd_image_text_retrieval",
        manifest_sha=rsicd_sha,
        checkpoint_step=stage,
        source=None,
    )
    if not math.isclose(
        _finite(rsicd_zero["mean_recall"], "RSICD step0 mean recall"),
        _finite(rsicd_official["mean_recall"], "RSICD official mean recall"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("RSICD step-0 mean recall does not match official initialization")

    sky_delta = _finite(sky_stage["mean_recall"], "SkyScript stage mean recall") - _finite(
        sky_zero["mean_recall"], "SkyScript step0 mean recall"
    )
    rsicd_delta = _finite(rsicd_stage["mean_recall"], "RSICD stage mean recall") - _finite(
        rsicd_official["mean_recall"], "RSICD official mean recall"
    )
    checks = {
        "training_artifacts": True,
        "train_validation_overlap": True,
        "step0_parity": True,
        "training_rsicd_val_overlap": True,
        "validation_current_below_step0_and_best_not_step0": (
            validation[str(stage)] < validation["0"] and best_step != 0
        ),
        "skyscript_current_mean_recall_above_step0": sky_delta > 0,
        "rsicd_current_mean_recall_drop_within_0.01": rsicd_delta >= -RETENTION_MARGIN,
    }
    return {
        "format_version": 1,
        "gate": "SkyScript_S2_stage",
        "stage": stage,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "criteria": {
            "validation": f"step{stage} loss < step0 loss and best step through stage != 0",
            "skyscript": f"step{stage} mean_recall > step0 mean_recall",
            "rsicd_retention": (
                f"step{stage} mean_recall - official mean_recall >= -{RETENTION_MARGIN}"
            ),
        },
        "training": {
            "report": str(training_report_path),
            "project_commit": training.get("project_commit"),
            "dinov3_commit": training.get("dinov3_commit"),
            "train_manifest_sha256": train_sha,
            "val_manifest_sha256": val_sha,
            "completed": training["completed"],
            "best_validation_step": best_step,
        },
        "validation_loss": {"steps": validation, "best_step": best_step},
        "skyscript": {
            "step0": sky_zero,
            f"step{stage}": sky_stage,
            "current_minus_step0_mean_recall": sky_delta,
        },
        "rsicd_val": {
            "official": rsicd_official,
            "step0": rsicd_zero,
            f"step{stage}": rsicd_stage,
            "current_minus_official_mean_recall": rsicd_delta,
        },
    }


def summarize_final(*, s1_summary_path: Path, gate_dir: Path) -> dict[str, Any]:
    s1 = _read_json(s1_summary_path)
    if s1.get("gate") != "SkyScript_S1" or s1.get("status") != "pass":
        raise ValueError("Gate S1 prerequisite is not a passing SkyScript_S1 summary")
    stages = {
        stage: _read_json(gate_dir / f"stage_{stage}_summary.json") for stage in STAGES
    }
    for stage, report in stages.items():
        if (
            report.get("gate") != "SkyScript_S2_stage"
            or report.get("stage") != stage
            or report.get("status") not in {"pass", "fail"}
        ):
            raise ValueError(f"Invalid Gate S2 stage summary for step {stage}")
        checks = report.get("checks")
        if not isinstance(checks, dict) or not checks:
            raise ValueError(f"Gate S2 stage {stage} has no checks")
        expected_status = "pass" if all(checks.values()) else "fail"
        if report["status"] != expected_status:
            raise ValueError(f"Gate S2 stage {stage} status disagrees with its checks")
    commits = {report.get("training", {}).get("project_commit") for report in stages.values()}
    if len(commits) != 1 or None in commits:
        raise ValueError("Gate S2 stage summaries do not have one training project commit")
    status = "pass" if all(report["status"] == "pass" for report in stages.values()) else "fail"
    final_stage = stages[500]
    return {
        "format_version": 1,
        "gate": "SkyScript_S2",
        "status": status,
        "stages": list(STAGES),
        "prerequisite_s1": {
            "path": str(s1_summary_path),
            "sha256": hashlib.sha256(s1_summary_path.read_bytes()).hexdigest(),
            "status": "pass",
            "seeds": s1.get("seeds"),
        },
        "project_commit": next(iter(commits)),
        "criteria": {
            "all_stage_gates": "steps 100, 250, and 500 must each pass before progression",
            "initialization": (
                "new run from official initialization; not resumed from a 100-step run"
            ),
            "resume_boundaries": "strict resume at steps 100 and 250",
        },
        "stage_checks": {str(stage): stages[stage]["checks"] for stage in STAGES},
        "validation_loss": final_stage["validation_loss"],
        "skyscript": {
            "step0": stages[100]["skyscript"]["step0"],
            "steps": {
                str(stage): stages[stage]["skyscript"][f"step{stage}"] for stage in STAGES
            },
            "current_minus_step0_mean_recall": {
                str(stage): stages[stage]["skyscript"]["current_minus_step0_mean_recall"]
                for stage in STAGES
            },
        },
        "rsicd_val": {
            "official": stages[100]["rsicd_val"]["official"],
            "step0": stages[100]["rsicd_val"]["step0"],
            "steps": {
                str(stage): stages[stage]["rsicd_val"][f"step{stage}"] for stage in STAGES
            },
            "current_minus_official_mean_recall": {
                str(stage): stages[stage]["rsicd_val"][
                    "current_minus_official_mean_recall"
                ]
                for stage in STAGES
            },
        },
        "stage_reports": {
            str(stage): str(gate_dir / f"stage_{stage}_summary.json") for stage in STAGES
        },
    }


def main() -> None:
    args = parse_args()
    if args.command == "stage":
        summary = summarize_stage(
            run_dir=args.run_dir,
            gate_dir=args.gate_dir,
            stage=args.stage,
            training_report_path=args.training_report,
        )
    else:
        summary = summarize_final(s1_summary_path=args.s1_summary, gate_dir=args.gate_dir)
    _write_atomic_immutable(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
