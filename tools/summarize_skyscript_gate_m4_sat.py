#!/usr/bin/env python3
"""Validate staged SAT evidence for SkyScript M4-A.

The numerical and artifact Gate is deliberately delegated to the frozen S2
summarizer so the Web and SAT thresholds cannot drift. This wrapper adds the
SAT-specific model identity checks and M4 report labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.summarize_skyscript_gate_s2 import (
    STAGES,
    _read_json,
    _write_atomic_immutable,
)
from tools.summarize_skyscript_gate_s2 import summarize_stage as summarize_s2_stage

SAT_BACKBONE_SHA256 = "eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48"
DINOTXT_SHA256 = "a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0"
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
ADAPTER_TRAINABLE_PARAMETERS = 1_054_976


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage", help="Validate one SAT stage boundary")
    stage.add_argument("--run-dir", required=True, type=Path)
    stage.add_argument("--gate-dir", required=True, type=Path)
    stage.add_argument("--stage", required=True, type=int, choices=STAGES)
    stage.add_argument("--training-report", required=True, type=Path)
    stage.add_argument("--output", required=True, type=Path)

    final = subparsers.add_parser("final", help="Combine three passing SAT stage reports")
    final.add_argument("--gate-dir", required=True, type=Path)
    final.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _validate_file_identity(metadata: dict[str, Any], name: str, expected_sha: str) -> None:
    record = metadata.get(name)
    if not isinstance(record, dict) or record.get("sha256") != expected_sha:
        raise ValueError(f"Unexpected SAT evidence identity for {name}")


def _validate_model_identity(metadata: Any, *, expected_trainable: int) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("SAT evidence has no model metadata")
    if metadata.get("backbone_domain") != "sat":
        raise ValueError("SAT evidence was not evaluated with backbone_domain='sat'")
    _validate_file_identity(metadata, "backbone_weights", SAT_BACKBONE_SHA256)
    _validate_file_identity(metadata, "dinotxt_weights", DINOTXT_SHA256)
    _validate_file_identity(metadata, "bpe_vocab", BPE_SHA256)
    trainable = metadata.get("trainable_parameters")
    if not isinstance(trainable, dict) or trainable.get("trainable") != expected_trainable:
        raise ValueError(
            "Unexpected SAT evidence trainable parameter count: "
            f"expected {expected_trainable}"
        )


def validate_sat_evidence_identity(*, run_dir: Path, gate_dir: Path, stage: int) -> None:
    parity = _read_json(run_dir / "skyscript_step0_parity.json")
    if parity.get("backbone_domain") != "sat":
        raise ValueError("Step-0 parity report does not describe the SAT backbone")
    _validate_model_identity(parity.get("official_model"), expected_trainable=0)
    _validate_model_identity(
        parity.get("step0_model"), expected_trainable=ADAPTER_TRAINABLE_PARAMETERS
    )

    evaluation_paths = (
        gate_dir / "skyscript_val_step000.json",
        gate_dir / f"skyscript_val_step{stage:03d}.json",
        gate_dir / "rsicd_val_official.json",
        gate_dir / "rsicd_val_step000.json",
        gate_dir / f"rsicd_val_step{stage:03d}.json",
    )
    for path in evaluation_paths:
        report = _read_json(path)
        _validate_model_identity(
            report.get("model"), expected_trainable=ADAPTER_TRAINABLE_PARAMETERS
        )


def summarize_stage(
    *, run_dir: Path, gate_dir: Path, stage: int, training_report_path: Path
) -> dict[str, Any]:
    summary = summarize_s2_stage(
        run_dir=run_dir,
        gate_dir=gate_dir,
        stage=stage,
        training_report_path=training_report_path,
    )
    validate_sat_evidence_identity(run_dir=run_dir, gate_dir=gate_dir, stage=stage)
    summary["gate"] = "SkyScript_M4_SAT_stage"
    summary["domain"] = "sat"
    summary["shared_gate_implementation"] = "SkyScript_S2_stage"
    summary["model_identity"] = {
        "backbone_domain": "sat",
        "backbone_weights_sha256": SAT_BACKBONE_SHA256,
        "dinotxt_weights_sha256": DINOTXT_SHA256,
        "bpe_vocab_sha256": BPE_SHA256,
        "trainable_parameters": ADAPTER_TRAINABLE_PARAMETERS,
    }
    return summary


def summarize_final(*, gate_dir: Path) -> dict[str, Any]:
    stages = {stage: _read_json(gate_dir / f"stage_{stage}_summary.json") for stage in STAGES}
    for stage, report in stages.items():
        if (
            report.get("gate") != "SkyScript_M4_SAT_stage"
            or report.get("domain") != "sat"
            or report.get("stage") != stage
            or report.get("status") not in {"pass", "fail"}
        ):
            raise ValueError(f"Invalid M4 SAT stage summary for step {stage}")
        checks = report.get("checks")
        if not isinstance(checks, dict) or not checks:
            raise ValueError(f"M4 SAT stage {stage} has no checks")
        expected_status = "pass" if all(checks.values()) else "fail"
        if report["status"] != expected_status:
            raise ValueError(f"M4 SAT stage {stage} status disagrees with its checks")

    commits = {report.get("training", {}).get("project_commit") for report in stages.values()}
    if len(commits) != 1 or None in commits:
        raise ValueError("M4 SAT stage reports do not have one training project commit")
    identities = {
        json.dumps(report.get("model_identity"), sort_keys=True)
        for report in stages.values()
    }
    if len(identities) != 1:
        raise ValueError("M4 SAT stage reports do not have one model identity")

    status = "pass" if all(report["status"] == "pass" for report in stages.values()) else "fail"
    final_stage = stages[500]
    return {
        "format_version": 1,
        "gate": "SkyScript_M4_SAT",
        "status": status,
        "domain": "sat",
        "stages": list(STAGES),
        "project_commit": next(iter(commits)),
        "model_identity": final_stage["model_identity"],
        "criteria": {
            "all_stage_gates": "steps 100, 250, and 500 must pass before progression",
            "initialization": "new SAT run from its own official initialization",
            "resume_boundaries": "strict resume at steps 100 and 250",
            "comparison_baseline": "each SAT stage is compared with SAT step0/official",
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
        summary = summarize_final(gate_dir=args.gate_dir)
    _write_atomic_immutable(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
