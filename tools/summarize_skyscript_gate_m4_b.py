#!/usr/bin/env python3
"""Validate one staged Web/SAT run in the SkyScript M4-B seed matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.summarize_skyscript_gate_s2 import (  # noqa: E402
    STAGES,
    _read_json,
    _write_atomic_immutable,
)
from tools.summarize_skyscript_gate_s2 import (  # noqa: E402
    summarize_stage as summarize_s2_stage,
)

SEEDS = (23, 47)
MODEL_IDENTITIES = {
    "web": {
        "backbone_weights_sha256": (
            "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
        ),
        "dinotxt_weights_sha256": (
            "a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0"
        ),
        "bpe_vocab_sha256": (
            "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
        ),
        "trainable_parameters": 1_054_976,
    },
    "sat": {
        "backbone_weights_sha256": (
            "eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48"
        ),
        "dinotxt_weights_sha256": (
            "a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0"
        ),
        "bpe_vocab_sha256": (
            "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
        ),
        "trainable_parameters": 1_054_976,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("stage", "final"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--domain", required=True, choices=MODEL_IDENTITIES)
        subparser.add_argument("--seed", required=True, type=int, choices=SEEDS)
        subparser.add_argument("--gate-dir", required=True, type=Path)
        subparser.add_argument("--output", required=True, type=Path)
        if command == "stage":
            subparser.add_argument("--run-dir", required=True, type=Path)
            subparser.add_argument("--stage", required=True, type=int, choices=STAGES)
            subparser.add_argument("--training-report", required=True, type=Path)
    return parser.parse_args()


def _validate_file_identity(metadata: dict[str, Any], name: str, expected: str) -> None:
    record = metadata.get(name)
    if not isinstance(record, dict) or record.get("sha256") != expected:
        raise ValueError(f"Unexpected M4-B evidence identity for {name}")


def _validate_model_identity(metadata: Any, domain: str, trainable: int) -> None:
    if not isinstance(metadata, dict) or metadata.get("backbone_domain") != domain:
        raise ValueError(f"M4-B evidence does not use backbone_domain={domain!r}")
    identity = MODEL_IDENTITIES[domain]
    _validate_file_identity(
        metadata, "backbone_weights", identity["backbone_weights_sha256"]
    )
    _validate_file_identity(
        metadata, "dinotxt_weights", identity["dinotxt_weights_sha256"]
    )
    _validate_file_identity(metadata, "bpe_vocab", identity["bpe_vocab_sha256"])
    counts = metadata.get("trainable_parameters")
    if not isinstance(counts, dict) or counts.get("trainable") != trainable:
        raise ValueError(f"Unexpected M4-B trainable parameter count: {trainable}")


def validate_evidence_identity(
    *, run_dir: Path, gate_dir: Path, stage: int, domain: str
) -> None:
    parity = _read_json(run_dir / "skyscript_step0_parity.json")
    if parity.get("backbone_domain") != domain:
        raise ValueError("M4-B step-0 parity report has the wrong domain")
    _validate_model_identity(parity.get("official_model"), domain, 0)
    _validate_model_identity(
        parity.get("step0_model"), domain, MODEL_IDENTITIES[domain]["trainable_parameters"]
    )
    paths = (
        gate_dir / "skyscript_val_step000.json",
        gate_dir / f"skyscript_val_step{stage:03d}.json",
        gate_dir / "rsicd_val_official.json",
        gate_dir / "rsicd_val_step000.json",
        gate_dir / f"rsicd_val_step{stage:03d}.json",
    )
    for path in paths:
        report = _read_json(path)
        _validate_model_identity(
            report.get("model"),
            domain,
            MODEL_IDENTITIES[domain]["trainable_parameters"],
        )


def summarize_stage(
    *,
    run_dir: Path,
    gate_dir: Path,
    stage: int,
    training_report_path: Path,
    domain: str,
    seed: int,
) -> dict[str, Any]:
    summary = summarize_s2_stage(
        run_dir=run_dir,
        gate_dir=gate_dir,
        stage=stage,
        training_report_path=training_report_path,
    )
    validate_evidence_identity(
        run_dir=run_dir, gate_dir=gate_dir, stage=stage, domain=domain
    )
    summary.update(
        {
            "gate": "SkyScript_M4_B_stage",
            "domain": domain,
            "seed": seed,
            "shared_gate_implementation": "SkyScript_S2_stage",
            "model_identity": {"backbone_domain": domain, **MODEL_IDENTITIES[domain]},
        }
    )
    return summary


def summarize_final(*, gate_dir: Path, domain: str, seed: int) -> dict[str, Any]:
    reports = {
        stage: _read_json(gate_dir / f"stage_{stage}_summary.json") for stage in STAGES
    }
    for stage, report in reports.items():
        checks = report.get("checks")
        if (
            report.get("gate") != "SkyScript_M4_B_stage"
            or report.get("domain") != domain
            or report.get("seed") != seed
            or report.get("stage") != stage
            or not isinstance(checks, dict)
            or not checks
        ):
            raise ValueError(f"Invalid M4-B stage summary for {domain} seed{seed} step{stage}")
        expected_status = "pass" if all(checks.values()) else "fail"
        if report.get("status") != expected_status:
            raise ValueError(f"M4-B stage{stage} status disagrees with its checks")
    commits = {
        report.get("training", {}).get("project_commit") for report in reports.values()
    }
    if len(commits) != 1 or None in commits:
        raise ValueError("M4-B stage reports do not have one initial training commit")
    final = reports[500]
    status = "pass" if all(report["status"] == "pass" for report in reports.values()) else "fail"
    return {
        "format_version": 1,
        "gate": "SkyScript_M4_B_run",
        "status": status,
        "domain": domain,
        "seed": seed,
        "stages": list(STAGES),
        "project_commit": commits.pop(),
        "model_identity": final["model_identity"],
        "stage_checks": {str(stage): reports[stage]["checks"] for stage in STAGES},
        "validation_loss": final["validation_loss"],
        "skyscript": {
            "step0": reports[100]["skyscript"]["step0"],
            "steps": {
                str(stage): reports[stage]["skyscript"][f"step{stage}"]
                for stage in STAGES
            },
            "current_minus_step0_mean_recall": {
                str(stage): reports[stage]["skyscript"][
                    "current_minus_step0_mean_recall"
                ]
                for stage in STAGES
            },
        },
        "rsicd_val": {
            "official": reports[100]["rsicd_val"]["official"],
            "step0": reports[100]["rsicd_val"]["step0"],
            "steps": {
                str(stage): reports[stage]["rsicd_val"][f"step{stage}"]
                for stage in STAGES
            },
            "current_minus_official_mean_recall": {
                str(stage): reports[stage]["rsicd_val"][
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
            domain=args.domain,
            seed=args.seed,
        )
    else:
        summary = summarize_final(
            gate_dir=args.gate_dir, domain=args.domain, seed=args.seed
        )
    _write_atomic_immutable(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
