#!/usr/bin/env python3
"""Gate the SAT pilot on the already-reviewed Web 500-step pilot result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the Web 500-step pilot improved validation"
    )
    parser.add_argument("--report", required=True, type=Path, help="Web recovery_report.json")
    parser.add_argument(
        "--allow-degraded-artifacts",
        action="store_true",
        help="Permit the known missing resume checkpoint, while reporting it explicitly",
    )
    return parser.parse_args()


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


def check_web_pilot_gate(
    report_path: Path, *, allow_degraded_artifacts: bool = False
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Recovery report must be a JSON object")
    summary = report.get("training_summary")
    validation = report.get("validation")
    consistency = report.get("best_checkpoint_consistency")
    if not isinstance(summary, dict) or not isinstance(validation, dict):
        raise ValueError("Recovery report lacks training summary or validation evidence")
    if not isinstance(consistency, dict):
        raise ValueError("Recovery report lacks best checkpoint consistency evidence")
    if summary.get("steps") != 500 or summary.get("target_steps") != 5000:
        raise ValueError("Web gate requires the 500-step cap on the 5,000-step formal schedule")
    if summary.get("completed") is not False:
        raise ValueError("Web gate must be an intentionally incomplete pilot")
    improvement = _finite(
        validation.get("best_relative_change_from_step_zero"),
        "validation best_relative_change_from_step_zero",
    )
    if improvement >= 0:
        raise ValueError("Web validation did not improve on step 0")
    for field in (
        "best_step_matches_validation_records",
        "best_loss_matches_validation_records",
        "best_file_matches_summary_best_step",
    ):
        if consistency.get(field) is not True:
            raise ValueError(f"Web best checkpoint evidence failed: {field}")

    evidence_status = report.get("artifact_evidence_status")
    if evidence_status not in {"complete", "degraded"}:
        raise ValueError(f"Unexpected Web artifact evidence status: {evidence_status!r}")
    if evidence_status == "degraded" and not allow_degraded_artifacts:
        raise ValueError(
            "Web resume artifacts are degraded; pass --allow-degraded-artifacts only after review"
        )

    return {
        "format_version": 1,
        "web_recovery_report": str(report_path.resolve()),
        "web_artifact_evidence_status": evidence_status,
        "web_validation_best_relative_change_from_step_zero": improvement,
        "web_validation_best_step": validation.get("best_step_from_records"),
        "decision": "proceed_to_sat_pilot",
        "warning": (
            "The Web resume checkpoint at step 250 is unavailable; this gate permits only the "
            "SAT pilot, not a formal 5,000-step launch."
            if evidence_status == "degraded"
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            check_web_pilot_gate(
                args.report, allow_degraded_artifacts=args.allow_degraded_artifacts
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
