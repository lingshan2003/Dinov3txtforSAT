#!/usr/bin/env python3
"""Aggregate matched three-seed Web/SAT evidence for SkyScript M4-B."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dinotxt_rs.training.provenance import git_commit, sha256_file  # noqa: E402
from tools.summarize_skyscript_gate_s2 import (  # noqa: E402
    STAGES,
    _read_json,
    _validate_metrics,
    _write_atomic_immutable,
)
from tools.verify_skyscript_m4_b_configs import (  # noqa: E402
    DOMAINS,
    SEEDS,
    config_path,
    load_config_matrix,
    verify_config_matrix,
)

M4_A_SUMMARY_SHA = "69bef90195afca5d19bd79a110f2261c0ece9a281fe6b71672aaa3d93ffaa9dc"
TRAIN_SHA = "4264d884d564631816baf3e339e7ca12be2958966d99a60f6a75ae3d5d4dbaec"
VAL_SHA = "062238716b8fddad890b4f97354588ad7524f1099667d3c0baa932002845b7f6"
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int, choices=STAGES)
    parser.add_argument(
        "--m4-a-summary",
        type=Path,
        default=Path("outputs/skyscript_m4_rq1_seed11/summary.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _difference(left: Any, right: Any, label: str) -> Any:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            raise ValueError(f"Fields differ at {label}")
        return {
            key: _difference(left[key], right[key], f"{label}.{key}") for key in left
        }
    return _finite(left, f"{label}.left") - _finite(right, f"{label}.right")


def _statistics(values: Sequence[Any], label: str) -> Any:
    if not values:
        raise ValueError(f"No values at {label}")
    if all(isinstance(value, Mapping) for value in values):
        keys = values[0].keys()
        if any(value.keys() != keys for value in values[1:]):
            raise ValueError(f"Fields differ at {label}")
        return {
            key: _statistics([value[key] for value in values], f"{label}.{key}")
            for key in keys
        }
    numbers = [_finite(value, label) for value in values]
    return {
        "mean": statistics.fmean(numbers),
        "sample_std": statistics.stdev(numbers),
    }


def stage_report_path(domain: str, seed: int, stage: int) -> Path:
    if seed == 11 and domain == "web":
        gate_dir = Path("outputs/skyscript_gate_s2_seed11")
    elif seed == 11:
        gate_dir = Path("outputs/skyscript_gate_m4_sat_seed11")
    else:
        gate_dir = Path(f"outputs/skyscript_gate_m4_b_{domain}_seed{seed}")
    return gate_dir / f"stage_{stage}_summary.json"


def _validate_stage_report(
    report: dict[str, Any], *, domain: str, seed: int, stage: int
) -> None:
    expected_gate = (
        "SkyScript_S2_stage"
        if (domain, seed) == ("web", 11)
        else "SkyScript_M4_SAT_stage"
        if (domain, seed) == ("sat", 11)
        else "SkyScript_M4_B_stage"
    )
    checks = report.get("checks")
    if (
        report.get("gate") != expected_gate
        or report.get("stage") != stage
        or report.get("status") not in {"pass", "fail"}
        or not isinstance(checks, dict)
        or not checks
    ):
        raise ValueError(f"Invalid {domain} seed{seed} stage{stage} report")
    if seed != 11 and (
        report.get("domain") != domain or report.get("seed") != seed
    ):
        raise ValueError(f"Wrong M4-B identity for {domain} seed{seed}")
    if report["status"] != ("pass" if all(checks.values()) else "fail"):
        raise ValueError(f"Status/check mismatch for {domain} seed{seed}")
    training = report.get("training")
    if (
        not isinstance(training, dict)
        or training.get("train_manifest_sha256") != TRAIN_SHA
        or training.get("val_manifest_sha256") != VAL_SHA
        or training.get("dinov3_commit") != DINOV3_COMMIT
    ):
        raise ValueError(f"Wrong training identity for {domain} seed{seed}")
    _finite(report.get("validation_loss", {}).get("steps", {}).get("0"), "loss0")
    _finite(
        report.get("validation_loss", {}).get("steps", {}).get(str(stage)),
        f"loss{stage}",
    )
    for task, baseline in (("skyscript", "step0"), ("rsicd_val", "official")):
        task_report = report.get(task)
        if not isinstance(task_report, dict):
            raise ValueError(f"Missing {task} for {domain} seed{seed}")
        _validate_metrics(task_report.get(baseline), f"{domain} seed{seed} {baseline}")
        _validate_metrics(
            task_report.get(f"step{stage}"), f"{domain} seed{seed} step{stage}"
        )


def _entry(report: dict[str, Any], task: str, stage: int) -> dict[str, Any]:
    if task == "validation_loss":
        baseline = report[task]["steps"]["0"]
        current = report[task]["steps"][str(stage)]
    else:
        baseline_name = "step0" if task == "skyscript" else "official"
        baseline = report[task][baseline_name]
        current = report[task][f"step{stage}"]
    return {
        "initialization": baseline,
        "absolute": current,
        "within_domain_delta": _difference(current, baseline, f"{task}.delta"),
    }


def _aggregate_task(
    reports: Mapping[tuple[str, int], dict[str, Any]], task: str, stage: int
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    paired: list[dict[str, Any]] = []
    for seed in SEEDS:
        entries = {
            domain: _entry(reports[(domain, seed)], task, stage) for domain in DOMAINS
        }
        difference = {
            field: _difference(
                entries["sat"][field],
                entries["web"][field],
                f"{task}.seed{seed}.{field}.sat_minus_web",
            )
            for field in ("initialization", "absolute", "within_domain_delta")
        }
        per_seed[str(seed)] = {**entries, "sat_minus_web": difference}
        paired.append(difference)
        for domain in DOMAINS:
            by_domain[domain].append(entries[domain])
    return {
        "per_seed": per_seed,
        "domain_statistics": {
            domain: _statistics(values, f"{task}.{domain}")
            for domain, values in by_domain.items()
        },
        "paired_sat_minus_web_statistics": _statistics(
            paired, f"{task}.paired_sat_minus_web"
        ),
    }


def summarize(stage: int, m4_a_summary: Path) -> dict[str, Any]:
    config_dir = Path("configs")
    verify_config_matrix(load_config_matrix(config_dir))
    if sha256_file(m4_a_summary) != M4_A_SUMMARY_SHA:
        raise ValueError("M4-A summary does not match the frozen prerequisite")
    prerequisite = _read_json(m4_a_summary)
    if prerequisite.get("gate") != "SkyScript_M4_RQ1_seed11" or prerequisite.get(
        "status"
    ) != "pass":
        raise ValueError("M4-A prerequisite did not pass")
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    for domain in DOMAINS:
        inputs[domain] = {}
        for seed in SEEDS:
            path = stage_report_path(domain, seed, stage)
            report = _read_json(path)
            _validate_stage_report(report, domain=domain, seed=seed, stage=stage)
            reports[(domain, seed)] = report
            inputs[domain][str(seed)] = {"path": str(path), "sha256": sha256_file(path)}
    status = "pass" if all(report["status"] == "pass" for report in reports.values()) else "fail"
    return {
        "format_version": 1,
        "gate": "SkyScript_M4_B_cross_seed",
        "status": status,
        "stage": stage,
        "domains": list(DOMAINS),
        "seeds": list(SEEDS),
        "scope": "matched three-seed Web/SAT comparison at a pre-registered stage",
        "difference_convention": {
            "between_domain": "sat_minus_web",
            "within_domain": "adapted_step_minus_own_initialization",
            "dispersion": "sample standard deviation (ddof=1)",
            "recall": "positive means higher",
            "rank": "negative means better",
        },
        "reporting_project_commit": git_commit(Path.cwd()),
        "m4_a_prerequisite": {
            "path": str(m4_a_summary),
            "sha256": M4_A_SUMMARY_SHA,
        },
        "inputs": inputs,
        "configs": {
            domain: {
                str(seed): {
                    "path": str(config_path(config_dir, domain, seed)),
                    "sha256": sha256_file(config_path(config_dir, domain, seed)),
                }
                for seed in SEEDS
            }
            for domain in DOMAINS
        },
        "run_status": {
            domain: {str(seed): reports[(domain, seed)]["status"] for seed in SEEDS}
            for domain in DOMAINS
        },
        "validation_loss": _aggregate_task(reports, "validation_loss", stage),
        "skyscript": _aggregate_task(reports, "skyscript", stage),
        "rsicd_val": _aggregate_task(reports, "rsicd_val", stage),
    }


def main() -> None:
    args = parse_args()
    summary = summarize(args.stage, args.m4_a_summary)
    _write_atomic_immutable(args.output, summary)
    print(f"m4_b_summary={args.output} status={summary['status']}")
    raise SystemExit(0 if summary["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
