#!/usr/bin/env python3
"""Validate and aggregate the three SkyScript Gate S1 seed results."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tomllib
from pathlib import Path
from typing import Any

EXPECTED_SEEDS = (11, 23, 47)
EXPECTED_STEPS = (0, 25, 50, 75, 100)
RETENTION_MARGIN = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate seed 11/23/47 Gate reports and write the Gate S1 summary"
    )
    parser.add_argument(
        "--seed-result",
        action="append",
        nargs=3,
        metavar=("SEED", "RUN_DIR", "GATE_SUMMARY"),
        required=True,
        help="Repeat exactly once for seeds 11, 23, and 47",
    )
    parser.add_argument("--output", required=True, type=Path)
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


def _normalized_protocol(config: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(config))
    experiment = normalized.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Config has no [experiment] table")
    for field in ("name", "seed", "output_dir"):
        if field not in experiment:
            raise ValueError(f"Config experiment has no {field!r}")
        del experiment[field]
    return normalized


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
    }


def _aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = records[0].keys()
    if any(record.keys() != keys for record in records[1:]):
        raise ValueError("Metric records do not have identical fields across seeds")
    result: dict[str, Any] = {}
    for key in keys:
        values = [record[key] for record in records]
        if all(isinstance(value, dict) for value in values):
            result[key] = _aggregate_metrics(values)
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        ):
            result[key] = _mean_std(
                [_finite(value, f"retrieval metric {key}") for value in values]
            )
        else:
            raise ValueError(f"Unsupported or inconsistent metric field: {key}")
    return result


def _write_atomic_immutable(destination: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"Refusing to overwrite a different existing S1 summary: {destination}"
            )
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"Incomplete prior S1 summary exists: {temporary}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, destination)


def _load_seed(seed: int, run_dir: Path, gate_path: Path) -> dict[str, Any]:
    config_path = run_dir / "config.toml"
    validation_path = run_dir / "validation.jsonl"
    verification_path = run_dir / "verification_report.json"
    for path in (config_path, validation_path, verification_path, gate_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Gate S1 input: {path}")

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    experiment = config.get("experiment", {})
    if experiment.get("seed") != seed:
        raise ValueError(
            f"Seed {seed} config identity mismatch: experiment.seed={experiment.get('seed')!r}"
        )

    validation_records = _read_jsonl(validation_path)
    steps = [record.get("step") for record in validation_records]
    if steps != list(EXPECTED_STEPS):
        raise ValueError(f"Seed {seed} validation steps are {steps}, expected {EXPECTED_STEPS}")
    validation = {
        str(step): _finite(record.get("loss"), f"seed {seed} validation step {step}")
        for step, record in zip(EXPECTED_STEPS, validation_records, strict=True)
    }
    best_step = min(EXPECTED_STEPS, key=lambda step: validation[str(step)])

    verification = _read_json(verification_path)
    if verification.get("completed") is not True or verification.get("steps") != 100:
        raise ValueError(f"Seed {seed} training verification is not complete at step 100")
    if verification.get("best_validation_step") != best_step:
        raise ValueError(f"Seed {seed} best validation step disagrees with validation.jsonl")

    gate = _read_json(gate_path)
    if gate.get("status") not in {"pass", "fail"}:
        raise ValueError(f"Seed {seed} Gate report has no valid status: {gate_path}")
    checks = gate.get("checks")
    integrity_checks = (
        "training_artifacts",
        "train_validation_overlap",
        "step0_parity",
        "training_rsicd_val_overlap",
    )
    if not isinstance(checks, dict) or any(
        checks.get(name) is not True for name in integrity_checks
    ):
        raise ValueError(f"Seed {seed} Gate integrity checks are incomplete or failed")

    sky = gate.get("skyscript", {})
    rsicd = gate.get("rsicd_val", {})
    sky_steps = sky.get("steps")
    rsicd_steps = rsicd.get("steps")
    if not isinstance(sky_steps, dict) or set(sky_steps) != {str(v) for v in EXPECTED_STEPS}:
        raise ValueError(f"Seed {seed} SkyScript retrieval steps are incomplete")
    if not isinstance(rsicd_steps, dict) or set(rsicd_steps) != {str(v) for v in EXPECTED_STEPS}:
        raise ValueError(f"Seed {seed} RSICD retrieval steps are incomplete")
    if not isinstance(rsicd.get("official"), dict):
        raise ValueError(f"Seed {seed} RSICD official metrics are missing")

    sky_delta = _finite(
        sky_steps["100"]["mean_recall"], f"seed {seed} SkyScript step100 mean recall"
    ) - _finite(sky_steps["0"]["mean_recall"], f"seed {seed} SkyScript step0 mean recall")
    rsicd_delta = _finite(
        rsicd_steps["100"]["mean_recall"], f"seed {seed} RSICD step100 mean recall"
    ) - _finite(rsicd["official"]["mean_recall"], f"seed {seed} official mean recall")
    criteria = {
        "validation_step100_below_step0": validation["100"] < validation["0"],
        "validation_best_not_step0": best_step != 0,
        "skyscript_step100_mean_recall_above_step0": sky_delta > 0,
        "rsicd_step100_mean_recall_drop_within_0.01": rsicd_delta >= -RETENTION_MARGIN,
    }
    expected_gate_status = "pass" if all(checks.values()) else "fail"
    if gate["status"] != expected_gate_status:
        raise ValueError(f"Seed {seed} Gate status disagrees with its checks")
    for name in (
        "skyscript_step100_mean_recall_above_step0",
        "rsicd_step100_mean_recall_drop_within_0.01",
    ):
        if checks.get(name) is not criteria[name]:
            raise ValueError(f"Seed {seed} Gate check {name!r} disagrees with its metrics")
    validation_check = "training_validation_step100_below_step0_and_best_not_step0"
    if validation_check in checks and checks[validation_check] is not (
        criteria["validation_step100_below_step0"]
        and criteria["validation_best_not_step0"]
    ):
        raise ValueError(f"Seed {seed} Gate validation check disagrees with validation.jsonl")
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "gate_summary": str(gate_path),
        "project_commit": verification.get("project_commit"),
        "config": config,
        "protocol": _normalized_protocol(config),
        "status": "pass" if all(criteria.values()) else "fail",
        "criteria": criteria,
        "validation_loss": {"steps": validation, "best_step": best_step},
        "skyscript": {
            "steps": sky_steps,
            "step100_minus_step0_mean_recall": sky_delta,
        },
        "rsicd_val": {
            "official": rsicd["official"],
            "steps": rsicd_steps,
            "step100_minus_official_mean_recall": rsicd_delta,
        },
    }


def summarize_seed_results(specs: list[tuple[int, Path, Path]]) -> dict[str, Any]:
    observed = tuple(sorted(seed for seed, _, _ in specs))
    if observed != EXPECTED_SEEDS or len(specs) != len(EXPECTED_SEEDS):
        raise ValueError(f"Expected exactly seeds {EXPECTED_SEEDS}, got {observed}")
    per_seed = {seed: _load_seed(seed, run_dir, gate) for seed, run_dir, gate in specs}
    reference_protocol = per_seed[EXPECTED_SEEDS[0]]["protocol"]
    for seed in EXPECTED_SEEDS[1:]:
        if per_seed[seed]["protocol"] != reference_protocol:
            raise ValueError(
                f"Seed {seed} config changes fields outside experiment.name/seed/output_dir"
            )

    aggregate_validation = {
        str(step): _mean_std(
            [per_seed[seed]["validation_loss"]["steps"][str(step)] for seed in EXPECTED_SEEDS]
        )
        for step in EXPECTED_STEPS
    }
    aggregate_sky = {
        str(step): _aggregate_metrics(
            [per_seed[seed]["skyscript"]["steps"][str(step)] for seed in EXPECTED_SEEDS]
        )
        for step in EXPECTED_STEPS
    }
    aggregate_rsicd = {
        str(step): _aggregate_metrics(
            [per_seed[seed]["rsicd_val"]["steps"][str(step)] for seed in EXPECTED_SEEDS]
        )
        for step in EXPECTED_STEPS
    }

    status = (
        "pass"
        if all(per_seed[seed]["status"] == "pass" for seed in EXPECTED_SEEDS)
        else "fail"
    )
    return {
        "format_version": 1,
        "gate": "SkyScript_S1",
        "status": status,
        "seeds": list(EXPECTED_SEEDS),
        "dispersion": "sample standard deviation (ddof=1)",
        "criteria": {
            "per_seed_validation": "step100 loss < step0 loss and best step != 0",
            "per_seed_skyscript": "step100 mean_recall > step0 mean_recall",
            "per_seed_rsicd_retention": (
                "step100 mean_recall - official mean_recall >= -0.01"
            ),
            "config_identity": (
                "all fields identical except experiment.name, experiment.seed, "
                "experiment.output_dir"
            ),
        },
        "per_seed": {
            str(seed): {key: value for key, value in per_seed[seed].items() if key != "protocol"}
            for seed in EXPECTED_SEEDS
        },
        "aggregate": {
            "validation_loss_by_step": aggregate_validation,
            "skyscript_metrics_by_step": aggregate_sky,
            "skyscript_step100_minus_step0_mean_recall": _mean_std(
                [
                    per_seed[seed]["skyscript"]["step100_minus_step0_mean_recall"]
                    for seed in EXPECTED_SEEDS
                ]
            ),
            "rsicd_val_official_metrics": _aggregate_metrics(
                [per_seed[seed]["rsicd_val"]["official"] for seed in EXPECTED_SEEDS]
            ),
            "rsicd_val_metrics_by_step": aggregate_rsicd,
            "rsicd_step100_minus_official_mean_recall": _mean_std(
                [
                    per_seed[seed]["rsicd_val"]["step100_minus_official_mean_recall"]
                    for seed in EXPECTED_SEEDS
                ]
            ),
        },
    }


def main() -> None:
    args = parse_args()
    specs = [(int(seed), Path(run_dir), Path(gate)) for seed, run_dir, gate in args.seed_result]
    summary = summarize_seed_results(specs)
    _write_atomic_immutable(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
