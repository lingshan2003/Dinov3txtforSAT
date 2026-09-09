#!/usr/bin/env python3
"""Build the matched-seed Web/SAT comparison report for SkyScript M4-A."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
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
from tools.verify_skyscript_m4_config_pair import (  # noqa: E402
    load_toml,
    verify_config_pair,
)

TRAIN_MANIFEST_SHA256 = "4264d884d564631816baf3e339e7ca12be2958966d99a60f6a75ae3d5d4dbaec"
VAL_MANIFEST_SHA256 = "062238716b8fddad890b4f97354588ad7524f1099667d3c0baa932002845b7f6"
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
WEB_BACKBONE_SHA256 = "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
SAT_BACKBONE_SHA256 = "eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48"
DINOTXT_SHA256 = "a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0"
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
ADAPTER_TRAINABLE_PARAMETERS = 1_054_976

EXPECTED_MODELS = {
    "web": WEB_BACKBONE_SHA256,
    "sat": SAT_BACKBONE_SHA256,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-summary", required=True, type=Path)
    parser.add_argument("--sat-summary", required=True, type=Path)
    parser.add_argument("--web-config", required=True, type=Path)
    parser.add_argument("--sat-config", required=True, type=Path)
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
    """Return left-minus-right while preserving a nested numeric mapping."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            raise ValueError(f"Metric fields differ at {label}")
        return {
            key: _difference(left[key], right[key], f"{label}.{key}") for key in left
        }
    return _finite(left, f"{label} left") - _finite(right, f"{label} right")


def _validate_summary(summary: dict[str, Any], domain: str) -> None:
    expected_gate = "SkyScript_S2" if domain == "web" else "SkyScript_M4_SAT"
    if (
        summary.get("gate") != expected_gate
        or summary.get("status") != "pass"
        or summary.get("stages") != list(STAGES)
    ):
        raise ValueError(f"{domain} summary is not a passing 100/250/500 report")
    if domain == "sat" and summary.get("domain") != "sat":
        raise ValueError("SAT summary has the wrong domain")
    validation = summary.get("validation_loss", {}).get("steps")
    if not isinstance(validation, dict) or sorted(map(int, validation)) != list(range(0, 501, 50)):
        raise ValueError(f"{domain} summary has an incomplete validation curve")
    if summary.get("validation_loss", {}).get("best_step") != 500:
        raise ValueError(f"{domain} summary best validation checkpoint is not step500")
    for task in ("skyscript", "rsicd_val"):
        task_report = summary.get(task)
        if not isinstance(task_report, dict):
            raise ValueError(f"{domain} summary has no {task} report")
        baseline_name = "step0" if task == "skyscript" else "official"
        _validate_metrics(task_report.get(baseline_name), f"{domain} {task} baseline")
        steps = task_report.get("steps")
        if not isinstance(steps, dict) or set(steps) != {str(stage) for stage in STAGES}:
            raise ValueError(f"{domain} summary has incomplete {task} stages")
        for stage in STAGES:
            _validate_metrics(steps[str(stage)], f"{domain} {task} step{stage}")


def _validate_training_identity(summary: dict[str, Any], domain: str) -> dict[str, Any]:
    stage_paths = summary.get("stage_reports")
    if not isinstance(stage_paths, dict) or set(stage_paths) != {str(stage) for stage in STAGES}:
        raise ValueError(f"{domain} summary has invalid stage report paths")
    reports = {stage: _read_json(Path(stage_paths[str(stage)])) for stage in STAGES}
    expected_gate = "SkyScript_S2_stage" if domain == "web" else "SkyScript_M4_SAT_stage"
    for stage, report in reports.items():
        training = report.get("training")
        if (
            report.get("gate") != expected_gate
            or report.get("stage") != stage
            or report.get("status") != "pass"
            or not isinstance(training, dict)
            or training.get("train_manifest_sha256") != TRAIN_MANIFEST_SHA256
            or training.get("val_manifest_sha256") != VAL_MANIFEST_SHA256
            or training.get("dinov3_commit") != DINOV3_COMMIT
        ):
            raise ValueError(f"{domain} stage{stage} training identity is invalid")
    commits = {report["training"].get("project_commit") for report in reports.values()}
    if len(commits) != 1 or None in commits:
        raise ValueError(f"{domain} stages do not have one initial training commit")
    return {
        "initial_training_project_commit": commits.pop(),
        "dinov3_commit": DINOV3_COMMIT,
        "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
        "val_manifest_sha256": VAL_MANIFEST_SHA256,
    }


def _validate_model_metadata(metadata: Any, domain: str, label: str) -> None:
    if not isinstance(metadata, dict) or metadata.get("backbone_domain") != domain:
        raise ValueError(f"{label} has the wrong backbone domain")
    expected_files = {
        "backbone_weights": EXPECTED_MODELS[domain],
        "dinotxt_weights": DINOTXT_SHA256,
        "bpe_vocab": BPE_SHA256,
    }
    for name, expected_sha in expected_files.items():
        if metadata.get(name, {}).get("sha256") != expected_sha:
            raise ValueError(f"{label} has the wrong {name} identity")
    if metadata.get("trainable_parameters", {}).get("trainable") != ADAPTER_TRAINABLE_PARAMETERS:
        raise ValueError(f"{label} has the wrong trainable parameter count")


def _validate_evaluation_identities(
    summary_path: Path, summary: dict[str, Any], domain: str
) -> dict[str, Any]:
    gate_dir = summary_path.parent
    report_metrics = {
        gate_dir / "skyscript_val_step000.json": (summary["skyscript"]["step0"], 0),
        gate_dir / "rsicd_val_official.json": (summary["rsicd_val"]["official"], None),
        gate_dir / "rsicd_val_step000.json": (summary["rsicd_val"]["step0"], 0),
    }
    for stage in STAGES:
        report_metrics[gate_dir / f"skyscript_val_step{stage:03d}.json"] = (
            summary["skyscript"]["steps"][str(stage)],
            stage,
        )
        report_metrics[gate_dir / f"rsicd_val_step{stage:03d}.json"] = (
            summary["rsicd_val"]["steps"][str(stage)],
            stage,
        )
    checkpoint_commits: dict[str, str | None] = {}
    for path, (expected_metrics, expected_step) in report_metrics.items():
        report = _read_json(path)
        model = report.get("model")
        _validate_model_metadata(model, domain, str(path))
        if report.get("metrics") != expected_metrics:
            raise ValueError(f"{path} metrics do not match the domain summary")
        checkpoint = model.get("checkpoint")
        if expected_step is None:
            if checkpoint is not None:
                raise ValueError(f"{path} should describe official initialization")
            continue
        if not isinstance(checkpoint, dict) or checkpoint.get("step") != expected_step:
            raise ValueError(f"{path} has the wrong checkpoint step")
        project_commit = checkpoint.get("run_identity", {}).get("project_commit")
        if not isinstance(project_commit, str) or not project_commit:
            raise ValueError(f"{path} has no checkpoint project commit")
        step_key = str(expected_step)
        previous = checkpoint_commits.setdefault(step_key, project_commit)
        if previous != project_commit:
            raise ValueError(f"{domain} step{expected_step} evaluations use different commits")
    return {
        "backbone_domain": domain,
        "preprocessing_domain": domain,
        "backbone_weights_sha256": EXPECTED_MODELS[domain],
        "dinotxt_weights_sha256": DINOTXT_SHA256,
        "bpe_vocab_sha256": BPE_SHA256,
        "trainable_parameters": ADAPTER_TRAINABLE_PARAMETERS,
        "checkpoint_project_commits": checkpoint_commits,
    }


def _task_comparison(
    web_task: dict[str, Any], sat_task: dict[str, Any], *, baseline_name: str
) -> dict[str, Any]:
    web_baseline = web_task[baseline_name]
    sat_baseline = sat_task[baseline_name]
    result: dict[str, Any] = {
        "baseline": {
            "web": web_baseline,
            "sat": sat_baseline,
            "sat_minus_web": _difference(sat_baseline, web_baseline, "baseline"),
        },
        "steps": {},
    }
    for stage in STAGES:
        key = str(stage)
        web_metrics = web_task["steps"][key]
        sat_metrics = sat_task["steps"][key]
        web_delta = _difference(web_metrics, web_baseline, f"web.step{stage}.delta")
        sat_delta = _difference(sat_metrics, sat_baseline, f"sat.step{stage}.delta")
        result["steps"][key] = {
            "absolute": {
                "web": web_metrics,
                "sat": sat_metrics,
                "sat_minus_web": _difference(
                    sat_metrics, web_metrics, f"step{stage}.sat_minus_web"
                ),
            },
            "within_domain_delta": {"web": web_delta, "sat": sat_delta},
            "sat_delta_minus_web_delta": _difference(
                sat_delta, web_delta, f"step{stage}.delta_difference"
            ),
        }
    return result


def summarize_m4_a(
    *, web_summary_path: Path, sat_summary_path: Path, web_config: Path, sat_config: Path
) -> dict[str, Any]:
    verify_config_pair(load_toml(web_config), load_toml(sat_config))
    web = _read_json(web_summary_path)
    sat = _read_json(sat_summary_path)
    _validate_summary(web, "web")
    _validate_summary(sat, "sat")

    web_training = _validate_training_identity(web, "web")
    sat_training = _validate_training_identity(sat, "sat")
    if (
        web_training["train_manifest_sha256"] != sat_training["train_manifest_sha256"]
        or web_training["val_manifest_sha256"] != sat_training["val_manifest_sha256"]
    ):
        raise ValueError("Web and SAT summaries do not use the same manifests")

    web_model = _validate_evaluation_identities(web_summary_path, web, "web")
    sat_model = _validate_evaluation_identities(sat_summary_path, sat, "sat")
    web_loss0 = _finite(web["validation_loss"]["steps"]["0"], "web loss 0")
    sat_loss0 = _finite(sat["validation_loss"]["steps"]["0"], "sat loss 0")
    validation = {}
    for step in range(0, 501, 50):
        web_loss = _finite(web["validation_loss"]["steps"][str(step)], f"web loss {step}")
        sat_loss = _finite(sat["validation_loss"]["steps"][str(step)], f"sat loss {step}")
        web_delta = web_loss - web_loss0
        sat_delta = sat_loss - sat_loss0
        validation[str(step)] = {
            "absolute": {"web": web_loss, "sat": sat_loss, "sat_minus_web": sat_loss - web_loss},
            "within_domain_delta": {"web": web_delta, "sat": sat_delta},
            "sat_delta_minus_web_delta": sat_delta - web_delta,
        }
    skyscript = _task_comparison(
        web["skyscript"], sat["skyscript"], baseline_name="step0"
    )
    rsicd = _task_comparison(
        web["rsicd_val"], sat["rsicd_val"], baseline_name="official"
    )
    headline = {}
    for name, comparison in (("skyscript", skyscript), ("rsicd_val", rsicd)):
        step500 = comparison["steps"]["500"]
        headline[name] = {
            "web_mean_recall": step500["absolute"]["web"]["mean_recall"],
            "sat_mean_recall": step500["absolute"]["sat"]["mean_recall"],
            "sat_minus_web_mean_recall": step500["absolute"]["sat_minus_web"][
                "mean_recall"
            ],
            "web_within_domain_delta": step500["within_domain_delta"]["web"][
                "mean_recall"
            ],
            "sat_within_domain_delta": step500["within_domain_delta"]["sat"][
                "mean_recall"
            ],
            "sat_delta_minus_web_delta": step500["sat_delta_minus_web_delta"][
                "mean_recall"
            ],
        }
    return {
        "format_version": 1,
        "gate": "SkyScript_M4_RQ1_seed11",
        "status": "pass",
        "scope": "matched_seed11_pilot; not a cross-seed RQ1 conclusion",
        "stages": list(STAGES),
        "difference_convention": {
            "between_domain": "sat_minus_web",
            "within_domain": "adapted_step_minus_own_initialization",
            "recall": "positive means higher",
            "rank": "negative means better",
        },
        "inputs": {
            "web_summary": {
                "path": str(web_summary_path),
                "sha256": sha256_file(web_summary_path),
            },
            "sat_summary": {
                "path": str(sat_summary_path),
                "sha256": sha256_file(sat_summary_path),
            },
            "web_config": {"path": str(web_config), "sha256": sha256_file(web_config)},
            "sat_config": {"path": str(sat_config), "sha256": sha256_file(sat_config)},
        },
        "reporting_project_commit": git_commit(Path.cwd()),
        "protocol": {
            "seed": 11,
            "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
            "val_manifest_sha256": VAL_MANIFEST_SHA256,
            "matched_config_check": "pass",
        },
        "domains": {
            "web": {"model": web_model, "training": web_training},
            "sat": {"model": sat_model, "training": sat_training},
        },
        "headline_step500": headline,
        "validation_loss": validation,
        "skyscript": skyscript,
        "rsicd_val": rsicd,
    }


def main() -> None:
    args = parse_args()
    summary = summarize_m4_a(
        web_summary_path=args.web_summary,
        sat_summary_path=args.sat_summary,
        web_config=args.web_config,
        sat_config=args.sat_config,
    )
    _write_atomic_immutable(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
