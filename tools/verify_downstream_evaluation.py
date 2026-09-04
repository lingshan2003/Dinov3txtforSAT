#!/usr/bin/env python3
"""Verify that the four downstream pilot reports are complete and comparable."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

LABELS = ("web_official", "sat_official", "web_500step_best", "sat_500step_best")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify EuroSAT and RSICD downstream pilot reports"
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
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


def verify_downstream_evaluation(output: Path) -> dict[str, Any]:
    output = output.resolve()
    eurosat_manifest_hashes: set[str] = set()
    rsicd_manifest_hashes: set[str] = set()
    summary: dict[str, Any] = {}
    for label in LABELS:
        eurosat_path = output / f"eurosat_{label}.json"
        rsicd_path = output / f"rsicd_{label}.json"
        if not eurosat_path.is_file() or not rsicd_path.is_file():
            raise FileNotFoundError(f"Missing downstream report(s) for {label}")
        eurosat = _read(eurosat_path)
        rsicd = _read(rsicd_path)
        if eurosat.get("task") != "eurosat_zero_shot_classification":
            raise ValueError(f"Unexpected EuroSAT task in {eurosat_path}")
        if rsicd.get("task") != "rsicd_image_text_retrieval" or rsicd.get("split") != "test":
            raise ValueError(f"Unexpected RSICD task or split in {rsicd_path}")
        eurosat_manifest = eurosat.get("manifest", {})
        rsicd_manifest = rsicd.get("manifest", {})
        eurosat_hash = eurosat_manifest.get("sha256")
        rsicd_hash = rsicd_manifest.get("sha256")
        if not isinstance(eurosat_hash, str) or not isinstance(rsicd_hash, str):
            raise ValueError(f"Downstream report has no manifest hash for {label}")
        eurosat_manifest_hashes.add(eurosat_hash)
        rsicd_manifest_hashes.add(rsicd_hash)
        euro_metrics = eurosat.get("metrics")
        rsicd_metrics = rsicd.get("metrics")
        if not isinstance(euro_metrics, dict) or not isinstance(rsicd_metrics, dict):
            raise ValueError(f"Downstream report has no metrics for {label}")
        top1 = _finite(euro_metrics.get("top1_accuracy"), f"{label} EuroSAT top1")
        mean_per_class = _finite(
            euro_metrics.get("mean_per_class_accuracy"), f"{label} EuroSAT mean per-class"
        )
        if not 0 <= top1 <= 1 or not 0 <= mean_per_class <= 1:
            raise ValueError(f"EuroSAT accuracy outside [0, 1] for {label}")
        retrieval_summary: dict[str, Any] = {}
        for direction in ("image_to_text", "text_to_image"):
            direction_metrics = rsicd_metrics.get(direction)
            if not isinstance(direction_metrics, dict):
                raise ValueError(f"RSICD report has no {direction} metrics for {label}")
            r1 = _finite(direction_metrics.get("r1"), f"{label} {direction} r1")
            r5 = _finite(direction_metrics.get("r5"), f"{label} {direction} r5")
            r10 = _finite(direction_metrics.get("r10"), f"{label} {direction} r10")
            median_rank = _finite(
                direction_metrics.get("median_rank"), f"{label} {direction} median rank"
            )
            if not 0 <= r1 <= r5 <= r10 <= 1 or median_rank < 1:
                raise ValueError(f"Invalid RSICD retrieval ordering for {label} {direction}")
            retrieval_summary[direction] = {"r1": r1, "r5": r5, "r10": r10}
        checkpoint = eurosat.get("model", {}).get("checkpoint")
        if label.endswith("official") and checkpoint is not None:
            raise ValueError(f"Official baseline unexpectedly used a checkpoint: {label}")
        if label.endswith("best") and not isinstance(checkpoint, dict):
            raise ValueError(f"Fine-tuned evaluation has no checkpoint provenance: {label}")
        summary[label] = {
            "eurosat_top1_accuracy": top1,
            "eurosat_mean_per_class_accuracy": mean_per_class,
            "rsicd": retrieval_summary,
            "checkpoint_step": None if checkpoint is None else checkpoint.get("step"),
        }
    if len(eurosat_manifest_hashes) != 1 or len(rsicd_manifest_hashes) != 1:
        raise ValueError("All models must use the same EuroSAT and RSICD manifests")
    return {
        "format_version": 1,
        "output_dir": str(output),
        "models": summary,
        "eurosat_manifest_sha256": next(iter(eurosat_manifest_hashes)),
        "rsicd_manifest_sha256": next(iter(rsicd_manifest_hashes)),
        "status": "complete",
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(verify_downstream_evaluation(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
