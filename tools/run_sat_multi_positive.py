#!/usr/bin/env python3
"""Train three group-data ablations and compare four models on fixed retrieval pools."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dinotxt_rs.config import load_config, required_paths
from dinotxt_rs.evaluation.retrieval import (
    RETRIEVAL_TIE_POLICY,
    load_caption_group_records,
    load_paired_records,
    load_rsicd_records,
)

TARGET_STEPS = 1710
METHODS = ("rotate", "multipos", "maskpos")
BASELINE = "skyscript_sat_adapter_textlora_3epoch_seed11"
COMPARISON_DIR = Path("outputs/sat_multi_positive_seed11")
GROUP_DIR = Path("assets/data/manifests/skyscript_caption_groups_v1")
TRAIN_REFERENCE = Path(
    "assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl"
)
MANIFESTS = {
    "skyscript_unique": Path(
        "assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
    ),
    "skyscript_group": GROUP_DIR / "val_grouped.jsonl",
    "rsicd": Path("assets/data/manifests/rsicd_val_retrieval_v1.jsonl"),
}


@dataclass(frozen=True)
class Trial:
    method: str
    config: Path
    output: Path


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def training_state(trial: Trial) -> str:
    """Determine safe continuation without inspecting or hashing checkpoint payloads."""
    output = trial.output
    if output.exists() and not output.is_dir():
        raise ValueError(f"Training output is not a directory: {output}")
    snapshot = output / "config.toml"
    if snapshot.exists() and snapshot.read_bytes() != trial.config.read_bytes():
        raise ValueError(f"Saved configuration differs: {snapshot}")
    summary_path = output / "training_summary.json"
    summary = _read(summary_path) if summary_path.exists() else {}
    complete = summary.get("completed") is True
    if complete:
        if summary.get("steps") != TARGET_STEPS or summary.get("target_steps") != TARGET_STEPS:
            raise ValueError(f"Unexpected completed training budget: {output}")
        validation = summary.get("validation", {})
        if validation.get("last_step") != TARGET_STEPS:
            raise ValueError(f"Missing terminal validation: {output}")
        best_step = validation.get("best_step")
        if type(best_step) is not int or not 0 <= best_step <= TARGET_STEPS:
            raise ValueError(f"Invalid best checkpoint step: {output}")
        for name in ("step_0000000.pt", "best.pt", "latest.pt"):
            if not (output / name).is_file():
                raise FileNotFoundError(output / name)
    resume = (output / "latest.pt").is_file()
    if complete or resume:
        for name in ("config.toml", "provenance.json"):
            if not (output / name).is_file():
                raise FileNotFoundError(output / name)
        return "complete" if complete else "resume"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Existing nonempty run has no latest.pt: {output}; inspect it manually")
    return "fresh"


def _recalls(metrics: dict[str, Any], label: str) -> list[float]:
    values = [metrics[key] for key in ("r1", "r5", "r10")]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError(f"Invalid Recall values: {label}")
    if not 0 <= values[0] <= values[1] <= values[2] <= 1:
        raise ValueError(f"Invalid Recall ordering: {label}")
    return values


def validate_report(
    report: Path, trial: Trial, dataset: str, tag: str, manifest: Path, counts: dict[str, int]
) -> None:
    value = _read(report)
    summary = _read(trial.output / "training_summary.json")
    expected_step = 0 if tag == "step_0000000" else summary["validation"]["best_step"]
    task = {
        "skyscript_unique": "paired_image_text_global_retrieval",
        "skyscript_group": "caption_group_image_text_global_retrieval",
        "rsicd": "rsicd_image_text_retrieval",
    }[dataset]
    if value.get("split") != "val" or value.get("task") != task:
        raise ValueError(f"Unexpected retrieval task/split: {report}")
    if any(value["counts"].get(key) != count for key, count in counts.items()):
        raise ValueError(f"Unexpected retrieval candidate counts: {report}")
    if Path(value["manifest"]["path"]).resolve() != manifest.resolve():
        raise ValueError(f"Unexpected retrieval manifest: {report}")
    model = value["model"]
    checkpoint = model["checkpoint"]
    if checkpoint["step"] != expected_step:
        raise ValueError(f"Unexpected checkpoint step: {report}")
    if Path(checkpoint["path"]).resolve() != (trial.output / f"{tag}.pt").resolve():
        raise ValueError(f"Unexpected checkpoint path: {report}")
    if Path(model["model_config"]).resolve() != (trial.output / "config.toml").resolve():
        raise ValueError(f"Unexpected model configuration: {report}")
    identity = checkpoint["identity_check"]
    if identity["status"] not in ("match", "warning") or identity["blocking_mismatches"]:
        raise ValueError(f"Checkpoint identity mismatch: {report}")
    if value.get("tie_policy") != RETRIEVAL_TIE_POLICY:
        raise ValueError(f"Unexpected retrieval tie policy: {report}")
    metrics = value["metrics"]
    recalls = []
    for direction in ("image_to_text", "text_to_image"):
        recalls.extend(_recalls(metrics[direction], str(report)))
        for key in ("mean_rank", "median_rank"):
            rank = metrics[direction][key]
            if type(rank) not in (int, float) or not math.isfinite(rank) or rank < 1:
                raise ValueError(f"Invalid retrieval rank: {report}")
    mean = metrics["mean_recall"]
    if (
        type(mean) not in (int, float)
        or not math.isfinite(mean)
        or not math.isclose(mean, sum(recalls) / 6, abs_tol=1e-12)
    ):
        raise ValueError(f"Invalid mean Recall: {report}")
    if dataset == "skyscript_unique":
        if value.get("positive_definition") != "manifest_row_one_to_one":
            raise ValueError(f"Unexpected positive definition: {report}")
    elif dataset == "skyscript_group":
        if (
            value.get("positive_definition") != "normalized_complete_caption_group"
            or value.get("recall_definition") != "fraction_of_queries_with_any_positive_in_top_k"
        ):
            raise ValueError(f"Unexpected group positive/Recall definition: {report}")
        balanced = _recalls(metrics["image_to_text_group_balanced"], str(report))
        balanced_mean = metrics["group_balanced_mean_recall"]
        if type(balanced_mean) not in (int, float) or not math.isfinite(balanced_mean):
            raise ValueError(f"Invalid group-balanced mean Recall: {report}")
        if not math.isclose(balanced_mean, (sum(balanced) + sum(recalls[3:])) / 6, abs_tol=1e-12):
            raise ValueError(f"Invalid group-balanced mean Recall: {report}")


def _pool_counts(
    root: Path, *, training_images: set[Path] | None = None, training_ids: set[str] | None = None
) -> dict[str, dict[str, int]]:
    unique = load_paired_records(
        root / MANIFESTS["skyscript_unique"], expected_split="val", expected_source="SkyScript"
    )
    images, texts = load_caption_group_records(
        root / MANIFESTS["skyscript_group"], expected_split="val"
    )
    rs_images, rs_texts = load_rsicd_records(root / MANIFESTS["rsicd"], expected_split="val")
    # Expanded validation must preserve the original held-out caption groups.
    unique_groups = {" ".join(record["caption"].split()).casefold() for record in unique}
    if {record["group_id"] for record in texts} != unique_groups:
        raise ValueError("Grouped validation caption pool differs from original held-out groups")
    representatives = {(Path(record["image"]).resolve(), record["caption"]) for record in unique}
    expanded_pairs = {(Path(record["image"]).resolve(), record["caption"]) for record in images}
    if not representatives <= expanded_pairs:
        raise ValueError(
            "Grouped validation must retain every original representative image/caption pair"
        )
    original_captions = {
        " ".join(record["caption"].split()).casefold(): record["caption"] for record in unique
    }
    if any(record["caption"] != original_captions[record["group_id"]] for record in texts):
        raise ValueError("Caption candidates must preserve original validation caption spelling")
    if training_images and training_images & {Path(record["image"]).resolve() for record in images}:
        raise ValueError("Grouped training and expanded validation image paths overlap")
    if training_ids and training_ids & {record["id"] for record in images}:
        raise ValueError("Grouped training and expanded validation sample ids overlap")
    return {
        "skyscript_unique": {"images": len(unique), "captions": len(unique), "pairs": len(unique)},
        "skyscript_group": {
            "images": len(images),
            "captions": len(texts),
            "groups": len(texts),
            "positive_pairs": len(images),
        },
        "rsicd": {"images": len(rs_images), "captions": len(rs_texts)},
    }


def _execute(command: list[str], log: Path, *, root: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("command=" + " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def _package(root: Path, trials: list[Trial], baseline: Trial | None) -> Path:
    archive = root / "outputs/sat_multi_positive_seed11_reports.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".part")
    paths: set[Path] = set()
    for trial in trials:
        paths.update(
            path
            for path in trial.output.iterdir()
            if path.is_file() and path.suffix in {".json", ".jsonl", ".toml", ".log"}
        )
    if baseline is not None:
        paths.update(
            baseline.output / name
            for name in (
                "config.toml",
                "training_summary.json",
                "provenance.json",
                "optimizer_groups.json",
            )
            if (baseline.output / name).is_file()
        )
    comparison = root / COMPARISON_DIR
    if baseline is not None and comparison.exists():
        paths.update(
            path
            for path in comparison.rglob("*")
            if path.is_file() and path.suffix in {".json", ".log"}
        )
    audit = root / GROUP_DIR / "audit.json"
    if audit.is_file():
        paths.add(audit)
    try:
        with tarfile.open(temporary, "w:gz") as output:
            for path in sorted(paths):
                if not path.is_symlink():
                    output.add(path, arcname=str(path.relative_to(root)), recursive=False)
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"reports_archive={archive}", flush=True)
    return archive


def run_pipeline(root: Path, python: str, *, mode: str = "all") -> Path:
    if mode not in {"all", "train-only", "evaluate-only"}:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    root = root.resolve()
    trials = [
        Trial(
            method,
            root / f"configs/skyscript_sat_adapter_textlora_{method}_1710step_seed11.toml",
            root / f"outputs/skyscript_sat_adapter_textlora_{method}_1710step_seed11",
        )
        for method in METHODS
    ]
    states = {}
    checked_data: set[tuple[Path, Path]] = set()
    all_training_images: set[Path] = set()
    all_training_ids: set[str] = set()
    # All read-only checks precede the first GPU subprocess.
    for trial in trials:
        config = load_config(trial.config)
        output = config.experiment.output_dir
        if not output.is_absolute():
            output = root / output
        if output.resolve() != trial.output.resolve() or config.experiment.seed != 11:
            raise ValueError(f"Unexpected experiment output/seed: {trial.config}")
        if config.train.max_steps != TARGET_STEPS:
            raise ValueError(f"Unexpected training budget: {trial.config}")
        missing = [
            path
            for path in required_paths(config)
            if not (path if path.is_absolute() else root / path).exists()
        ]
        if missing:
            raise FileNotFoundError("Missing required paths: " + ", ".join(map(str, missing)))
        train_manifest = config.data.train_manifest
        val_manifest = config.data.val_manifest
        if val_manifest is None:
            raise ValueError(f"Original unique validation is required: {trial.config}")
        train_manifest = (root / train_manifest).resolve()
        val_manifest = (root / val_manifest).resolve()
        if val_manifest != (root / MANIFESTS["skyscript_unique"]).resolve():
            raise ValueError(f"Training must use the original unique validation: {trial.config}")
        data_key = (train_manifest, val_manifest)
        if data_key not in checked_data:
            train_records, train_captions = load_caption_group_records(
                train_manifest, expected_split="train"
            )
            train_reference = load_paired_records(
                root / TRAIN_REFERENCE, expected_split="train", expected_source="SkyScript"
            )
            val_records = load_paired_records(
                val_manifest, expected_split="val", expected_source="SkyScript"
            )
            train_groups = {record["group_id"] for record in train_captions}
            reference_groups = {
                " ".join(record["caption"].split()).casefold() for record in train_reference
            }
            if train_groups != reference_groups:
                raise ValueError(
                    "Grouped training caption pool differs from original training groups"
                )
            reference_pairs = {
                (Path(record["image"]).resolve(), record["caption"]) for record in train_reference
            }
            train_pairs = {
                (Path(record["image"]).resolve(), record["caption"]) for record in train_records
            }
            if not reference_pairs <= train_pairs:
                raise ValueError(
                    "Grouped training must retain every original representative image/caption pair"
                )
            val_groups = {" ".join(record["caption"].split()).casefold() for record in val_records}
            train_images = {Path(record["image"]).resolve() for record in train_records}
            val_images = {Path(record["image"]).resolve() for record in val_records}
            if train_groups & val_groups or train_images & val_images:
                raise ValueError("Grouped training and unique validation overlap")
            train_ids = {record["id"] for record in train_records}
            if train_ids & {record["id"] for record in val_records}:
                raise ValueError("Grouped training and unique validation sample ids overlap")
            all_training_images.update(train_images)
            all_training_ids.update(train_ids)
            checked_data.add(data_key)
        states[trial.method] = training_state(trial)
        if mode == "evaluate-only" and states[trial.method] != "complete":
            raise RuntimeError(f"--evaluate-only requires completed training: {trial.output}")
        print(f"preflight={trial.method} state={states[trial.method]}", flush=True)
    baseline = None
    counts = {}
    if mode != "train-only":
        baseline = Trial(
            "baseline", root / f"outputs/{BASELINE}/config.toml", root / f"outputs/{BASELINE}"
        )
        if training_state(baseline) != "complete":
            raise RuntimeError("Original adapter+LoRA baseline must already be complete")
        counts = _pool_counts(
            root, training_images=all_training_images, training_ids=all_training_ids
        )
        for trial in [baseline, *trials]:
            for dataset in MANIFESTS:
                for tag in ("step_0000000", "best"):
                    report = root / COMPARISON_DIR / trial.method / f"{dataset}_{tag}.json"
                    if report.exists():
                        if trial is not baseline and states[trial.method] != "complete":
                            raise RuntimeError(
                                f"Existing evaluation for incomplete training: {report}"
                            )
                        validate_report(
                            report, trial, dataset, tag, root / MANIFESTS[dataset], counts[dataset]
                        )
    if mode != "evaluate-only":
        for trial in trials:
            state = states[trial.method]
            if state == "complete":
                print(f"already_completed={trial.method}", flush=True)
                continue
            command = [python, "-u", "-m", "dinotxt_rs.cli.train", "--config", str(trial.config)]
            if state == "resume":
                command += ["--resume", str(trial.output / "latest.pt")]
            _execute(command, trial.output / "train.log", root=root)
            if training_state(trial) != "complete":
                raise RuntimeError(f"Training did not complete: {trial.output}")
    if mode != "train-only":
        assert baseline is not None
        for trial in [baseline, *trials]:
            for dataset, relative_manifest in MANIFESTS.items():
                manifest = root / relative_manifest
                for tag in ("step_0000000", "best"):
                    report = root / COMPARISON_DIR / trial.method / f"{dataset}_{tag}.json"
                    if not report.exists():
                        cli = "rsicd" if dataset == "rsicd" else "skyscript"
                        command = [
                            python,
                            "-u",
                            "-m",
                            f"dinotxt_rs.cli.evaluate_{cli}",
                            "--config",
                            str(trial.output / "config.toml"),
                            "--manifest",
                            str(manifest),
                            "--checkpoint",
                            str(trial.output / f"{tag}.pt"),
                            "--training-output",
                            str(trial.output),
                            "--split",
                            "val",
                            "--output",
                            str(report),
                        ]
                        if dataset == "skyscript_group":
                            command += ["--positive-definition", "caption-group"]
                        _execute(command, report.parent / "retrieval.log", root=root)
                    validate_report(report, trial, dataset, tag, manifest, counts[dataset])
    return _package(root, trials, baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train-only", action="store_true")
    modes.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    mode = "train-only" if args.train_only else "evaluate-only" if args.evaluate_only else "all"
    root = Path(__file__).resolve().parents[1]
    run_pipeline(root, sys.executable, mode=mode)


if __name__ == "__main__":
    main()
