#!/usr/bin/env python3
"""Explore all training images for one epoch, independently of the 1710-step series."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tarfile
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Also support the repository's usual ``python tools/name.py`` entry point.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinotxt_rs.config import load_config, required_paths  # noqa: E402
from dinotxt_rs.evaluation.retrieval import (  # noqa: E402
    load_caption_group_records,
    load_paired_records,
)
from tools.run_sat_multi_positive import (  # noqa: E402
    GROUP_DIR,
    MANIFESTS,
    TRAIN_REFERENCE,
    Trial,
    _execute,
    _pool_counts,
    _read,
    training_state,
    validate_report,
)

METHODS = ("multipos", "maskpos")
TAGS = ("step_0000000", "best", "latest")
REPORT_DIR = Path("outputs/sat_full_image_epoch_seed11")


def full_training_state(trial: Trial, *, images: int, epochs: int, steps: int) -> str:
    state = training_state(trial, target_steps=steps)
    if state == "complete":
        summary = _read(trial.output / "training_summary.json")
        coverage = summary.get("image_epochs", {})
        if (
            coverage.get("target_epochs") != epochs
            or coverage.get("completed_epochs") != epochs
            or coverage.get("current_epoch_batch_offset") != 0
            or coverage.get("epoch_definition") != "one_pass_over_every_training_image"
            or coverage.get("sample_exposures") != images * epochs
            or coverage.get("unique_images_seen") != images
            or coverage.get("image_pool_coverage") != 1.0
            or summary.get("samples_in_manifest") != images
            or summary.get("skipped_optimizer_steps") != 0
        ):
            raise ValueError(f"Completed run lacks full image coverage: {trial.output}")
    return state


def _training_pool(root: Path, manifest: Path) -> tuple[list[dict[str, Any]], set[Path], set[str]]:
    records, texts = load_caption_group_records(manifest, expected_split="train")
    reference = load_paired_records(
        root / TRAIN_REFERENCE, expected_split="train", expected_source="SkyScript"
    )
    groups = {record["group_id"] for record in texts}
    reference_groups = {" ".join(record["caption"].split()).casefold() for record in reference}
    if groups != reference_groups:
        raise ValueError("Full-image training must retain the original training caption groups")
    representatives = {(Path(row["image"]).resolve(), row["caption"]) for row in reference}
    expanded = {(Path(row["image"]).resolve(), row["caption"]) for row in records}
    if not representatives <= expanded:
        raise ValueError("Full-image pool is missing original training representatives")
    audit = _read(root / GROUP_DIR / "audit.json")
    if (
        audit["splits"]["train"]["records"] != len(records)
        or audit["splits"]["train"]["caption_groups"] != len(groups)
    ):
        raise ValueError("Full-image manifest counts disagree with the data audit")
    return (
        records, {Path(row["image"]).resolve() for row in records},
        {row["id"] for row in records},
    )


def _package(
    root: Path, trials: list[Trial], *, report_dir: Path = REPORT_DIR,
    series: str = "sat_full_image_epoch_seed11",
) -> Path:
    archive = root / "outputs" / f"{series}_reports.tar.gz"
    temporary = archive.with_name(archive.name + ".part")
    paths: set[Path] = {root / GROUP_DIR / "audit.json"}
    for trial in trials:
        if trial.output.exists():
            paths.update(path for path in trial.output.iterdir()
                         if path.is_file() and path.suffix in {".json", ".jsonl", ".toml", ".log"})
    comparison = root / report_dir
    if comparison.exists():
        paths.update(path for path in comparison.rglob("*")
                     if path.is_file() and path.suffix in {".json", ".log"})
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(temporary, "w:gz") as bundle:
            for path in sorted(paths):
                if not path.is_symlink():
                    bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"reports_archive={archive}", flush=True)
    return archive


def run_pipeline(
    root: Path, python: str, *, mode: str = "all", trials: list[Trial] | None = None,
    report_dir: Path = REPORT_DIR, series: str = "sat_full_image_epoch_seed11",
    require_matching_recipes: bool = True,
) -> Path | None:
    if mode not in {"all", "train-only", "evaluate-only", "preflight-only"}:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    root = root.resolve()
    trials = trials if trials is not None else [Trial(
        method,
        root / f"configs/skyscript_sat_adapter_textlora_{method}_fullimage1epoch_seed11.toml",
        root / f"outputs/skyscript_sat_adapter_textlora_{method}_fullimage1epoch_seed11",
    ) for method in METHODS]
    states: dict[str, str] = {}
    configs = {}
    pool = None
    shared = None
    for trial in trials:
        config = load_config(trial.config)
        configs[trial.method] = config
        if (
            config.experiment.seed != 11
            or (root / config.experiment.output_dir).resolve() != trial.output
            or config.data.caption_sampling != "image_epoch"
            or (require_matching_recipes and config.train.image_epochs != 1)
            or config.data.images_per_caption != 2
            or config.train.contrastive_objective != ({
                "multipos": "multi_positive", "maskpos": "mask_same_caption"
            }[trial.method] if require_matching_recipes else "mask_same_caption")
        ):
            raise ValueError(f"Unexpected full-image protocol: {trial.config}")
        missing = [path for path in required_paths(config)
                   if not (path if path.is_absolute() else root / path).exists()]
        if missing:
            raise FileNotFoundError("Missing required paths: " + ", ".join(map(str, missing)))
        if config.data.val_manifest is None or (
            root / config.data.val_manifest
        ).resolve() != (root / MANIFESTS["skyscript_unique"]).resolve():
            raise ValueError("Full-image series retains the original unique validation")
        signature = (config.model, config.data, config.experiment.seed,
                     {key: value for key, value in vars(config.train).items()
                      if key != "contrastive_objective"})
        if shared is not None and require_matching_recipes and signature != shared:
            raise ValueError("multipos and maskpos must share model, sampling and training budget")
        if shared is not None and not require_matching_recipes and config.data != shared[1]:
            raise ValueError("Full-image follow-ups must share the same data protocol")
        shared = signature
        if pool is None:
            pool = _training_pool(root, (root / config.data.train_manifest).resolve())
        records, training_images, training_ids = pool
        batches = math.ceil(len(records) / config.train.batch_size)
        steps = math.ceil(batches * config.train.image_epochs / config.train.gradient_accumulation)
        if config.train.max_steps != steps:
            raise ValueError(f"Expected max_steps={steps} for the full image pool: {trial.config}")
        states[trial.method] = full_training_state(
            trial, images=len(records), epochs=config.train.image_epochs, steps=steps
        )
        if mode == "evaluate-only" and states[trial.method] != "complete":
            raise RuntimeError(f"--evaluate-only requires completed training: {trial.output}")
        print(
            f"preflight={trial.method} state={states[trial.method]} images={len(records)} "
            f"epochs={config.train.image_epochs} micro_batches={batches} optimizer_steps={steps}",
            flush=True,
        )
    assert pool is not None
    records, training_images, training_ids = pool
    counts = _pool_counts(
        root, training_images=training_images, training_ids=training_ids,
        include_rsicd=mode != "train-only",
    )
    # Also guard original unique validation explicitly, including its sample IDs.
    unique = load_paired_records(root / MANIFESTS["skyscript_unique"], expected_split="val")
    val_groups = {" ".join(row["caption"].split()).casefold() for row in unique}
    if {row["group_id"] for row in records} & val_groups:
        raise ValueError("Full-image training and validation caption groups overlap")
    if training_ids & {row["id"] for row in unique} or training_images & {
        Path(row["image"]).resolve() for row in unique
    }:
        raise ValueError("Full-image training and unique validation images or sample IDs overlap")
    for trial in trials:
        for dataset, relative_manifest in MANIFESTS.items():
            for tag in TAGS:
                report = root / report_dir / trial.method / f"{dataset}_{tag}.json"
                if report.exists():
                    if states[trial.method] != "complete":
                        raise RuntimeError(f"Existing evaluation for incomplete training: {report}")
                    if dataset in counts:
                        validate_report(report, trial, dataset, tag, root / relative_manifest,
                                        counts[dataset])
    if mode == "preflight-only":
        print("preflight_complete=true; no training or output writes", flush=True)
        return None
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
            config = configs[trial.method]
            if full_training_state(trial, images=len(records), epochs=config.train.image_epochs,
                                   steps=config.train.max_steps) != "complete":
                raise RuntimeError(f"Training did not complete full coverage: {trial.output}")
    if mode != "train-only":
        for trial in trials:
            for dataset, relative_manifest in MANIFESTS.items():
                for tag in TAGS:
                    report = root / report_dir / trial.method / f"{dataset}_{tag}.json"
                    if not report.exists():
                        cli = "rsicd" if dataset == "rsicd" else "skyscript"
                        command = [
                            python, "-u", "-m", f"dinotxt_rs.cli.evaluate_{cli}",
                            "--config", str(trial.output / "config.toml"),
                            "--manifest", str(root / relative_manifest),
                            "--checkpoint", str(trial.output / f"{tag}.pt"),
                            "--training-output", str(trial.output), "--split", "val",
                            "--output", str(report),
                        ]
                        if dataset == "skyscript_group":
                            command += ["--positive-definition", "caption-group"]
                        _execute(command, report.parent / "retrieval.log", root=root)
                    validate_report(report, trial, dataset, tag, root / relative_manifest,
                                    counts[dataset])
    summary = {
        "series": series, "mode": mode,
        "comparison_scope": "full_image_coverage_exploration_not_matched_to_1710_step_series",
        "training_images": len(records),
        "image_epochs": (
            configs[trials[0].method].train.image_epochs if require_matching_recipes
            else {trial.method: configs[trial.method].train.image_epochs for trial in trials}
        ),
        "trials": {trial.method: _read(trial.output / "training_summary.json") for trial in trials},
        "retrieval": {
            trial.method: {
                f"{dataset}_{tag}": _read(
                    root / report_dir / trial.method / f"{dataset}_{tag}.json"
                )["metrics"] for dataset in MANIFESTS for tag in TAGS
            } for trial in trials
        } if mode != "train-only" else {},
    }
    output_reports = root / report_dir
    output_reports.mkdir(parents=True, exist_ok=True)
    temporary = output_reports / "summary.json.part"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_reports / "summary.json")
    return _package(root, trials, report_dir=report_dir, series=series)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ("train-only", "evaluate-only", "preflight-only"):
        modes.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    mode = next((name for name in ("train-only", "evaluate-only", "preflight-only")
                 if getattr(args, name.replace("-", "_"))), "all")
    run_pipeline(Path(__file__).resolve().parents[1], sys.executable, mode=mode)


if __name__ == "__main__":
    main()
