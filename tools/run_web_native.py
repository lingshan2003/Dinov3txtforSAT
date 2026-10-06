#!/usr/bin/env python3
"""Evaluate native DINOv3.txt first, then adapt it to remote sensing for one image epoch."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinotxt_rs.config import Config, load_config  # noqa: E402
from dinotxt_rs.evaluation.common import write_json_atomic  # noqa: E402
from dinotxt_rs.evaluation.retrieval import RETRIEVAL_TIE_POLICY  # noqa: E402
from tools import run_sat_full_image_epoch as shared  # noqa: E402
from tools.run_sat_multi_positive import Trial, _recalls  # noqa: E402

REPORT_DIR = Path("outputs/web_native_seed11")
SERIES = "web_native_seed11"
WEB_WEIGHTS = Path("assets/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
VARIANTS = {
    "adapter_lora": "skyscript_web_adapter_textlora_maskpos_fullimage1epoch_seed11",
    "adapter_only": "skyscript_web_adapteronly_maskpos_fullimage1epoch_seed11",
    "head_lora": "skyscript_web_head_textlora_maskpos_fullimage1epoch_seed11",
}


def validate_official_report(
    report: Path, config: Config, dataset: str, manifest: Path, counts: dict[str, int],
) -> None:
    """Reject cached baselines from another model, split, or retrieval pool."""
    value = shared._read(report)
    task = {
        "skyscript_unique": "paired_image_text_global_retrieval",
        "skyscript_group": "caption_group_image_text_global_retrieval",
        "rsicd": "rsicd_image_text_retrieval",
    }[dataset]
    model = value.get("model", {})
    if (
        value.get("task") != task
        or value.get("split") != "val"
        or value.get("tie_policy") != RETRIEVAL_TIE_POLICY
        or any(value.get("counts", {}).get(key) != count for key, count in counts.items())
        or Path(value.get("manifest", {}).get("path", "")).resolve() != manifest.resolve()
        or Path(model.get("model_config", "")).resolve() != config.source.resolve()
        or model.get("backbone_domain") != "web"
        or model.get("model_variant") != "official_without_image_adapter"
        or model.get("checkpoint") is not None
        or model.get("trainable_parameters", {}).get("trainable") != 0
        or any(
            Path(model.get(key, {}).get("path", "")).resolve()
            != (config.source.parent.parent / getattr(config.model, key)).resolve()
            for key in ("backbone_weights", "dinotxt_weights", "bpe_vocab")
        )
    ):
        raise ValueError(f"Unexpected native official reference identity: {report}")
    metrics = value["metrics"]
    recalls = []
    for direction in ("image_to_text", "text_to_image"):
        recalls.extend(_recalls(metrics[direction], str(report)))
        for key in ("mean_rank", "median_rank"):
            rank = metrics[direction][key]
            if type(rank) not in (int, float) or not math.isfinite(rank) or rank < 1:
                raise ValueError(f"Invalid native retrieval rank: {report}")
    mean = metrics["mean_recall"]
    if type(mean) not in (int, float) or not math.isclose(mean, sum(recalls) / 6, abs_tol=1e-12):
        raise ValueError(f"Invalid native mean Recall: {report}")
    if dataset == "skyscript_unique":
        if value.get("positive_definition") != "manifest_row_one_to_one":
            raise ValueError(f"Unexpected native positive definition: {report}")
    elif dataset == "skyscript_group":
        if (
            value.get("positive_definition") != "normalized_complete_caption_group"
            or value.get("recall_definition")
            != "fraction_of_queries_with_any_positive_in_top_k"
        ):
            raise ValueError(f"Unexpected native group Recall definition: {report}")
        balanced = _recalls(metrics["image_to_text_group_balanced"], str(report))
        balanced_mean = metrics["group_balanced_mean_recall"]
        if type(balanced_mean) not in (int, float) or not math.isclose(
            balanced_mean, (sum(balanced) + sum(recalls[3:])) / 6, abs_tol=1e-12,
        ):
            raise ValueError(f"Invalid native group-balanced Recall: {report}")


def _official_baseline(root: Path, python: str, config: Config) -> None:
    directory = root / REPORT_DIR / "official"
    snapshot = directory / "config.toml"
    config_bytes = config.source.read_bytes()
    if snapshot.exists():
        if snapshot.read_bytes() != config_bytes:
            raise ValueError(f"Native reference configuration differs: {snapshot}")
    elif directory.exists() and any(directory.glob("*.json")):
        raise ValueError(f"Cached native reference has no config snapshot: {directory}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(config_bytes)
    counts = shared._pool_counts(root)
    reports: dict[str, Any] = {}
    for dataset, relative_manifest in shared.MANIFESTS.items():
        manifest = root / relative_manifest
        report = directory / f"{dataset}.json"
        if not report.exists():
            cli = "rsicd" if dataset == "rsicd" else "skyscript"
            command = [
                python, "-u", "-m", f"dinotxt_rs.cli.evaluate_{cli}",
                "--config", str(config.source), "--manifest", str(manifest),
                "--official", "--split", "val", "--output", str(report),
            ]
            if dataset == "skyscript_group":
                command += ["--positive-definition", "caption-group"]
            shared._execute(command, directory / "retrieval.log", root=root)
        validate_official_report(report, config, dataset, manifest, counts[dataset])
        reports[dataset] = shared._read(report)["metrics"]
        print(f"official={dataset} mean_recall={reports[dataset]['mean_recall']:.6f}", flush=True)
    write_json_atomic(directory / "summary.json", {
        "series": SERIES, "model": "native_official_web_dinov3_txt",
        "training_performed": False, "retrieval": reports,
    })


def validated_trials(root: Path, methods: tuple[str, ...]) -> list[Trial]:
    """Validate selected Web variants without starting work or writing output."""
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(VARIANTS):
        raise ValueError(f"Invalid native Web variant selection: {methods}")
    root = root.resolve()
    reference = load_config(
        root / "configs/skyscript_sat_adapter_textlora_maskpos_fullimage1epoch_seed11.toml"
    )
    trials = []
    for method in methods:
        name = VARIANTS[method]
        trial = Trial(method, root / "configs" / f"{name}.toml", root / "outputs" / name)
        config = load_config(trial.config)
        head = method == "head_lora"
        lora = method != "adapter_only"
        expected_model = replace(
            reference.model, backbone_domain="web", backbone_weights=WEB_WEIGHTS,
            train_vision_head=head, image_adapter_bottleneck=0 if head else 256,
            text_lora_rank=8 if lora else 0, text_lora_include_projection=lora,
        )
        expected_train = reference.train
        if head:
            expected_train = replace(
                expected_train, image_adapter_learning_rate=None, image_adapter_weight_decay=None,
                vision_head_learning_rate=1e-5, vision_head_weight_decay=0.01,
            )
        if (
            config.experiment.seed != 11
            or config.experiment.name != name
            or (root / config.experiment.output_dir).resolve() != trial.output
            or config.model != expected_model
            or config.data != reference.data
            or config.train != expected_train
        ):
            raise ValueError(f"Unexpected native Web adaptation protocol: {trial.config}")
        trials.append(trial)
    return trials


def run_pipeline(
    root: Path, python: str, *, mode: str = "all", include_controls: bool = False,
) -> Path | None:
    if mode not in {"all", "baseline-only", "preflight-only", "train-only", "evaluate-only"}:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    root = root.resolve()
    methods = tuple(VARIANTS) if include_controls else ("adapter_lora",)
    trials = validated_trials(root, methods)
    options = {
        "trials": trials, "report_dir": REPORT_DIR, "series": SERIES,
        "require_matching_recipes": False,
    }
    # Check every selected variant and input before evaluating or launching any GPU work.
    if mode == "train-only":
        return shared.run_pipeline(root, python, mode=mode, **options)
    shared.run_pipeline(root, python, mode="preflight-only", **options)
    if mode == "preflight-only":
        return None
    if mode == "evaluate-only":
        for trial in trials:
            config = load_config(trial.config)
            images = shared._read(root / shared.GROUP_DIR / "audit.json")["splits"]["train"][
                "records"
            ]
            if shared.full_training_state(
                trial, images=images, epochs=config.train.image_epochs,
                steps=config.train.max_steps,
            ) != "complete":
                raise RuntimeError(f"--evaluate-only requires completed training: {trial.output}")
    _official_baseline(root, python, load_config(trials[0].config))
    if mode == "baseline-only":
        return shared._package(root, trials, report_dir=REPORT_DIR, series=SERIES)
    return shared.run_pipeline(root, python, mode=mode, **options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-controls", action="store_true",
        help="Also train adapter-only and head plus text LoRA; default trains only adapter+LoRA.",
    )
    modes = parser.add_mutually_exclusive_group()
    for name in ("baseline-only", "train-only", "evaluate-only", "preflight-only"):
        modes.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    mode = next((name for name in ("baseline-only", "train-only", "evaluate-only", "preflight-only")
                 if getattr(args, name.replace("-", "_"))), "all")
    run_pipeline(Path(__file__).resolve().parents[1], sys.executable, mode=mode,
                 include_controls=args.include_controls)


if __name__ == "__main__":
    main()
