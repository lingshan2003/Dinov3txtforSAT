#!/usr/bin/env python3
"""Train one Web adapter-only control and compare it with completed adapter+LoRA."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinotxt_rs.config import load_config  # noqa: E402
from dinotxt_rs.evaluation.common import write_json_atomic  # noqa: E402
from tools import run_sat_full_image_epoch as shared  # noqa: E402
from tools import run_web_native as native  # noqa: E402
from tools.run_sat_multi_positive import Trial  # noqa: E402

REPORT_DIR = Path("outputs/web_adapter_control_seed11")
SERIES = "web_adapter_control_seed11"
METHODS = ("adapter_lora", "adapter_only")


def _training_state(root: Path, trial: Trial) -> str:
    config = load_config(trial.config)
    images = shared._read(root / shared.GROUP_DIR / "audit.json")["splits"]["train"]["records"]
    return shared.full_training_state(
        trial, images=images, epochs=config.train.image_epochs, steps=config.train.max_steps,
    )


def _copy_cached_reports(root: Path, primary: Trial, counts: dict[str, dict[str, int]]) -> None:
    """Validate source reports before copying metadata into the control's report directory."""
    config = load_config(primary.config)
    for dataset, relative_manifest in shared.MANIFESTS.items():
        manifest = root / relative_manifest
        source = root / native.REPORT_DIR / "official" / f"{dataset}.json"
        native.validate_official_report(source, config, dataset, manifest, counts[dataset])
        target = root / REPORT_DIR / "official" / source.name
        if target.exists():
            native.validate_official_report(target, config, dataset, manifest, counts[dataset])
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for tag in shared.TAGS:
            source = root / native.REPORT_DIR / primary.method / f"{dataset}_{tag}.json"
            target = root / REPORT_DIR / primary.method / source.name
            if source.exists():
                shared.validate_report(source, primary, dataset, tag, manifest, counts[dataset])
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            if target.exists():
                shared.validate_report(target, primary, dataset, tag, manifest, counts[dataset])
    for name in ("summary.json", "config.toml"):
        source = root / native.REPORT_DIR / "official" / name
        if source.exists():
            shutil.copyfile(source, root / REPORT_DIR / "official" / name)


def _delta_pp(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    values = {"mean_recall": 100 * (after["mean_recall"] - before["mean_recall"])}
    for direction in ("image_to_text", "text_to_image"):
        for key in ("r1", "r5", "r10"):
            values[f"{direction}_{key}"] = 100 * (after[direction][key] - before[direction][key])
    if "group_balanced_mean_recall" in after:
        values["group_balanced_mean_recall"] = 100 * (
            after["group_balanced_mean_recall"] - before["group_balanced_mean_recall"]
        )
    return values


def _write_comparison(root: Path, trials: list[Trial]) -> None:
    counts = shared._pool_counts(root)
    configs = {trial.method: load_config(trial.config) for trial in trials}
    primary = next(trial for trial in trials if trial.method == "adapter_lora")
    official = {}
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    parity = {}
    for dataset, relative_manifest in shared.MANIFESTS.items():
        manifest = root / relative_manifest
        path = root / REPORT_DIR / "official" / f"{dataset}.json"
        native.validate_official_report(
            path, configs[primary.method], dataset, manifest, counts[dataset],
        )
        baseline = shared._read(path)
        official[dataset] = baseline["metrics"]
        reports[dataset] = {}
        parity[dataset] = {}
        for trial in trials:
            reports[dataset][trial.method] = {}
            for tag in shared.TAGS:
                path = root / REPORT_DIR / trial.method / f"{dataset}_{tag}.json"
                shared.validate_report(path, trial, dataset, tag, manifest, counts[dataset])
                value = shared._read(path)
                if any(value.get(key) != baseline.get(key) for key in (
                    "manifest", "counts", "tie_policy", "positive_definition", "recall_definition",
                )):
                    raise ValueError(f"Comparison retrieval protocols differ: {path}")
                for key in ("backbone_weights", "dinotxt_weights", "bpe_vocab"):
                    if value["model"][key] != baseline["model"][key]:
                        raise ValueError(f"Comparison model assets differ: {path}")
                reports[dataset][trial.method][tag] = value["metrics"]
            parity[dataset][trial.method] = (
                reports[dataset][trial.method]["step_0000000"] == baseline["metrics"]
            )
    comparisons = {}
    for tag in ("latest", "best"):
        comparisons[tag] = {}
        for dataset in shared.MANIFESTS:
            lora = reports[dataset]["adapter_lora"][tag]
            adapter = reports[dataset]["adapter_only"][tag]
            comparisons[tag][dataset] = {
                "adapter_lora": lora, "adapter_only": adapter,
                "delta_lora_minus_adapter_only_pp": _delta_pp(lora, adapter),
                "delta_to_official_pp": {
                    "adapter_lora": _delta_pp(lora, official[dataset]),
                    "adapter_only": _delta_pp(adapter, official[dataset]),
                },
            }
    training = {trial.method: shared._read(trial.output / "training_summary.json")
                for trial in trials}
    write_json_atomic(root / REPORT_DIR / "comparison.json", {
        "series": SERIES, "primary_checkpoint": "latest",
        "delta_definition": "Recall percentage points: adapter_lora minus adapter_only",
        "training": training, "counts": counts, "official": official,
        "step0_metrics_match_official": parity, "comparisons": comparisons,
    })
    lines = [
        "# Web adapter-only 与 adapter＋文本LoRA 对照", "",
        "主比较使用完整图片覆盖的 latest；best 按固定 unique-val loss 选择，详见 JSON。", "",
        "| 方法 | 完成step | best step | 验证loss（终点） | 可训练参数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        record = training[method]
        lines.append(
            f"| {method} | {record['steps']} | {record['validation']['best_step']} | "
            f"{record['validation']['final_loss']:.6f} | "
            f"{record['optimizer_parameter_groups']['trainable_parameters']:,} |"
        )
    lines += [
        "", "| latest mean Recall | 原生 | adapter-only | adapter＋LoRA | LoRA−adapter (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    rows = (
        ("unique-val", "skyscript_unique", "mean_recall"),
        ("多图按图", "skyscript_group", "mean_recall"),
        ("多图按caption组", "skyscript_group", "group_balanced_mean_recall"),
        ("RSICD-val", "rsicd", "mean_recall"),
    )
    for label, dataset, metric in rows:
        record = comparisons["latest"][dataset]
        lines.append(
            f"| {label} | {official[dataset][metric]*100:.4f}% | "
            f"{record['adapter_only'][metric]*100:.4f}% | "
            f"{record['adapter_lora'][metric]*100:.4f}% | "
            f"{record['delta_lora_minus_adapter_only_pp'][metric]:+.4f} |"
        )
    lines += [
        "", "| latest R@1 | 原生 | adapter-only | adapter＋LoRA | LoRA−adapter (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in shared.MANIFESTS:
        for direction in ("image_to_text", "text_to_image"):
            record = comparisons["latest"][dataset]
            lines.append(
                f"| {dataset} / {direction} | {official[dataset][direction]['r1']*100:.4f}% | "
                f"{record['adapter_only'][direction]['r1']*100:.4f}% | "
                f"{record['adapter_lora'][direction]['r1']*100:.4f}% | "
                f"{record['delta_lora_minus_adapter_only_pp'][direction+'_r1']:+.4f} |"
            )
    lines += [
        "", f"step0与官方指标一致：{parity}", "",
        "单seed开发验证；正增量表示LoRA候选Recall更高，不构成显著性或独立test结论。",
        "所有双向R@1/5/10、best/latest与相对原生变化均保存在comparison.json。", "",
    ]
    path = root / REPORT_DIR / "comparison.md"
    temporary = path.with_suffix(".md.part")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    print(f"comparison_report={path}", flush=True)


def run_pipeline(root: Path, python: str, *, mode: str = "all") -> Path | None:
    if mode not in {"all", "train-only", "evaluate-only", "preflight-only"}:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    root = root.resolve()
    trials = native.validated_trials(root, METHODS)
    primary, control = trials
    if _training_state(root, primary) != "complete":
        raise RuntimeError(
            "This control requires completed adapter+LoRA training; "
            "run scripts/run_web_native.sh first. The primary will not be retrained."
        )
    options = {
        "trials": trials, "report_dir": REPORT_DIR, "series": SERIES,
        "require_matching_recipes": False,
    }
    if mode == "train-only":
        return shared.run_pipeline(root, python, mode=mode, **options)
    shared.run_pipeline(root, python, mode="preflight-only", **options)
    if mode == "preflight-only":
        return None
    if mode == "evaluate-only" and _training_state(root, control) != "complete":
        raise RuntimeError(
            f"--evaluate-only requires completed adapter-only training: {control.output}"
        )
    native._official_baseline(root, python, load_config(primary.config))
    _copy_cached_reports(root, primary, shared._pool_counts(root))
    shared.run_pipeline(root, python, mode=mode, **options)
    _write_comparison(root, trials)
    return shared._package(root, trials, report_dir=REPORT_DIR, series=SERIES)


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
