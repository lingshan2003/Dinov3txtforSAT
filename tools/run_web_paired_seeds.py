#!/usr/bin/env python3
"""Replicate Web adapter-only versus adapter+LoRA on paired seeds 11/23/47."""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinotxt_rs.config import load_config  # noqa: E402
from dinotxt_rs.evaluation.common import write_json_atomic  # noqa: E402
from tools import run_sat_full_image_epoch as shared  # noqa: E402
from tools import run_web_adapter_control as control  # noqa: E402
from tools import run_web_native as native  # noqa: E402
from tools.run_sat_multi_positive import Trial  # noqa: E402

SEEDS = (11, 23, 47)
METHODS = control.METHODS
REPORT_DIR = Path("outputs/web_paired_seeds")
SERIES = "web_paired_seeds"


def validated_trials(root: Path, seed: int) -> list[Trial]:
    if seed not in SEEDS:
        raise ValueError(f"Unexpected paired experiment seed: {seed}")
    root = root.resolve()
    references = native.validated_trials(root, METHODS)
    trials = []
    for reference in references:
        name = reference.config.stem.replace("seed11", f"seed{seed}")
        trial = Trial(reference.method, root / "configs" / f"{name}.toml",
                      root / "outputs" / name)
        config, original = load_config(trial.config), load_config(reference.config)
        if (
            config.experiment != replace(
                original.experiment, name=name, seed=seed, output_dir=Path("outputs") / name,
            )
            or config.model != original.model
            or config.data != original.data
            or config.train != original.train
        ):
            raise ValueError(f"Paired seeds must change only name/output_dir/seed: {trial.config}")
        trials.append(trial)
    return trials


def _options(seed: int, trials: list[Trial]) -> dict[str, Any]:
    return {
        "trials": trials, "report_dir": REPORT_DIR / f"seed{seed}",
        "series": f"{SERIES}_seed{seed}", "require_matching_recipes": False,
        "expected_seed": seed,
    }


def _validate_seed11_sources(
    root: Path, trials: list[Trial], counts: dict[str, dict[str, int]],
) -> None:
    """Check any source caches and aggregate baseline before starting GPU work."""
    config = load_config(trials[0].config)
    for directory in (root / native.REPORT_DIR / "official", root / REPORT_DIR / "official"):
        snapshot = directory / "config.toml"
        if snapshot.exists() and snapshot.read_bytes() != config.source.read_bytes():
            raise ValueError(f"Native reference configuration differs: {snapshot}")
        if any(directory.glob("*.json")) and not snapshot.exists():
            raise ValueError(f"Cached native reference has no config snapshot: {directory}")
        for dataset, relative in shared.MANIFESTS.items():
            path = directory / f"{dataset}.json"
            if path.exists():
                native.validate_official_report(path, config, dataset, root / relative,
                                                counts[dataset])
    for trial in trials:
        for dataset, relative in shared.MANIFESTS.items():
            for tag in shared.TAGS:
                path = root / control.REPORT_DIR / trial.method / f"{dataset}_{tag}.json"
                if path.exists():
                    shared.validate_report(path, trial, dataset, tag, root / relative,
                                           counts[dataset])


def _copy_cached_reports(
    root: Path, trials: list[Trial], counts: dict[str, dict[str, int]],
) -> None:
    _validate_seed11_sources(root, trials, counts)
    official = root / REPORT_DIR / "official"
    official.mkdir(parents=True, exist_ok=True)
    for dataset in shared.MANIFESTS:
        source = root / native.REPORT_DIR / "official" / f"{dataset}.json"
        target = official / source.name
        if not target.exists():
            shutil.copyfile(source, target)
    for name in ("config.toml", "summary.json"):
        shutil.copyfile(root / native.REPORT_DIR / "official" / name, official / name)
    for trial in trials:
        for dataset, relative in shared.MANIFESTS.items():
            for tag in shared.TAGS:
                source = root / control.REPORT_DIR / trial.method / f"{dataset}_{tag}.json"
                target = root / REPORT_DIR / "seed11" / trial.method / source.name
                if source.exists() and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                if target.exists():
                    shared.validate_report(target, trial, dataset, tag, root / relative,
                                           counts[dataset])


def _metrics_flat(metrics: dict[str, Any]) -> dict[str, float]:
    values = {"mean_recall": metrics["mean_recall"]}
    for direction in ("image_to_text", "text_to_image"):
        for key in ("r1", "r5", "r10"):
            values[f"{direction}_{key}"] = metrics[direction][key]
    if "group_balanced_mean_recall" in metrics:
        values["group_balanced_mean_recall"] = metrics["group_balanced_mean_recall"]
    return values


def _stats(values: list[float]) -> dict[str, Any]:
    return {"values": values, "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values)}


def _write_summary(root: Path, grid: dict[int, list[Trial]]) -> None:
    if tuple(grid) != SEEDS:
        raise ValueError("Aggregate requires all three paired seeds in canonical order")
    counts = shared._pool_counts(root)
    canonical = load_config(grid[11][0].config)
    official = {}
    reports: dict[int, Any] = {}
    training = {}
    parity = {}
    for dataset, relative in shared.MANIFESTS.items():
        path = root / REPORT_DIR / "official" / f"{dataset}.json"
        native.validate_official_report(path, canonical, dataset, root / relative, counts[dataset])
        official[dataset] = shared._read(path)
    for seed, trials in grid.items():
        reports[seed], training[seed], parity[seed] = {}, {}, {}
        for trial in trials:
            training[seed][trial.method] = shared._read(trial.output / "training_summary.json")
            reports[seed][trial.method], parity[seed][trial.method] = {}, {}
            for dataset, relative in shared.MANIFESTS.items():
                baseline = official[dataset]
                reports[seed][trial.method][dataset] = {}
                for tag in shared.TAGS:
                    path = (root / REPORT_DIR / f"seed{seed}" / trial.method
                            / f"{dataset}_{tag}.json")
                    shared.validate_report(path, trial, dataset, tag, root / relative,
                                           counts[dataset])
                    value = shared._read(path)
                    if any(value.get(key) != baseline.get(key) for key in (
                        "manifest", "counts", "tie_policy", "positive_definition",
                        "recall_definition",
                    )) or any(value["model"][key] != baseline["model"][key] for key in (
                        "backbone_weights", "dinotxt_weights", "bpe_vocab",
                    )):
                        raise ValueError(f"Paired comparison protocols/assets differ: {path}")
                    reports[seed][trial.method][dataset][tag] = value["metrics"]
                parity[seed][trial.method][dataset] = (
                    reports[seed][trial.method][dataset]["step_0000000"] == baseline["metrics"]
                )
    aggregates = {}
    for tag in ("latest", "best"):
        aggregates[tag] = {}
        for dataset, baseline in official.items():
            by_method = {
                method: [_metrics_flat(reports[seed][method][dataset][tag]) for seed in SEEDS]
                for method in METHODS
            }
            reference = _metrics_flat(baseline["metrics"])
            aggregates[tag][dataset] = {
                "methods_percent": {
                    method: {key: _stats([100 * row[key] for row in rows]) for key in reference}
                    for method, rows in by_method.items()
                },
                "paired_lora_minus_adapter_only_pp": {
                    key: _stats([100 * (a[key] - b[key]) for a, b in zip(
                        by_method["adapter_lora"], by_method["adapter_only"], strict=True,
                    )]) for key in reference
                },
                "delta_to_official_pp": {
                    method: {key: _stats([100 * (row[key] - reference[key]) for row in rows])
                             for key in reference} for method, rows in by_method.items()
                },
            }
    write_json_atomic(root / REPORT_DIR / "comparison.json", {
        "series": SERIES, "seeds": SEEDS, "primary_checkpoint": "latest",
        "std_definition": "sample standard deviation across seeds; ddof=1, not confidence interval",
        "delta_definition": "paired same-seed adapter_lora minus adapter_only, percentage points",
        "counts": counts, "official": {key: value["metrics"] for key, value in official.items()},
        "training": training, "per_seed_metrics": reports,
        "step0_metrics_match_official": parity, "aggregates": aggregates,
    })
    lines = [
        "# Web 三seed配对复现", "",
        "seed11/23/47；主比较使用完整图片覆盖latest。mean ± sample std（ddof=1）。", "",
        "| latest mean Recall | 原生 | adapter-only | adapter＋LoRA | 配对差值 (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, dataset, key in (
        ("unique-val", "skyscript_unique", "mean_recall"),
        ("多图按图", "skyscript_group", "mean_recall"),
        ("多图按caption组", "skyscript_group", "group_balanced_mean_recall"),
        ("RSICD-val", "rsicd", "mean_recall"),
    ):
        value = aggregates["latest"][dataset]
        a, b = (value["methods_percent"][method][key] for method in (
            "adapter_only", "adapter_lora",
        ))
        delta = value["paired_lora_minus_adapter_only_pp"][key]
        lines.append(
            f"| {label} | {official[dataset]['metrics'][key]*100:.4f}% | "
            f"{a['mean']:.4f} ± {a['sample_std']:.4f}% | "
            f"{b['mean']:.4f} ± {b['sample_std']:.4f}% | "
            f"{delta['mean']:+.4f} ± {delta['sample_std']:.4f} |"
        )
    lines += ["", "| seed | 方法 | unique MR | 多图 MR | 组均衡 MR | RSICD MR |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for seed in SEEDS:
        for method in METHODS:
            metrics = reports[seed][method]
            values = [metrics[dataset]["latest"][key] * 100 for dataset, key in (
                ("skyscript_unique", "mean_recall"), ("skyscript_group", "mean_recall"),
                ("skyscript_group", "group_balanced_mean_recall"), ("rsicd", "mean_recall"),
            )]
            lines.append(f"| {seed} | {method} | " + " | ".join(
                f"{value:.4f}%" for value in values) + " |")
    all_parity = all(value for a in parity.values() for b in a.values() for value in b.values())
    lines += [
        "", f"step0与官方指标全部一致：{all_parity}",
        "", "配对差值先在每个seed内相减，再计算均值与样本标准差。", "",
        "三seed为训练随机性重复，验证池固定；标准差不是置信区间，不自动作显著性结论。",
        "所有双向R@1/5/10、best/latest、原生增量与各seed数据详见comparison.json。", "",
    ]
    path = root / REPORT_DIR / "comparison.md"
    temporary = path.with_suffix(".md.part")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    print(f"paired_comparison={path}", flush=True)


def run_pipeline(root: Path, python: str, *, mode: str = "all") -> Path | None:
    if mode not in {"all", "train-only", "evaluate-only", "preflight-only"}:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    root = root.resolve()
    grid = {seed: validated_trials(root, seed) for seed in SEEDS}
    for trial in grid[11]:
        if control._training_state(root, trial) != "complete":
            raise RuntimeError(f"Paired replication requires completed seed11: {trial.output}")
    # Check every selected input/state before running even the first GPU subprocess.
    for seed, trials in grid.items():
        if mode == "evaluate-only":
            for trial in trials:
                if control._training_state(root, trial) != "complete":
                    raise RuntimeError(
                        f"--evaluate-only requires completed training: {trial.output}"
                    )
        shared.run_pipeline(root, python, mode="preflight-only", **_options(seed, trials))
    if mode != "train-only":
        counts = shared._pool_counts(root)
        _validate_seed11_sources(root, grid[11], counts)
    if mode == "preflight-only":
        return None
    if mode != "train-only":
        native._official_baseline(root, python, load_config(grid[11][0].config))
        _copy_cached_reports(root, grid[11], counts)
    for seed, trials in grid.items():
        shared.run_pipeline(root, python, mode=mode, **_options(seed, trials))
    if mode != "train-only":
        _write_summary(root, grid)
    return shared._package(root, [trial for trials in grid.values() for trial in trials],
                           report_dir=REPORT_DIR, series=SERIES)


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
