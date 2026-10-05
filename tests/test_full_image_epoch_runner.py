from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from dinotxt_rs.evaluation.retrieval import RETRIEVAL_TIE_POLICY
from tools import run_sat_full_image_epoch as runner


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(root: Path, relative: Path, rows: list[dict]) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _rows(root: Path, label: str, captions: tuple[str, ...], *, split: str,
          source: str = "SkyScript", grouped: bool = False) -> list[dict]:
    rows = []
    for index, caption in enumerate(captions):
        image = root / "images" / f"{label}-{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"fake image")
        row = {
            "id": f"{label}:{index}", "image": str(image), "caption": caption,
            "split": split, "source": source,
        }
        if grouped:
            row["group_id"] = " ".join(caption.split()).casefold()
        if source == "RSICD":
            row["image_id"] = str(index)
        rows.append(row)
    return rows


def _trial(root: Path, method: str):
    name = f"skyscript_sat_adapter_textlora_{method}_fullimage1epoch_seed11"
    return runner.Trial(
        method, root / "configs" / f"{name}.toml", root / "outputs" / name
    )


def _fake_project(tmp_path: Path, monkeypatch, *, max_steps: dict[str, int] | None = None):
    root = tmp_path / "project with spaces"
    (root / "configs").mkdir(parents=True)
    for method in runner.METHODS:
        _trial(root, method).config.write_text(f"{method}\n", encoding="utf-8")

    train_rows = _rows(
        root, "train", ("road", "road", "river", "river", "field",
                         "field", "factory", "factory", "road"),
        split="train", grouped=True,
    )
    train_manifest = _manifest(root, runner.GROUP_DIR / "train_grouped.jsonl", train_rows)
    reference = [dict(row) for row in (train_rows[0], train_rows[2], train_rows[4], train_rows[6])]
    for row in reference:
        row.pop("group_id")
    _manifest(root, runner.TRAIN_REFERENCE, reference)
    _json(root / runner.GROUP_DIR / "audit.json", {
        "splits": {"train": {"records": 9, "caption_groups": 4}}
    })

    unique_rows = _rows(root, "unique-val", ("airport", "forest"), split="val")
    unique_path = _manifest(root, runner.MANIFESTS["skyscript_unique"], unique_rows)
    group_rows = _rows(
        root, "group-val", ("airport", "airport", "forest"), split="val", grouped=True
    )
    group_rows[0]["image"] = unique_rows[0]["image"]
    group_rows[2]["image"] = unique_rows[1]["image"]
    _manifest(root, runner.MANIFESTS["skyscript_group"], group_rows)
    rs_rows = _rows(root, "rsicd-val", ("a river", "a field"),
                    split="val", source="RSICD")
    _manifest(root, runner.MANIFESTS["rsicd"], rs_rows)

    config_cache = {}

    def config_loader(path):
        source = Path(path)
        method = "multipos" if "multipos" in source.name else "maskpos"
        output = Path("outputs") / source.stem
        config = SimpleNamespace(
            source=source,
            experiment=SimpleNamespace(seed=11, output_dir=output),
            model=SimpleNamespace(image_size=16, backbone_domain="web"),
            data=SimpleNamespace(
                train_manifest=train_manifest, val_manifest=unique_path,
                caption_sampling="image_epoch", images_per_caption=2,
                shuffle_train=True, train_augmentation=False,
            ),
            train=SimpleNamespace(
                max_steps=(max_steps or {}).get(method, 2), image_epochs=1,
                batch_size=4, gradient_accumulation=2, warmup_steps=1,
                contrastive_objective=(
                    "multi_positive" if method == "multipos" else "mask_same_caption"
                ),
            ),
        )
        config_cache[source] = config
        return config

    monkeypatch.setattr(runner, "load_config", config_loader)
    monkeypatch.setattr(
        runner, "required_paths",
        lambda config: [config.data.train_manifest, config.data.val_manifest],
    )
    calls: list[list[str]] = []

    def write_training(trial):
        trial.output.mkdir(parents=True, exist_ok=True)
        (trial.output / "config.toml").write_bytes(trial.config.read_bytes())
        _json(trial.output / "provenance.json", {"files": {}})
        _json(trial.output / "training_summary.json", {
            "completed": True, "steps": 2, "target_steps": 2,
            "samples_in_manifest": 9, "skipped_optimizer_steps": 0,
            "image_epochs": {
                "target_epochs": 1, "completed_epochs": 1,
                "current_epoch_batch_offset": 0,
                "epoch_definition": "one_pass_over_every_training_image",
                "sample_exposures": 9, "unique_images_seen": 9,
                "image_pool_coverage": 1.0,
            },
            "validation": {"last_step": 2, "best_step": 1},
        })
        for name in runner.TAGS:
            (trial.output / f"{name}.pt").write_bytes(b"fake checkpoint")
        (trial.output / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
        (trial.output / "train.log").write_text("fake train\n", encoding="utf-8")

    def fake_execute(command: list[str], log: Path, *, root: Path):
        calls.append(command)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("fake subprocess\n", encoding="utf-8")
        if command[3] == "dinotxt_rs.cli.train":
            config_path = Path(command[command.index("--config") + 1])
            method = "multipos" if "multipos" in config_path.name else "maskpos"
            write_training(_trial(root, method))
            return

        config_path = Path(command[command.index("--config") + 1])
        method = "multipos" if "multipos" in config_path.parent.name else "maskpos"
        trial = _trial(root, method)
        manifest = Path(command[command.index("--manifest") + 1])
        dataset = next(key for key, value in runner.MANIFESTS.items()
                       if root / value == manifest)
        output = Path(command[command.index("--output") + 1])
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        counts = runner._pool_counts(root)[dataset]
        recalls = {"r1": 0.1, "r5": 0.2, "r10": 0.3,
                   "mean_rank": 5.0, "median_rank": 3.0}
        metrics = {
            "image_to_text": dict(recalls), "text_to_image": dict(recalls),
            "mean_recall": 0.2,
        }
        report = {
            "split": "val", "tie_policy": RETRIEVAL_TIE_POLICY,
            "task": {
                "skyscript_unique": "paired_image_text_global_retrieval",
                "skyscript_group": "caption_group_image_text_global_retrieval",
                "rsicd": "rsicd_image_text_retrieval",
            }[dataset],
            "manifest": {"path": str(manifest)}, "counts": counts,
            "metrics": metrics,
            "model": {
                "model_config": str(trial.output / "config.toml"),
                "checkpoint": {
                    "step": {"step_0000000": 0, "best": 1, "latest": 2}[checkpoint.stem],
                    "path": str(checkpoint),
                    "identity_check": {"status": "match", "blocking_mismatches": []},
                },
            },
        }
        if dataset == "skyscript_unique":
            report["positive_definition"] = "manifest_row_one_to_one"
        elif dataset == "skyscript_group":
            report["positive_definition"] = "normalized_complete_caption_group"
            report["recall_definition"] = "fraction_of_queries_with_any_positive_in_top_k"
            metrics["image_to_text_group_balanced"] = {
                "r1": 0.15, "r5": 0.25, "r10": 0.35,
            }
            metrics["group_balanced_mean_recall"] = 0.225
        _json(output, report)

    monkeypatch.setattr(runner, "_execute", fake_execute)
    return root, calls, write_training


def test_default_runs_two_trainings_eighteen_evaluations_and_metadata_archive(
    tmp_path, monkeypatch,
):
    root, calls, _ = _fake_project(tmp_path, monkeypatch)
    archive = runner.run_pipeline(root, sys.executable)
    assert len(calls) == 20
    assert sum(command[3] == "dinotxt_rs.cli.train" for command in calls) == 2
    assert sum(command[3] != "dinotxt_rs.cli.train" for command in calls) == 18
    assert all("--resume" not in command for command in calls[:2])
    assert len(list((root / runner.REPORT_DIR).glob("*/*.json"))) == 18
    with tarfile.open(archive) as package:
        members = package.getnames()
    assert str(runner.GROUP_DIR / "audit.json") in members
    assert not any(name.endswith((".pt", ".part")) for name in members)

    calls.clear()
    runner.run_pipeline(root, sys.executable)
    assert calls == []


def test_resume_uses_latest_and_completed_trial_is_skipped(tmp_path, monkeypatch):
    root, calls, write_training = _fake_project(tmp_path, monkeypatch)
    # Training still checks both SkyScript validation views, but RSICD is only
    # required when retrieval is requested.
    (root / runner.MANIFESTS["rsicd"]).unlink()
    for method in runner.METHODS:
        trial = _trial(root, method)
        if method == "maskpos":
            write_training(trial)
        else:
            trial.output.mkdir(parents=True)
            (trial.output / "config.toml").write_bytes(trial.config.read_bytes())
            _json(trial.output / "provenance.json", {"files": {}})
            (trial.output / "latest.pt").write_bytes(b"resume checkpoint")
    runner.run_pipeline(root, sys.executable, mode="train-only")
    training_calls = [command for command in calls if command[3] == "dinotxt_rs.cli.train"]
    assert len(training_calls) == 1
    assert "--resume" in training_calls[0]
    assert training_calls[0][-1] == str(_trial(root, "multipos").output / "latest.pt")


def test_incomplete_coverage_summary_is_rejected_before_training(tmp_path, monkeypatch):
    root, calls, write_training = _fake_project(tmp_path, monkeypatch)
    trial = _trial(root, "multipos")
    write_training(trial)
    summary_path = trial.output / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["image_epochs"]["image_pool_coverage"] = 0.99
    _json(summary_path, summary)
    with pytest.raises(ValueError, match="full image coverage"):
        runner.run_pipeline(root, sys.executable)
    assert calls == []


def test_max_steps_mismatch_fails_preflight_before_training(tmp_path, monkeypatch):
    root, calls, _ = _fake_project(
        tmp_path, monkeypatch, max_steps={"multipos": 3, "maskpos": 3}
    )
    with pytest.raises(ValueError, match="max_steps=2"):
        runner.run_pipeline(root, sys.executable)
    assert calls == []


def test_preflight_only_does_not_create_run_or_report_outputs(tmp_path, monkeypatch):
    root, calls, _ = _fake_project(tmp_path, monkeypatch)
    result = runner.run_pipeline(root, sys.executable, mode="preflight-only")
    assert result is None
    assert calls == []
    assert not any(_trial(root, method).output.exists() for method in runner.METHODS)
    assert not (root / runner.REPORT_DIR).exists()
