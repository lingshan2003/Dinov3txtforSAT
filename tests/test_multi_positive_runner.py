"""CPU-only orchestration tests; no weights are loaded and no GPU subprocess runs."""

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_sat_multi_positive as runner


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _rows(root, manifest, captions, *, split="val", source="SkyScript", grouped=False):
    values = []
    for index, caption in enumerate(captions):
        image = root / "images" / f"{manifest.stem}-{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        row = {
            "id": f"{manifest.stem}:{index}",
            "image": str(image),
            "caption": caption,
            "split": split,
            "source": source,
        }
        if grouped:
            row["group_id"] = " ".join(caption.split()).casefold()
        if source == "RSICD":
            row["image_id"] = str(index)
        values.append(row)
    path = root / manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
    return path


def _trial(root, method):
    name = f"skyscript_sat_adapter_textlora_{method}_1710step_seed11"
    return runner.Trial(method, root / f"configs/{name}.toml", root / f"outputs/{name}")


def _complete(trial):
    trial.output.mkdir(parents=True, exist_ok=True)
    config = trial.config.read_bytes() if trial.config.exists() else b"baseline config\n"
    (trial.output / "config.toml").write_bytes(config)
    for name in ("step_0000000.pt", "best.pt", "latest.pt"):
        (trial.output / name).write_bytes(b"fake checkpoint")
    _json(trial.output / "provenance.json", {"files": {}})
    _json(
        trial.output / "training_summary.json",
        {
            "completed": True,
            "steps": 1710,
            "target_steps": 1710,
            "validation": {"last_step": 1710, "best_step": 1600},
        },
    )
    (trial.output / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (trial.output / "train.log").write_text("fake train\n", encoding="utf-8")


def _fake_project(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    (root / "configs").mkdir(parents=True)
    for method in runner.METHODS:
        _trial(root, method).config.write_text(method + "\n", encoding="utf-8")
    train = _rows(
        root,
        runner.GROUP_DIR / "train_grouped.jsonl",
        ("bridge", "bridge", "road"),
        split="train",
        grouped=True,
    )
    train_rows = [json.loads(line) for line in train.read_text().splitlines()]
    reference_path = root / runner.TRAIN_REFERENCE
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(
        "".join(
            json.dumps(
                {key: value for key, value in train_rows[index].items() if key != "group_id"}
            )
            + "\n"
            for index in (0, 2)
        ),
        encoding="utf-8",
    )
    _rows(root, runner.MANIFESTS["skyscript_unique"], ("airport", "forest"))
    _rows(root, runner.MANIFESTS["skyscript_group"], ("airport", "airport", "forest"), grouped=True)
    # Restored validation retains the fixed representatives and adds one sibling.
    unique_path = root / runner.MANIFESTS["skyscript_unique"]
    group_path = root / runner.MANIFESTS["skyscript_group"]
    unique_rows = [json.loads(line) for line in unique_path.read_text().splitlines()]
    group_rows = [json.loads(line) for line in group_path.read_text().splitlines()]
    group_rows[0]["image"] = unique_rows[0]["image"]
    group_rows[2]["image"] = unique_rows[1]["image"]
    group_path.write_text("".join(json.dumps(row) + "\n" for row in group_rows), encoding="utf-8")
    _rows(root, runner.MANIFESTS["rsicd"], ("a river", "a field"), source="RSICD")
    _json(root / runner.GROUP_DIR / "audit.json", {"caption_group_overlap": 0})
    baseline = runner.Trial(
        "baseline",
        root / f"outputs/{runner.BASELINE}/config.toml",
        root / f"outputs/{runner.BASELINE}",
    )
    _complete(baseline)
    (baseline.output / "train.log").write_text("baseline log not archived\n", encoding="utf-8")

    def config_loader(path):
        source = Path(path)
        return SimpleNamespace(
            source=source,
            experiment=SimpleNamespace(output_dir=Path("outputs") / source.stem, seed=11),
            train=SimpleNamespace(max_steps=1710),
            data=SimpleNamespace(
                train_manifest=train, val_manifest=root / runner.MANIFESTS["skyscript_unique"]
            ),
        )

    monkeypatch.setattr(runner, "load_config", config_loader)
    monkeypatch.setattr(
        runner,
        "required_paths",
        lambda config: [config.data.train_manifest, config.data.val_manifest],
    )
    calls = []

    def fake_execute(command, log, *, root):
        calls.append(command)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("fake subprocess\n", encoding="utf-8")
        if command[3] == "dinotxt_rs.cli.train":
            source = Path(command[command.index("--config") + 1])
            method = source.stem.split("_")[-3]
            _complete(_trial(root, method))
            return
        output = Path(command[command.index("--output") + 1])
        manifest = Path(command[command.index("--manifest") + 1])
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        config = Path(command[command.index("--config") + 1])
        dataset = next(key for key, value in runner.MANIFESTS.items() if manifest == root / value)
        assert ("--positive-definition" in command) == (dataset == "skyscript_group")
        counts = runner._pool_counts(root)[dataset]
        metrics = {
            direction: {"r1": 0.2, "r5": 0.4, "r10": 0.6, "mean_rank": 5.0, "median_rank": 3.0}
            for direction in ("image_to_text", "text_to_image")
        }
        metrics["mean_recall"] = 0.4
        report = {
            "split": "val",
            "tie_policy": runner.RETRIEVAL_TIE_POLICY,
            "task": {
                "skyscript_unique": "paired_image_text_global_retrieval",
                "skyscript_group": "caption_group_image_text_global_retrieval",
                "rsicd": "rsicd_image_text_retrieval",
            }[dataset],
            "manifest": {"path": str(manifest)},
            "counts": counts,
            "metrics": metrics,
            "model": {
                "model_config": str(config),
                "checkpoint": {
                    "step": 0 if checkpoint.stem == "step_0000000" else 1600,
                    "path": str(checkpoint),
                    "identity_check": {"status": "match", "blocking_mismatches": []},
                },
            },
        }
        if dataset == "skyscript_unique":
            report["positive_definition"] = "manifest_row_one_to_one"
        if dataset == "skyscript_group":
            report["positive_definition"] = "normalized_complete_caption_group"
            report["recall_definition"] = "fraction_of_queries_with_any_positive_in_top_k"
            metrics["image_to_text_group_balanced"] = {"r1": 0.3, "r5": 0.5, "r10": 0.7}
            metrics["group_balanced_mean_recall"] = 0.45
        _json(output, report)

    monkeypatch.setattr(runner, "_execute", fake_execute)
    return root, calls


def test_default_three_train_twenty_four_eval_archive_and_repeat_skip(tmp_path, monkeypatch):
    root, calls = _fake_project(tmp_path, monkeypatch)
    archive = runner.run_pipeline(root, sys.executable)
    assert len(calls) == 27
    assert [Path(command[5]).stem for command in calls[:3]] == [
        _trial(root, method).config.stem for method in runner.METHODS
    ]
    assert all("--resume" not in command for command in calls[:3])
    assert len(list((root / runner.COMPARISON_DIR).rglob("*.json"))) == 24
    with tarfile.open(archive) as package:
        names = package.getnames()
    assert not any(name.endswith((".pt", ".part")) for name in names)
    assert str(runner.GROUP_DIR / "audit.json") in names
    assert f"outputs/{runner.BASELINE}/training_summary.json" in names
    assert f"outputs/{runner.BASELINE}/train.log" not in names
    assert not archive.with_name(archive.name + ".part").exists()
    calls.clear()
    runner.run_pipeline(root, sys.executable)
    assert calls == []


def test_train_only_ignores_baseline_and_extra_eval_assets_then_can_evaluate(tmp_path, monkeypatch):
    root, calls = _fake_project(tmp_path, monkeypatch)
    shutil.rmtree(root / f"outputs/{runner.BASELINE}")
    (root / runner.MANIFESTS["skyscript_group"]).unlink()
    (root / runner.MANIFESTS["rsicd"]).unlink()
    runner.run_pipeline(root, sys.executable, mode="train-only")
    assert len(calls) == 3
    assert not (root / runner.COMPARISON_DIR).exists()


def test_resume_latest_is_explicit_and_completed_run_skipped(tmp_path, monkeypatch):
    root, calls = _fake_project(tmp_path, monkeypatch)
    rotate = _trial(root, "rotate")
    _complete(rotate)
    (rotate.output / "training_summary.json").unlink()
    _complete(_trial(root, "multipos"))
    runner.run_pipeline(root, sys.executable, mode="train-only")
    assert len(calls) == 2
    assert "--resume" in calls[0]
    assert calls[0][-1] == str(rotate.output / "latest.pt")
    assert "maskpos" in calls[1][5]


@pytest.mark.parametrize("problem", ["snapshot", "nonempty", "missing_manifest", "terminal"])
def test_all_preflight_finishes_before_any_training(tmp_path, monkeypatch, problem):
    root, calls = _fake_project(tmp_path, monkeypatch)
    last = _trial(root, "maskpos")
    if problem == "snapshot":
        last.output.mkdir(parents=True)
        (last.output / "config.toml").write_text("different", encoding="utf-8")
    elif problem == "nonempty":
        last.output.mkdir(parents=True)
        (last.output / "train.log").write_text("interrupted before checkpoint", encoding="utf-8")
    elif problem == "missing_manifest":
        (root / runner.MANIFESTS["rsicd"]).unlink()
    else:
        _complete(last)
        path = last.output / "training_summary.json"
        value = json.loads(path.read_text())
        value["validation"]["last_step"] = 1600
        _json(path, value)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        runner.run_pipeline(root, sys.executable)
    assert calls == []


def test_failure_stops_pipeline_preserves_previous_archive(tmp_path, monkeypatch):
    root, calls = _fake_project(tmp_path, monkeypatch)
    archive = root / "outputs/sat_multi_positive_seed11_reports.tar.gz"
    archive.write_bytes(b"previous archive")

    def fail(command, log, *, root):
        calls.append(command)
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(runner, "_execute", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_pipeline(root, sys.executable)
    assert len(calls) == 1
    assert archive.read_bytes() == b"previous archive"


@pytest.mark.parametrize("corruption", ["checkpoint", "identity", "groups", "mean", "tie"])
def test_invalid_existing_report_blocks_without_overwrite_or_gpu(tmp_path, monkeypatch, corruption):
    root, calls = _fake_project(tmp_path, monkeypatch)
    runner.run_pipeline(root, sys.executable)
    calls.clear()
    path = root / runner.COMPARISON_DIR / "baseline/skyscript_group_best.json"
    value = json.loads(path.read_text())
    if corruption == "checkpoint":
        value["model"]["checkpoint"]["step"] = 17
    elif corruption == "identity":
        value["model"]["checkpoint"]["identity_check"]["blocking_mismatches"] = ["config"]
    elif corruption == "groups":
        value["counts"]["groups"] = 55
    elif corruption == "mean":
        value["metrics"]["mean_recall"] = 0.8
    else:
        value["tie_policy"] = "random"
    _json(path, value)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        runner.run_pipeline(root, sys.executable)
    assert calls == []
    assert path.read_bytes() == original


def test_evaluate_only_requires_completed_runs_and_never_trains(tmp_path, monkeypatch):
    root, calls = _fake_project(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="requires completed"):
        runner.run_pipeline(root, sys.executable, mode="evaluate-only")
    assert calls == []
    for method in runner.METHODS:
        _complete(_trial(root, method))
    runner.run_pipeline(root, sys.executable, mode="evaluate-only")
    assert len(calls) == 24
    assert all("dinotxt_rs.cli.train" not in command for command in calls)


@pytest.mark.parametrize(
    "problem",
    [
        "train_omitted_group",
        "train_missing_representative",
        "train_changed_caption",
        "train_val_ids",
        "val_omitted_group",
        "val_missing_representative",
        "val_changed_caption",
        "val_candidate_spelling",
    ],
)
def test_protocol_reference_preservation_and_holdout_checked_before_gpu(
    tmp_path, monkeypatch, problem
):
    root, calls = _fake_project(tmp_path, monkeypatch)
    path = (
        root / runner.GROUP_DIR / "train_grouped.jsonl"
        if problem.startswith("train")
        else root / runner.MANIFESTS["skyscript_group"]
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if problem.endswith("omitted_group"):
        rows.pop(2)
    elif problem.endswith("missing_representative"):
        rows.pop(0)
    elif problem.endswith("changed_caption"):
        rows[0]["caption"] = rows[0]["caption"].upper()
    elif problem == "train_val_ids":
        val = json.loads((root / runner.MANIFESTS["skyscript_unique"]).read_text().splitlines()[0])
        rows[0]["id"] = val["id"]
    else:
        rows[1]["caption"] = "Airport"
        rows[0], rows[1] = rows[1], rows[0]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.run_pipeline(root, sys.executable)
    assert calls == []


def test_archive_failure_preserves_previous_archive_and_removes_partial(tmp_path, monkeypatch):
    root, calls = _fake_project(tmp_path, monkeypatch)
    archive = runner.run_pipeline(root, sys.executable)
    previous = archive.read_bytes()
    calls.clear()

    def fail_archive(path, *args, **kwargs):
        Path(path).write_bytes(b"incomplete archive")
        raise OSError("archive write failed")

    monkeypatch.setattr(runner.tarfile, "open", fail_archive)
    with pytest.raises(OSError, match="archive write failed"):
        runner.run_pipeline(root, sys.executable)
    assert archive.read_bytes() == previous
    assert not archive.with_name(archive.name + ".part").exists()
    assert calls == []


def test_real_subprocess_failure_logs_stderr_and_preserves_status(tmp_path):
    log = tmp_path / "run.log"
    command = [sys.executable, "-c", "import sys; print('failed', file=sys.stderr); sys.exit(7)"]
    with pytest.raises(subprocess.CalledProcessError) as failure:
        runner._execute(command, log, root=tmp_path)
    assert failure.value.returncode == 7
    assert "failed" in log.read_text()
