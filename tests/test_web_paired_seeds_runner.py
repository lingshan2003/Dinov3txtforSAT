from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dinotxt_rs.config import load_config
from tools import run_sat_full_image_epoch as shared
from tools import run_web_adapter_control as control
from tools import run_web_native as native
from tools import run_web_paired_seeds as runner
from tools.run_sat_multi_positive import Trial

ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "paired project"
    (root / "configs").mkdir(parents=True)
    stems = set(native.VARIANTS.values())
    stems.add("skyscript_sat_adapter_textlora_maskpos_fullimage1epoch_seed11")
    stems.update(
        f"{native.VARIANTS[method].replace('seed11', f'seed{seed}')}"
        for method in runner.METHODS
        for seed in runner.SEEDS[1:]
    )
    for stem in stems:
        source = ROOT / "configs" / f"{stem}.toml"
        (root / "configs" / source.name).write_bytes(source.read_bytes())
    return root


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _states(monkeypatch, values: dict[tuple[int, str], str]) -> None:
    def state(root, trial):
        seed = int(trial.config.stem.rsplit("seed", 1)[1])
        return values.get((seed, trial.method), "fresh")

    monkeypatch.setattr(control, "_training_state", state)


def test_seed_trials_preserve_protocol_and_reject_seed_or_recipe_contamination(tmp_path):
    root = _project(tmp_path)
    assert runner.SEEDS == (11, 23, 47)
    assert runner.METHODS == ("adapter_lora", "adapter_only")
    for seed in runner.SEEDS:
        trials = runner.validated_trials(root, seed)
        assert [trial.method for trial in trials] == list(runner.METHODS)
        for trial in trials:
            config = load_config(trial.config)
            assert config.experiment.seed == seed
            assert trial.output == (root / config.experiment.output_dir).resolve()
            assert "seed" not in trial.method

    bad_seed = (
        root / "configs" / "skyscript_web_adapter_textlora_maskpos_fullimage1epoch_seed47.toml"
    )
    bad_seed.write_text(bad_seed.read_text().replace("seed = 47", "seed = 23", 1))
    with pytest.raises(ValueError, match="change only name/output_dir/seed"):
        runner.validated_trials(root, 47)

    bad_recipe = root / "configs" / "skyscript_web_adapteronly_maskpos_fullimage1epoch_seed23.toml"
    bad_recipe.write_text(
        bad_recipe.read_text().replace("text_lora_rank = 0", "text_lora_rank = 8", 1)
    )
    with pytest.raises(ValueError, match="change only name/output_dir/seed"):
        runner.validated_trials(root, 23)


def test_default_preflights_every_seed_before_baseline_or_training(tmp_path, monkeypatch):
    root = _project(tmp_path)
    _states(monkeypatch, {(11, method): "complete" for method in runner.METHODS})
    events = []

    def delegate(root, python, *, mode, **kwargs):
        events.append(
            ("shared", kwargs["expected_seed"], mode, kwargs["report_dir"], kwargs["series"])
        )

    monkeypatch.setattr(shared, "run_pipeline", delegate)
    monkeypatch.setattr(shared, "_pool_counts", lambda root: {})
    monkeypatch.setattr(
        runner, "_validate_seed11_sources", lambda *args: events.append(("validate",))
    )
    monkeypatch.setattr(native, "_official_baseline", lambda *args: events.append(("baseline",)))
    monkeypatch.setattr(runner, "_copy_cached_reports", lambda *args: events.append(("copy",)))
    monkeypatch.setattr(runner, "_write_summary", lambda *args: events.append(("summary",)))
    monkeypatch.setattr(shared, "_package", lambda *args, **kwargs: Path("archive.tar.gz"))

    assert runner.run_pipeline(root, sys.executable) == Path("archive.tar.gz")
    preflight_indices = [
        i for i, event in enumerate(events) if event[0] == "shared" and event[2] == "preflight-only"
    ]
    gpu_indices = [
        i for i, event in enumerate(events) if event[0] == "shared" and event[2] == "all"
    ]
    assert len(preflight_indices) == len(gpu_indices) == 3
    assert max(preflight_indices) < min(gpu_indices)
    assert [events[i][1] for i in preflight_indices] == list(runner.SEEDS)
    assert [events[i][1] for i in gpu_indices] == list(runner.SEEDS)
    assert all(
        events[i][3] == runner.REPORT_DIR / f"seed{seed}"
        and events[i][4] == f"{runner.SERIES}_seed{seed}"
        for i, seed in zip(gpu_indices, runner.SEEDS, strict=True)
    )
    assert all(events[i][0] != "baseline" for i in range(max(preflight_indices) + 1))


def test_preflight_failure_never_starts_any_seed_training(tmp_path, monkeypatch):
    root = _project(tmp_path)
    _states(monkeypatch, {(11, method): "complete" for method in runner.METHODS})
    events = []

    def delegate(root, python, *, mode, **kwargs):
        seed = kwargs["expected_seed"]
        events.append((mode, seed))
        if mode == "preflight-only" and seed == 47:
            raise ValueError("seed47 input failure")

    monkeypatch.setattr(shared, "run_pipeline", delegate)
    with pytest.raises(ValueError, match="seed47 input failure"):
        runner.run_pipeline(root, sys.executable)
    assert events == [("preflight-only", 11), ("preflight-only", 23), ("preflight-only", 47)]


def test_preflight_only_has_no_cache_or_report_writes(tmp_path, monkeypatch):
    root = _project(tmp_path)
    _states(monkeypatch, {(11, method): "complete" for method in runner.METHODS})
    calls = []
    monkeypatch.setattr(
        shared, "run_pipeline", lambda *a, **kw: calls.append((kw["mode"], kw["expected_seed"]))
    )
    monkeypatch.setattr(shared, "_pool_counts", lambda root: {})
    monkeypatch.setattr(runner, "_validate_seed11_sources", lambda *a: None)
    monkeypatch.setattr(native, "_official_baseline", lambda *a: pytest.fail("baseline"))
    monkeypatch.setattr(runner, "_copy_cached_reports", lambda *a: pytest.fail("cache copy"))
    monkeypatch.setattr(runner, "_write_summary", lambda *a: pytest.fail("summary"))

    assert runner.run_pipeline(root, sys.executable, mode="preflight-only") is None
    assert calls == [("preflight-only", seed) for seed in runner.SEEDS]
    assert not (root / runner.REPORT_DIR).exists()
    assert not (root / "outputs").exists()


def test_seed11_completion_gate_precedes_preflight(tmp_path, monkeypatch):
    root = _project(tmp_path)
    _states(monkeypatch, {})
    calls = []
    monkeypatch.setattr(shared, "run_pipeline", lambda *a, **kw: calls.append(kw))
    with pytest.raises(RuntimeError, match="requires completed seed11"):
        runner.run_pipeline(root, sys.executable)
    assert calls == []


def test_cached_seed11_report_identity_is_checked_before_copy(tmp_path, monkeypatch):
    root = _project(tmp_path)
    trials = runner.validated_trials(root, 11)
    counts = {dataset: {"images": 2, "texts": 3} for dataset in shared.MANIFESTS}
    dataset = next(iter(shared.MANIFESTS))
    source = root / control.REPORT_DIR / trials[0].method / f"{dataset}_latest.json"
    _json(source, {})
    seen = []

    def reject(path, *args):
        seen.append(path)
        raise ValueError("cached seed11 report identity mismatch")

    monkeypatch.setattr(shared, "validate_report", reject)
    with pytest.raises(ValueError, match="cached seed11 report identity mismatch"):
        runner._copy_cached_reports(root, trials, counts)
    target = root / runner.REPORT_DIR / "seed11" / trials[0].method / source.name
    assert seen == [source]
    assert not target.exists()


def test_train_only_skips_baseline_cache_and_aggregate(tmp_path, monkeypatch):
    root = _project(tmp_path)
    _states(monkeypatch, {(11, method): "complete" for method in runner.METHODS})
    calls = []
    monkeypatch.setattr(
        shared, "run_pipeline", lambda *a, **kw: calls.append((kw["mode"], kw["expected_seed"]))
    )
    monkeypatch.setattr(native, "_official_baseline", lambda *a: pytest.fail("baseline"))
    monkeypatch.setattr(runner, "_copy_cached_reports", lambda *a: pytest.fail("cache copy"))
    monkeypatch.setattr(runner, "_write_summary", lambda *a: pytest.fail("summary"))
    monkeypatch.setattr(shared, "_package", lambda *a, **kw: Path("archive.tar.gz"))

    runner.run_pipeline(root, sys.executable, mode="train-only")
    assert calls == (
        [("preflight-only", seed) for seed in runner.SEEDS]
        + [("train-only", seed) for seed in runner.SEEDS]
    )


def test_evaluate_only_requires_every_seed_complete_then_evaluates_all(tmp_path, monkeypatch):
    root = _project(tmp_path)
    values = {(11, method): "complete" for method in runner.METHODS}
    values[(23, "adapter_only")] = "fresh"
    _states(monkeypatch, values)
    calls = []
    monkeypatch.setattr(
        shared, "run_pipeline", lambda *a, **kw: calls.append((kw["mode"], kw["expected_seed"]))
    )
    with pytest.raises(RuntimeError, match="evaluate-only requires completed training"):
        runner.run_pipeline(root, sys.executable, mode="evaluate-only")
    assert calls == [("preflight-only", 11)]

    calls.clear()
    values.update({(seed, method): "complete" for seed in (23, 47) for method in runner.METHODS})
    monkeypatch.setattr(shared, "_pool_counts", lambda root: {})
    monkeypatch.setattr(runner, "_validate_seed11_sources", lambda *a: None)
    monkeypatch.setattr(native, "_official_baseline", lambda *a: None)
    monkeypatch.setattr(runner, "_copy_cached_reports", lambda *a: None)
    monkeypatch.setattr(runner, "_write_summary", lambda *a: None)
    monkeypatch.setattr(shared, "_package", lambda *a, **kw: Path("archive.tar.gz"))
    result = runner.run_pipeline(root, sys.executable, mode="evaluate-only")
    assert result == Path("archive.tar.gz")
    assert calls == (
        [("preflight-only", seed) for seed in runner.SEEDS]
        + [("evaluate-only", seed) for seed in runner.SEEDS]
    )


def test_paired_summary_uses_seedwise_delta_sample_std(tmp_path, monkeypatch):
    root = _project(tmp_path)
    grid = {seed: runner.validated_trials(root, seed) for seed in runner.SEEDS}
    counts = {dataset: {"images": 2, "texts": 3} for dataset in shared.MANIFESTS}
    monkeypatch.setattr(shared, "_pool_counts", lambda root: counts)
    monkeypatch.setattr(native, "validate_official_report", lambda *a: None)
    monkeypatch.setattr(shared, "validate_report", lambda *a: None)

    metric_by_seed = {11: 0.1, 23: 0.4, 47: 0.8}
    adapter_by_seed = {11: 0.1, 23: 0.3, 47: 0.7}

    def metrics(value):
        direction = {"r1": value, "r5": value + 0.01, "r10": value + 0.02}
        return {
            "mean_recall": value,
            "image_to_text": dict(direction),
            "text_to_image": dict(direction),
            "group_balanced_mean_recall": value,
        }

    baseline_model = {
        key: {"path": f"assets/{key}"}
        for key in ("backbone_weights", "dinotxt_weights", "bpe_vocab")
    }
    for dataset in shared.MANIFESTS:
        _json(
            root / runner.REPORT_DIR / "official" / f"{dataset}.json",
            {
                "metrics": metrics(0.2),
                "manifest": {"path": "same"},
                "counts": counts[dataset],
                "tie_policy": "same",
                "positive_definition": "same",
                "recall_definition": "same",
                "model": baseline_model,
            },
        )
    for seed, trials in grid.items():
        for trial in trials:
            _json(trial.output / "training_summary.json", {"seed": seed})
            value = (metric_by_seed if trial.method == "adapter_lora" else adapter_by_seed)[seed]
            for dataset in shared.MANIFESTS:
                for tag in shared.TAGS:
                    _json(
                        root
                        / runner.REPORT_DIR
                        / f"seed{seed}"
                        / trial.method
                        / f"{dataset}_{tag}.json",
                        {
                            "metrics": metrics(value),
                            "manifest": {"path": "same"},
                            "counts": counts[dataset],
                            "tie_policy": "same",
                            "positive_definition": "same",
                            "recall_definition": "same",
                            "model": baseline_model,
                        },
                    )

    runner._write_summary(root, grid)
    result = json.loads((root / runner.REPORT_DIR / "comparison.json").read_text())
    aggregate = result["aggregates"]["latest"]["skyscript_unique"]
    paired = aggregate["paired_lora_minus_adapter_only_pp"]["mean_recall"]
    expected = [100 * (metric_by_seed[s] - adapter_by_seed[s]) for s in runner.SEEDS]
    assert paired["values"] == expected
    assert paired["sample_std"] == pytest.approx(__import__("statistics").stdev(expected))
    separate = aggregate["methods_percent"]
    assert paired["sample_std"] < separate["adapter_lora"]["mean_recall"]["sample_std"]

    report = root / runner.REPORT_DIR / "seed23" / "adapter_only" / "rsicd_latest.json"
    value = json.loads(report.read_text())
    value["counts"]["images"] += 1
    _json(report, value)
    with pytest.raises(ValueError, match="protocols/assets differ"):
        runner._write_summary(root, grid)


@pytest.mark.parametrize(
    "initial_state,expected_resume", [("fresh", False), ("resume", True), ("complete", None)]
)
def test_shared_runner_accepts_seed23_and_runs_train_resume_or_skip(
    tmp_path,
    monkeypatch,
    initial_state,
    expected_resume,
):
    from test_full_image_epoch_runner import _fake_project

    root, calls, _ = _fake_project(tmp_path, monkeypatch)
    source = root / "configs" / "tiny_adapter_lora_fullimage_seed23.toml"
    source.write_text("synthetic seed23 config\n", encoding="utf-8")
    original_loader = shared.load_config

    def load_seed23(path):
        config = original_loader(path)
        experiment = SimpleNamespace(seed=23, output_dir=Path("outputs") / source.stem)
        config_copy = SimpleNamespace(**vars(config))
        config_copy.experiment = experiment
        return config_copy

    monkeypatch.setattr(shared, "load_config", load_seed23)
    trial = Trial("adapter_lora", source, root / "outputs" / source.stem)
    state_calls = []

    def full_state(*args, **kwargs):
        state_calls.append(True)
        return initial_state if len(state_calls) == 1 else "complete"

    monkeypatch.setattr(shared, "full_training_state", full_state)
    if initial_state == "complete":
        trial.output.mkdir(parents=True)
        _json(trial.output / "training_summary.json", {"completed": True})

    if initial_state != "complete":

        def execute(command, log, *, root):
            calls.append((command, log))
            trial.output.mkdir(parents=True, exist_ok=True)
            _json(trial.output / "training_summary.json", {"completed": True})

        monkeypatch.setattr(shared, "_execute", execute)

    shared.run_pipeline(
        root,
        sys.executable,
        mode="train-only",
        trials=[trial],
        expected_seed=23,
        report_dir=runner.REPORT_DIR / "seed23",
        series=f"{runner.SERIES}_seed23",
        require_matching_recipes=False,
    )
    train_calls = [call for call in calls if call[0][3] == "dinotxt_rs.cli.train"]
    assert len(train_calls) == (0 if initial_state == "complete" else 1)
    if initial_state == "resume":
        assert "--resume" in train_calls[0][0]


def test_shared_runner_still_rejects_non11_by_default(tmp_path, monkeypatch):
    from test_full_image_epoch_runner import _fake_project

    root, _, _ = _fake_project(tmp_path, monkeypatch)
    source = root / "configs" / "seed23.toml"
    source.write_text("synthetic\n", encoding="utf-8")
    original_loader = shared.load_config

    def load_seed23(path):
        config = original_loader(path)
        config_copy = SimpleNamespace(**vars(config))
        config_copy.experiment = SimpleNamespace(
            seed=23,
            output_dir=config.experiment.output_dir,
        )
        return config_copy

    monkeypatch.setattr(shared, "load_config", load_seed23)
    trial = Trial("adapter_lora", source, root / "outputs" / source.stem)
    with pytest.raises(ValueError, match="Unexpected full-image protocol"):
        shared.run_pipeline(root, sys.executable, mode="preflight-only", trials=[trial])
