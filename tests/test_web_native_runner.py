from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

from dinotxt_rs.cli import evaluate_rsicd, evaluate_skyscript
from dinotxt_rs.config import load_config
from dinotxt_rs.evaluation.retrieval import RETRIEVAL_TIE_POLICY
from tools import run_sat_full_image_epoch as shared
from tools import run_web_native as runner

ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = "skyscript_web_adapter_textlora_maskpos_fullimage1epoch_seed11"


def _report(config, dataset: str, manifest: Path, counts: dict[str, int]) -> dict:
    recalls = {"r1": 0.1, "r5": 0.2, "r10": 0.3,
               "mean_rank": 5.0, "median_rank": 3.0}
    metrics = {
        "image_to_text": dict(recalls), "text_to_image": dict(recalls),
        "mean_recall": 0.2,
    }
    value = {
        "task": {
            "skyscript_unique": "paired_image_text_global_retrieval",
            "skyscript_group": "caption_group_image_text_global_retrieval",
            "rsicd": "rsicd_image_text_retrieval",
        }[dataset],
        "split": "val", "tie_policy": RETRIEVAL_TIE_POLICY,
        "manifest": {"path": str(manifest)}, "counts": dict(counts),
        "metrics": metrics,
        "model": {
            "model_config": str(config.source), "backbone_domain": "web",
            "model_variant": "official_without_image_adapter", "checkpoint": None,
            "trainable_parameters": {"trainable": 0},
            **{
                name: {
                    "path": str(
                        (config.source.parent.parent / getattr(config.model, name)).resolve()
                    )
                }
                for name in ("backbone_weights", "dinotxt_weights", "bpe_vocab")
            },
        },
    }
    if dataset == "skyscript_unique":
        value["positive_definition"] = "manifest_row_one_to_one"
    if dataset == "skyscript_group":
        value["positive_definition"] = "normalized_complete_caption_group"
        value["recall_definition"] = "fraction_of_queries_with_any_positive_in_top_k"
        metrics["image_to_text_group_balanced"] = {"r1": 0.15, "r5": 0.25, "r10": 0.35}
        metrics["group_balanced_mean_recall"] = 0.225
    return value


def test_variants_keep_full_image_budget_and_sat_identity_diff_is_exactly_four_fields():
    sat_path = ROOT / "configs/skyscript_sat_adapter_textlora_maskpos_fullimage1epoch_seed11.toml"
    web_path = ROOT / "configs" / f"{MAIN_CONFIG}.toml"
    with sat_path.open("rb") as handle:
        sat = tomllib.load(handle)
    with web_path.open("rb") as handle:
        web = tomllib.load(handle)
    for config in (sat, web):
        config["experiment"].pop("name")
        config["experiment"].pop("output_dir")
        config["model"].pop("backbone_domain")
        config["model"].pop("backbone_weights")
    assert web == sat

    assert runner.VARIANTS == {
        "adapter_lora": MAIN_CONFIG,
        "adapter_only": "skyscript_web_adapteronly_maskpos_fullimage1epoch_seed11",
        "head_lora": "skyscript_web_head_textlora_maskpos_fullimage1epoch_seed11",
    }
    for method, stem in runner.VARIANTS.items():
        config = load_config(ROOT / "configs" / f"{stem}.toml")
        assert config.model.backbone_domain == "web"
        assert config.model.backbone_weights == runner.WEB_WEIGHTS
        assert config.data == load_config(web_path).data
        assert (config.train.batch_size, config.train.gradient_accumulation) == (16, 4)
        assert (config.train.max_steps, config.train.warmup_steps) == (5269, 527)
        assert config.train.queue_size == 0
        assert config.train.image_epochs == 1
        assert config.train.validation_at_start
        assert config.train.validation_every == config.train.checkpoint_every == 200
        assert config.train.checkpoint_policy == "rolling"
        assert config.train.contrastive_objective == "mask_same_caption"
        assert config.data.caption_sampling == "image_epoch"
        assert config.data.images_per_caption == 2
        assert not config.data.train_augmentation
        if method == "adapter_lora":
            assert config.model.image_adapter_bottleneck == 256
            assert config.model.text_lora_rank == 8
            assert config.model.text_lora_include_projection
            assert not config.model.train_vision_head
            assert config.train.learning_rate == config.train.image_adapter_learning_rate == 1e-4
        elif method == "adapter_only":
            assert config.model.image_adapter_bottleneck == 256
            assert config.model.text_lora_rank == 0
            assert not config.model.text_lora_include_projection
            assert not config.model.train_vision_head
            assert config.train.learning_rate == config.train.image_adapter_learning_rate == 1e-4
        else:
            assert config.model.image_adapter_bottleneck == 0
            assert config.model.text_lora_rank == 8
            assert config.model.text_lora_include_projection
            assert config.model.train_vision_head
            assert config.train.vision_head_learning_rate == 1e-5
            assert config.train.learning_rate == 1e-4


def _pipeline_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    for stem in (MAIN_CONFIG, *list(runner.VARIANTS.values())[1:]):
        (root / "configs" / f"{stem}.toml").write_bytes(
            (ROOT / "configs" / f"{stem}.toml").read_bytes()
        )
    sat_config = "skyscript_sat_adapter_textlora_maskpos_fullimage1epoch_seed11.toml"
    (root / "configs" / sat_config).write_bytes(
        (ROOT / "configs" / sat_config).read_bytes()
    )
    return root


def test_default_and_controls_pipeline_order_baseline_before_training(tmp_path, monkeypatch):
    root = _pipeline_project(tmp_path)
    events = []
    monkeypatch.setattr(shared, "run_pipeline", lambda *a, **kw: events.append(
        ("shared", kw["mode"], [trial.method for trial in kw["trials"]])
    ))
    monkeypatch.setattr(runner, "_official_baseline", lambda *a: events.append(("baseline",)))

    runner.run_pipeline(root, sys.executable)
    assert events == [
        ("shared", "preflight-only", ["adapter_lora"]),
        ("baseline",),
        ("shared", "all", ["adapter_lora"]),
    ]

    events.clear()
    runner.run_pipeline(root, sys.executable, include_controls=True)
    assert events == [
        ("shared", "preflight-only", ["adapter_lora", "adapter_only", "head_lora"]),
        ("baseline",),
        ("shared", "all", ["adapter_lora", "adapter_only", "head_lora"]),
    ]


@pytest.mark.parametrize("mode,expected", [
    ("train-only", [("shared", "train-only")]),
    ("preflight-only", [("shared", "preflight-only")]),
])
def test_train_and_preflight_modes_skip_baseline(tmp_path, monkeypatch, mode, expected):
    root = _pipeline_project(tmp_path)
    events = []

    def fake_shared(*args, **kwargs):
        events.append(("shared", kwargs["mode"]))

    monkeypatch.setattr(shared, "run_pipeline", fake_shared)
    monkeypatch.setattr(runner, "_official_baseline", lambda *a: events.append(("baseline",)))
    runner.run_pipeline(root, sys.executable, mode=mode)
    assert events == expected


def test_baseline_only_and_evaluate_only_order_and_completion_guard(tmp_path, monkeypatch):
    root = _pipeline_project(tmp_path)
    events = []
    monkeypatch.setattr(shared, "run_pipeline", lambda *a, **kw: events.append(
        ("shared", kw["mode"])
    ))
    monkeypatch.setattr(runner, "_official_baseline", lambda *a: events.append(("baseline",)))
    monkeypatch.setattr(shared, "_package", lambda *a, **kw: "archive")
    assert runner.run_pipeline(root, sys.executable, mode="baseline-only") == "archive"
    assert events == [("shared", "preflight-only"), ("baseline",)]

    events.clear()
    audit = root / shared.GROUP_DIR / "audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"splits": {"train": {"records": 337199}}}), encoding="utf-8")
    monkeypatch.setattr(shared, "full_training_state", lambda *a, **kw: "complete")
    runner.run_pipeline(root, sys.executable, mode="evaluate-only")
    assert events == [("shared", "preflight-only"), ("baseline",), ("shared", "evaluate-only")]

    events.clear()
    monkeypatch.setattr(shared, "full_training_state", lambda *a, **kw: "missing")
    with pytest.raises(RuntimeError, match="requires completed training"):
        runner.run_pipeline(root, sys.executable, mode="evaluate-only")
    assert events == [("shared", "preflight-only")]


def test_official_baseline_runs_three_official_evaluations_and_caches_identity(
    tmp_path, monkeypatch,
):
    root = _pipeline_project(tmp_path)
    config = load_config(root / "configs" / f"{MAIN_CONFIG}.toml")
    original_manifests = runner.shared.MANIFESTS
    manifests = {
        name: Path("manifests") / f"{name}.jsonl" for name in original_manifests
    }
    monkeypatch.setattr(shared, "MANIFESTS", manifests)
    counts = {name: {"images": 2, "texts": 3} for name in manifests}
    monkeypatch.setattr(shared, "_pool_counts", lambda root: counts)
    calls = []

    def fake_execute(command, log, *, root):
        calls.append(command)
        dataset = next(name for name, relative in manifests.items()
                       if str(root / relative) == command[command.index("--manifest") + 1])
        report = _report(config, dataset, root / manifests[dataset], counts[dataset])
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(shared, "_execute", fake_execute)
    runner._official_baseline(root, sys.executable, config)
    assert len(calls) == 3
    assert all("--official" in call for call in calls)
    assert all("--checkpoint" not in call and "--training-output" not in call for call in calls)
    assert (root / runner.REPORT_DIR / "official/config.toml").read_bytes() == (
        config.source.read_bytes()
    )
    assert (root / runner.REPORT_DIR / "official/summary.json").exists()

    calls.clear()
    runner._official_baseline(root, sys.executable, config)
    assert calls == []


@pytest.mark.parametrize("mutation", ["domain", "count", "task", "checkpoint", "trainable"])
def test_official_baseline_identity_mismatches_are_rejected(tmp_path, mutation):
    config = load_config(ROOT / "configs" / f"{MAIN_CONFIG}.toml")
    manifest = ROOT / "assets/data/manifests" / (
        "skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
    )
    counts = {"images": 2, "texts": 3}
    value = _report(config, "skyscript_unique", manifest, counts)
    if mutation == "domain":
        value["model"]["backbone_domain"] = "sat"
    elif mutation == "count":
        value["counts"]["images"] += 1
    elif mutation == "task":
        value["task"] = "rsicd_image_text_retrieval"
    elif mutation == "checkpoint":
        value["model"]["checkpoint"] = {"path": "checkpoint.pt"}
    else:
        value["model"]["trainable_parameters"]["trainable"] = 1
    report = tmp_path / "report.json"
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected native official reference identity"):
        runner.validate_official_report(report, config, "skyscript_unique", manifest, counts)


@pytest.mark.parametrize("module", [evaluate_skyscript, evaluate_rsicd])
def test_official_cli_uses_official_loader_without_gpu(tmp_path, monkeypatch, module):
    config_path = ROOT / "configs" / f"{MAIN_CONFIG}.toml"
    output = tmp_path / "report.json"
    model_marker = object()
    calls = []
    # Evaluation is replaced after model construction so the test never initializes a device.
    evaluator_names = [name for name in vars(module) if name.startswith("evaluate_")]
    for name in evaluator_names:
        monkeypatch.setattr(module, name, lambda *args, **kwargs: {"model": "fake"})
    def load_official(config):
        calls.append(config)
        return model_marker

    monkeypatch.setattr(module, "load_official_reference_model", load_official)
    monkeypatch.setattr(module, "write_json_atomic", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(config_path), "--manifest",
                                      str(tmp_path / "manifest.jsonl"), "--output", str(output),
                                      "--official"])
    module.main()
    assert len(calls) == 1
    assert calls[0].model.backbone_domain == "web"


@pytest.mark.parametrize("module", [evaluate_skyscript, evaluate_rsicd])
@pytest.mark.parametrize("conflict", ["--checkpoint", "--training-output"])
def test_official_cli_rejects_checkpoint_and_training_output_conflicts(
    monkeypatch, module, conflict, tmp_path,
):
    argv = ["evaluate", "--config", "config.toml", "--manifest", "manifest.jsonl",
            "--output", str(tmp_path / "report.json"), "--official", conflict, "value"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        module.parse_args()
    assert error.value.code == 2
