from __future__ import annotations

import json
import math
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from test_full_image_epoch_runner import _fake_project, _json, _trial
from test_joint_alignment import PairedModel

from dinotxt_rs.config import load_config
from dinotxt_rs.models import (
    add_image_embedding_adapter,
    add_text_lora,
    configure_trainable_parameters,
    optimizer_parameter_groups,
)
from tools import run_sat_maskpos_next as runner

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = load_config(
    ROOT / "configs/skyscript_sat_adapter_textlora_maskpos_fullimage1epoch_seed11.toml"
)


@pytest.fixture
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    ("variant", "batch", "accumulation", "epochs", "steps"),
    [
        ("batch32_1epoch", 32, 2, 1, 5269),
        ("epochs3_batch16", 16, 4, 3, 15807),
        ("head_lora_1epoch", 32, 2, 1, 5269),
        ("fulltext4_1epoch", 32, 2, 1, 5269),
    ],
)
def test_configs_keep_protocol_and_define_full_image_budgets(
    variant,
    batch,
    accumulation,
    epochs,
    steps,
):
    name, _, _, _, _, _ = runner.VARIANTS[variant]
    config = load_config(ROOT / "configs" / f"{name}.toml")

    assert config.experiment.seed == 11
    assert config.experiment.name == name
    assert config.model.backbone_domain == "sat"
    assert config.model.backbone_weights == ORIGINAL.model.backbone_weights
    assert config.model.dinotxt_weights == ORIGINAL.model.dinotxt_weights
    assert config.model.dinov3_repo == ORIGINAL.model.dinov3_repo
    assert config.model.bpe_vocab == ORIGINAL.model.bpe_vocab
    assert config.data == ORIGINAL.data
    assert config.data.train_augmentation is False
    assert config.train.queue_size == 0
    assert config.train.contrastive_objective == "mask_same_caption"
    assert config.train.batch_size == batch
    assert config.train.gradient_accumulation == accumulation
    assert batch * accumulation == 64
    assert config.train.image_epochs == epochs
    microbatches = math.ceil(337_199 / batch)
    assert math.ceil(microbatches * epochs / accumulation) == steps
    assert config.train.max_steps == steps


def test_head_lora_updates_only_the_configured_head_and_adapters():
    name = runner.VARIANTS["head_lora_1epoch"][0]
    config = load_config(ROOT / "configs" / f"{name}.toml")

    assert config.model.text_last_k == 0
    assert config.model.train_vision_head is True
    assert config.model.train_text_projection is False
    assert config.model.train_logit_scale is False
    assert config.model.image_adapter_bottleneck == 256
    assert config.model.text_lora_rank == 8
    assert config.model.text_lora_include_projection is True
    assert config.train.image_adapter_learning_rate == pytest.approx(1e-4)
    assert config.train.vision_head_learning_rate == pytest.approx(1e-5)
    assert config.train.text_projection_learning_rate == pytest.approx(1e-4)
    assert config.train.text_backbone_learning_rate == pytest.approx(1e-4)


def test_strong_variant_updates_last_four_text_blocks_and_projection_without_lora():
    name = runner.VARIANTS["fulltext4_1epoch"][0]
    config = load_config(ROOT / "configs" / f"{name}.toml")

    assert config.model.text_last_k == 4
    assert config.model.train_vision_head is True
    assert config.model.train_text_projection is True
    assert config.model.train_logit_scale is False
    assert config.model.text_lora_rank == 0
    assert config.model.text_lora_include_projection is False
    assert config.train.text_backbone_learning_rate == pytest.approx(5e-6)
    assert config.train.text_projection_learning_rate == pytest.approx(1e-5)
    assert config.train.vision_head_learning_rate == pytest.approx(1e-5)
    assert config.train.image_adapter_learning_rate == pytest.approx(1e-4)


@pytest.mark.parametrize(
    ("include_strong", "expected"),
    [
        (False, ("batch32_1epoch", "epochs3_batch16", "head_lora_1epoch")),
        (True, tuple(runner.VARIANTS)),
    ],
)
def test_pipeline_selects_variants_and_delegates_shared_options(
    monkeypatch,
    include_strong,
    expected,
):
    calls = []

    def delegate(root, python, **kwargs):
        calls.append((root, python, kwargs))
        return Path("delegated.tar.gz")

    monkeypatch.setattr(runner.shared, "run_pipeline", delegate)
    result = runner.run_pipeline(
        ROOT, "python-test", mode="train-only", include_strong=include_strong
    )

    assert result == Path("delegated.tar.gz")
    assert len(calls) == 1
    root, python, options = calls[0]
    assert root == ROOT.resolve()
    assert python == "python-test"
    assert options == {
        "mode": "train-only",
        "trials": [
            runner.Trial(
                key,
                root / "configs" / f"{runner.VARIANTS[key][0]}.toml",
                root / "outputs" / runner.VARIANTS[key][0],
            )
            for key in expected
        ],
        "report_dir": runner.REPORT_DIR,
        "series": "sat_maskpos_next_seed11",
        "require_matching_recipes": False,
    }


def test_all_selected_configs_are_preflighted_before_shared_runner(monkeypatch):
    original_loader = runner.load_config
    shared_calls = []

    def invalid_strong_config(path):
        config = original_loader(path)
        if Path(path).name == f"{runner.VARIANTS['fulltext4_1epoch'][0]}.toml":
            return replace(config, experiment=replace(config.experiment, seed=23))
        return config

    monkeypatch.setattr(runner, "load_config", invalid_strong_config)
    monkeypatch.setattr(
        runner.shared, "run_pipeline", lambda *args, **kwargs: shared_calls.append(args)
    )

    with pytest.raises(ValueError, match="Unexpected full-maskpos follow-up protocol"):
        runner.run_pipeline(ROOT, "python-test", include_strong=True)
    assert shared_calls == []


def test_epoch_variants_keep_their_own_report_series_and_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner.shared,
        "run_pipeline",
        lambda root, python, **kwargs: calls.append(kwargs) or None,
    )

    runner.run_pipeline(ROOT, "python-test")

    assert calls[0]["report_dir"] == Path("outputs/sat_maskpos_next_seed11")
    assert calls[0]["series"] == "sat_maskpos_next_seed11"
    selected = {trial.method: trial for trial in calls[0]["trials"]}
    assert selected["epochs3_batch16"].config.name == (
        "skyscript_sat_adapter_textlora_maskpos_fullimage3epoch_seed11.toml"
    )
    config = load_config(selected["epochs3_batch16"].config)
    assert (config.train.batch_size, config.train.gradient_accumulation) == (16, 4)
    assert (config.train.image_epochs, config.train.max_steps) == (3, 15807)


def test_shared_pipeline_runs_custom_three_epoch_trial_and_skips_it_when_complete(
    tmp_path,
    monkeypatch,
):
    from tools import run_sat_full_image_epoch as shared

    root, calls, _ = _fake_project(tmp_path, monkeypatch)
    original_loader = shared.load_config

    def three_epoch_config(path):
        config = original_loader(path)
        config.train.max_steps = 5
        config.train.image_epochs = 3
        return config

    monkeypatch.setattr(shared, "load_config", three_epoch_config)
    original_execute = shared._execute

    def three_epoch_execute(command, log, *, root):
        original_execute(command, log, root=root)
        if command[3] == "dinotxt_rs.cli.train":
            summary_path = _trial(root, "maskpos").output / "training_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["steps"] = summary["target_steps"] = 5
            summary["validation"]["last_step"] = 5
            summary["image_epochs"].update(
                target_epochs=3,
                completed_epochs=3,
                sample_exposures=27,
            )
            _json(summary_path, summary)
        else:
            report_path = Path(command[command.index("--output") + 1])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            tag = Path(command[command.index("--checkpoint") + 1]).stem
            report["model"]["checkpoint"]["step"] = {
                "step_0000000": 0,
                "best": 1,
                "latest": 5,
            }[tag]
            _json(report_path, report)

    monkeypatch.setattr(shared, "_execute", three_epoch_execute)
    trial = _trial(root, "maskpos")
    report_dir = Path("outputs/synthetic_maskpos_reports")
    archive = shared.run_pipeline(
        root,
        "python-test",
        trials=[trial],
        report_dir=report_dir,
        series="synthetic_maskpos_three_epochs",
        require_matching_recipes=False,
    )

    assert len(calls) == 10
    assert sum(command[3] == "dinotxt_rs.cli.train" for command in calls) == 1
    assert sum(command[3] != "dinotxt_rs.cli.train" for command in calls) == 9
    summary = json.loads((root / report_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["series"] == "synthetic_maskpos_three_epochs"
    assert summary["image_epochs"] == {"maskpos": 3}
    assert summary["trials"]["maskpos"]["image_epochs"]["completed_epochs"] == 3
    assert summary["trials"]["maskpos"]["image_epochs"]["sample_exposures"] == 27
    assert len(list((root / report_dir / "maskpos").glob("*.json"))) == 9
    assert archive == root / "outputs/synthetic_maskpos_three_epochs_reports.tar.gz"
    with tarfile.open(archive) as package:
        members = package.getnames()
    assert str(report_dir / "summary.json") in members
    assert not any(name.endswith((".pt", ".part")) for name in members)

    calls.clear()
    rerun_archive = shared.run_pipeline(
        root,
        "python-test",
        trials=[trial],
        report_dir=report_dir,
        series="synthetic_maskpos_three_epochs",
        require_matching_recipes=False,
    )
    assert rerun_archive == archive
    assert calls == []


@pytest.mark.parametrize("variant", ["head_lora_1epoch", "fulltext4_1epoch"])
def test_small_cpu_adamw_updates_configured_trainable_scope_only(
    variant,
    single_threaded_torch,
):
    torch.manual_seed(11)
    name, _, _, _, head, fulltext = runner.VARIANTS[variant]
    config = load_config(ROOT / "configs" / f"{name}.toml")
    model = PairedModel(depth=6)
    model = add_image_embedding_adapter(
        model, bottleneck_dim=config.model.image_adapter_bottleneck, embedding_dim=12
    )
    if config.model.text_lora_rank:
        model = add_text_lora(
            model,
            rank=config.model.text_lora_rank,
            alpha=config.model.text_lora_alpha,
            dropout=config.model.text_lora_dropout,
            include_projection=config.model.text_lora_include_projection,
        )
    configure_trainable_parameters(
        model,
        text_last_k=config.model.text_last_k,
        train_vision_head=config.model.train_vision_head,
        train_text_projection=config.model.train_text_projection,
        train_logit_scale=False,
        train_image_adapter=True,
        train_text_lora=bool(config.model.text_lora_rank),
    )
    groups, _, named = optimizer_parameter_groups(model, config.train)
    expected = {"image_adapter", "vision_head", "text_projection"}
    expected.add("text_backbone")
    assert {group["name"] for group in groups} == expected
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    before = {
        name: parameter.detach().clone() for group in expected for name, parameter in named[group]
    }
    optimizer = torch.optim.AdamW(groups)
    pixels = torch.randn(3, 8)
    tokens = torch.randint(0, 20, (3, 7))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        image, text, scale, _, _ = model(pixels, tokens)
        logits = scale * image @ text.T
        labels = torch.arange(3)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        loss.backward()
        for group in expected:
            assert (
                sum(
                    float(parameter.grad.abs().sum())
                    for _, parameter in named[group]
                    if parameter.grad is not None
                )
                > 0
            ), group
        optimizer.step()

    parameters = dict(model.named_parameters())
    assert all(torch.equal(parameters[name], value) for name, value in frozen.items())
    for group in expected:
        assert any(not torch.equal(parameters[name], before[name]) for name, _ in named[group]), (
            group
        )
    assert all(
        not parameter.requires_grad for parameter in model.visual_model.backbone.parameters()
    )
    assert (
        all(parameter.requires_grad for parameter in model.visual_model.head.parameters()) is head
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.text_model.backbone.token_embedding.parameters()
    )
    if fulltext:
        assert all(
            not parameter.requires_grad
            for block in model.text_model.backbone.blocks[:2]
            for parameter in block.parameters()
        )
    else:
        assert all(
            not parameter.requires_grad or name.endswith(("lora_A", "lora_B"))
            for name, parameter in model.named_parameters()
            if name.startswith("text_model.backbone.blocks.")
        )
