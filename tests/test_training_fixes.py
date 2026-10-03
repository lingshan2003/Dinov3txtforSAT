import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from test_trainer_verification import TinyModel, TinyTokenizer, _write_manifest

from dinotxt_rs.cli import train as train_cli
from dinotxt_rs.config import Config, DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from dinotxt_rs.training import trainer
from tools.verify_training_run import verify_training_run


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    images = []
    for index in range(4):
        image = tmp_path / f"image-{index}.png"
        Image.new("RGB", (8, 8), color=(20 + index * 30, 40, 60)).save(image)
        images.append(image)
    manifest = tmp_path / "pairs.jsonl"
    _write_manifest(manifest, images)
    source = tmp_path / "config.toml"
    source.write_text('[experiment]\nname="test"\n', encoding="utf-8")
    config = Config(
        ExperimentConfig("test", 11, tmp_path / "run"),
        ModelConfig(
            tmp_path,
            "sat",
            tmp_path / "backbone",
            tmp_path / "text",
            tmp_path / "bpe",
            image_size=16,
        ),
        DataConfig(
            manifest, manifest, validation_batch_size=2, num_workers=0, train_augmentation=False
        ),
        TrainConfig(
            device="cpu",
            precision="fp32",
            batch_size=2,
            max_steps=5,
            warmup_steps=0,
            validation_every=2,
            checkpoint_every=2,
            log_every=2,
        ),
        source,
    )
    monkeypatch.setattr(
        trainer,
        "build_provenance",
        lambda config: {
            "project_commit": "test",
            "dinov3_commit": "test",
            "files": {
                "train_manifest": {"sha256": "fixture"},
                "val_manifest": {"sha256": "fixture"},
            },
        },
    )
    monkeypatch.setattr(trainer, "sha256_file", lambda path: "fixture-checkpoint")
    return config


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("validation_every, expected_steps", [(2, [0, 2, 4, 5]), (5, [0, 5])])
def test_rolling_checkpoint_count_and_terminal_validation(
    experiment, monkeypatch, validation_every, expected_steps
):
    config = replace(experiment, train=replace(experiment.train, validation_every=validation_every))
    real_evaluate = trainer._evaluate_validation
    count = 0

    def improving_validation(**kwargs):
        nonlocal count
        result = real_evaluate(**kwargs)
        count += 1
        result["loss"] = 10.0 - count
        return result

    monkeypatch.setattr(trainer, "_evaluate_validation", improving_validation)
    summary = json.loads(trainer.train(config, TinyModel(), TinyTokenizer()).read_text())
    output = config.experiment.output_dir
    assert sorted(path.name for path in output.glob("*.pt")) == [
        "best.pt",
        "latest.pt",
        "step_0000000.pt",
    ]
    assert [record["step"] for record in records(output / "validation.jsonl")] == expected_steps
    assert summary["validation"]["last_step"] == 5
    assert summary["validation"]["final_loss"] == records(output / "validation.jsonl")[-1]["loss"]
    assert checkpoint(output / "latest.pt")["step"] == 5
    assert checkpoint(output / "best.pt")["step"] == 5
    assert checkpoint(output / "step_0000000.pt")["step"] == 0
    assert [r["step"] for r in records(output / "metrics.jsonl")] == [2, 4, 5]
    assert records(output / "metrics.jsonl")[-1]["averaged_optimizer_steps"] == 1
    assert not list(output.glob("*.part"))
    report = verify_training_run(
        output_dir=output,
        expected_steps=5,
        expected_train_manifest_sha256="fixture",
        require_completed=True,
        require_validation=True,
        validation_every=validation_every,
        expected_val_manifest_sha256="fixture",
        require_best_checkpoint=True,
    )
    assert report["final_checkpoint"] == str(output / "latest.pt")


def test_no_improvement_keeps_initial_best(experiment, monkeypatch):
    real_evaluate = trainer._evaluate_validation
    count = 0

    def worsening_validation(**kwargs):
        nonlocal count
        result = real_evaluate(**kwargs)
        count += 1
        result["loss"] = float(count)
        return result

    monkeypatch.setattr(trainer, "_evaluate_validation", worsening_validation)
    summary = json.loads(trainer.train(experiment, TinyModel(), TinyTokenizer()).read_text())
    output = experiment.experiment.output_dir
    assert summary["validation"]["best_step"] == 0
    assert checkpoint(output / "best.pt")["step"] == 0
    assert checkpoint(output / "latest.pt")["step"] == 5


def test_rolling_resume_matches_uninterrupted_training(experiment):
    trainer.seed_everything(123)
    uninterrupted = TinyModel()
    trainer.train(experiment, uninterrupted, TinyTokenizer())
    resumed_config = replace(
        experiment,
        experiment=replace(
            experiment.experiment, output_dir=experiment.experiment.output_dir.parent / "resumed"
        ),
    )
    trainer.seed_everything(123)
    trainer.train(resumed_config, TinyModel(), TinyTokenizer(), stop_after_step=3)
    output = resumed_config.experiment.output_dir
    assert [r["step"] for r in records(output / "validation.jsonl")] == [0, 2]
    resumed = TinyModel()
    trainer.train(resumed_config, resumed, TinyTokenizer(), resume=output / "latest.pt")
    for name, parameter in uninterrupted.named_parameters():
        assert torch.equal(parameter, dict(resumed.named_parameters())[name])
    assert [r["step"] for r in records(output / "metrics.jsonl")] == [2, 3, 4, 5]
    assert [r["step"] for r in records(output / "validation.jsonl")] == [0, 2, 4, 5]
    assert len(list(output.glob("*.pt"))) == 3


def test_resume_best_archives_future_logs(experiment, monkeypatch):
    monkeypatch.setattr(
        trainer,
        "_evaluate_validation",
        lambda **kwargs: {
            "loss": 1.0,
            "logit_scale": 1.0,
        },
    )
    trainer.train(experiment, TinyModel(), TinyTokenizer())
    output = experiment.experiment.output_dir
    assert checkpoint(output / "best.pt")["step"] == 0
    trainer.train(experiment, TinyModel(), TinyTokenizer(), resume=output / "best.pt")
    assert [r["step"] for r in records(output / "metrics.jsonl")] == [2, 4, 5]
    assert [r["step"] for r in records(output / "validation.jsonl")] == [0, 2, 4, 5]
    archive = records(output / "resume_discarded.jsonl")
    assert archive[0]["resume_step"] == 0
    assert sum(len(log["records"]) for log in archive[0]["logs"]) == 6


def test_frozen_logit_scale_is_not_modified(experiment):
    model = TinyModel()
    model.logit_scale.data.fill_(5.0)
    model.logit_scale.requires_grad_(False)
    trainer.train(experiment, model, TinyTokenizer())
    assert model.logit_scale.item() == 5.0


def test_validation_loader_does_not_consume_global_rng(experiment):
    _, loader = trainer._load_validation_loader(experiment, torch.device("cpu"))
    torch.manual_seed(17)
    before = torch.get_rng_state()
    next(iter(loader))
    assert torch.equal(before, torch.get_rng_state())


def test_cli_seeds_before_constructing_adapter(experiment, monkeypatch):
    config = replace(experiment, model=replace(experiment.model, image_adapter_bottleneck=256))
    monkeypatch.setattr(
        train_cli,
        "parse_args",
        lambda: SimpleNamespace(config=config.source, resume=None, stop_after_step=None),
    )
    monkeypatch.setattr(train_cli, "load_config", lambda path: config)
    monkeypatch.setattr(train_cli, "required_paths", lambda config: [])
    monkeypatch.setattr(
        train_cli, "load_official_dinotxt", lambda *args: (TinyModel(), TinyTokenizer())
    )
    monkeypatch.setattr(
        train_cli,
        "configure_trainable_parameters",
        lambda *args, **kwargs: {
            "total": 1,
            "trainable": 1,
        },
    )
    weights = []

    def capture_initialization(config, model, tokenizer, **kwargs):
        weights.append(model.image_adapter.down.weight.detach().clone())
        return config.experiment.output_dir / "training_summary.json"

    monkeypatch.setattr(train_cli, "train", capture_initialization)
    torch.manual_seed(1)
    train_cli.main()
    torch.manual_seed(999)
    train_cli.main()
    assert torch.equal(weights[0], weights[1])


def test_fp16_overflow_does_not_advance_optimizer_schedule(experiment, monkeypatch):
    class OverflowOnceScaler:
        def __init__(self, *args, **kwargs):
            self.attempts = 0
            self.scale_value = 65536.0
            self.updates = 0

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            self.attempts += 1
            if self.attempts == 1:
                optimizer.param_groups[0]["params"][0].grad.fill_(float("inf"))

        def step(self, optimizer):
            if self.attempts > 1:
                optimizer.step()
                self.updates += 1

        def update(self):
            if self.attempts == 1:
                self.scale_value /= 2

        def get_scale(self):
            return self.scale_value

        def state_dict(self):
            return {"attempts": self.attempts, "updates": self.updates}

    monkeypatch.setattr(torch.amp, "GradScaler", OverflowOnceScaler)
    config = replace(experiment, train=replace(experiment.train, max_steps=2))
    summary = json.loads(trainer.train(config, TinyModel(), TinyTokenizer()).read_text())
    payload = checkpoint(config.experiment.output_dir / "latest.pt")
    assert summary["steps"] == 2
    assert summary["micro_steps"] == 3
    assert summary["skipped_optimizer_steps"] == 1
    assert not summary["all_gradients_finite"]
    assert summary["all_applied_gradients_finite"]
    assert payload["scaler"] == {"attempts": 3, "updates": 2}
    assert payload["scheduler"]["last_epoch"] == 2
