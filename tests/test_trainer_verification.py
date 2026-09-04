import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from dinotxt_rs.config import Config, DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from dinotxt_rs.training.trainer import _evaluate_validation, train


class TinyTokenizer:
    def tokenize(self, captions: list[str]) -> torch.Tensor:
        return torch.tensor([[len(caption)] for caption in captions], dtype=torch.float32)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual_model = torch.nn.Module()
        self.visual_model.backbone = torch.nn.Identity()
        self.image_projection = torch.nn.Linear(3, 4, bias=False)
        self.text_projection = torch.nn.Linear(1, 4, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, pixels: torch.Tensor, tokens: torch.Tensor):
        image_features = F.normalize(self.image_projection(pixels.mean(dim=(-1, -2))), dim=-1)
        text_features = F.normalize(self.text_projection(tokens), dim=-1)
        return image_features, text_features, self.logit_scale.exp(), None, None


def _write_manifest(path: Path, image_paths: list[Path]) -> None:
    records = [
        {
            "id": f"chatearthnet:sample-{index}",
            "image": str(image_path),
            "caption": f"caption {index}",
            "split": "train",
            "source": "ChatEarthNet",
        }
        for index, image_path in enumerate(image_paths)
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_bounded_training_writes_finite_step_metrics_and_summary(tmp_path) -> None:
    image_paths = []
    for index, color in enumerate(((20, 40, 60), (80, 100, 120))):
        image_path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (8, 8), color=color).save(image_path)
        image_paths.append(image_path)
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, image_paths)
    for name in ("backbone.pth", "dinotxt.pth", "vocab.gz"):
        (tmp_path / name).write_bytes(b"test")
    source = tmp_path / "config.toml"
    source.write_text("[experiment]\nname = 'test'\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    config = Config(
        experiment=ExperimentConfig(name="test", seed=11, output_dir=output_dir),
        model=ModelConfig(
            dinov3_repo=tmp_path,
            backbone_domain="web",
            backbone_weights=tmp_path / "backbone.pth",
            dinotxt_weights=tmp_path / "dinotxt.pth",
            bpe_vocab=tmp_path / "vocab.gz",
            image_size=16,
        ),
        data=DataConfig(
            train_manifest=manifest,
            num_workers=0,
            train_augmentation=False,
            shuffle_train=False,
            fixed_monitor_manifest=manifest,
            fixed_monitor_batch_size=2,
        ),
        train=TrainConfig(
            device="cpu",
            precision="fp32",
            batch_size=2,
            gradient_accumulation=1,
            max_steps=2,
            warmup_steps=0,
            queue_size=2,
            fixed_monitor_every=1,
            log_every=1,
            checkpoint_every=2,
        ),
        source=source,
    )

    model = TinyModel()
    summary_path = train(config, model, TinyTokenizer())

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = [
        json.loads(line) for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    fixed_monitor = [
        json.loads(line) for line in (output_dir / "fixed_monitor.jsonl").read_text().splitlines()
    ]
    assert summary["steps"] == 2
    assert summary["all_losses_finite"]
    assert summary["all_gradients_finite"]
    assert summary["peak_cuda_allocated_bytes"] is None
    assert isinstance(summary["initial_loss"], float)
    assert isinstance(summary["final_loss"], float)
    assert isinstance(summary["initial_in_batch_loss"], float)
    assert isinstance(summary["final_in_batch_loss"], float)
    assert Path(summary["final_checkpoint"]).is_file()
    assert summary["fixed_monitor"]["samples"] == 2
    assert summary["fixed_monitor"]["every"] == 1
    assert not list(output_dir.glob("*.part"))
    assert [record["step"] for record in metrics] == [1, 2]
    assert all(record["in_batch_loss"] >= 0 for record in metrics)
    assert all(record["gradient_norm"] >= 0 for record in metrics)
    assert [record["queue_size"] for record in metrics] == [2, 2]
    assert [record["step"] for record in fixed_monitor] == [0, 1, 2]
    assert all(record["loss"] >= 0 for record in fixed_monitor)
    assert model.training
    assert not model.visual_model.backbone.training


def test_validation_best_checkpoint_and_strict_resume(tmp_path) -> None:
    image_paths = []
    for index, color in enumerate(((20, 40, 60), (80, 100, 120), (130, 140, 150), (30, 80, 90))):
        image_path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (8, 8), color=color).save(image_path)
        image_paths.append(image_path)
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, image_paths)
    for name in ("backbone.pth", "dinotxt.pth", "vocab.gz"):
        (tmp_path / name).write_bytes(b"test")
    source = tmp_path / "config.toml"
    source_text = "[experiment]\nname = 'resume-test'\n"
    source.write_text(source_text, encoding="utf-8")
    output_dir = tmp_path / "outputs"
    config = Config(
        experiment=ExperimentConfig(name="resume-test", seed=11, output_dir=output_dir),
        model=ModelConfig(
            dinov3_repo=tmp_path,
            backbone_domain="web",
            backbone_weights=tmp_path / "backbone.pth",
            dinotxt_weights=tmp_path / "dinotxt.pth",
            bpe_vocab=tmp_path / "vocab.gz",
            image_size=16,
        ),
        data=DataConfig(
            train_manifest=manifest,
            val_manifest=manifest,
            validation_batch_size=2,
            num_workers=0,
            train_augmentation=False,
            shuffle_train=True,
        ),
        train=TrainConfig(
            device="cpu",
            precision="fp32",
            batch_size=2,
            gradient_accumulation=1,
            max_steps=2,
            warmup_steps=0,
            queue_size=2,
            validation_every=1,
            log_every=1,
            checkpoint_every=1,
        ),
        source=source,
    )

    train(config, TinyModel(), TinyTokenizer(), stop_after_step=1)
    checkpoint = output_dir / "step_0000001.pt"
    assert checkpoint.is_file()
    summary_path = train(config, TinyModel(), TinyTokenizer(), resume=checkpoint)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = [
        json.loads(line) for line in (output_dir / "validation.jsonl").read_text().splitlines()
    ]
    resume_history = [
        json.loads(line) for line in (output_dir / "resume_history.jsonl").read_text().splitlines()
    ]
    assert summary["completed"]
    assert summary["resumed_from"] == str(checkpoint.resolve())
    assert summary["validation"]["evaluations"] == 3
    assert summary["validation"]["best_checkpoint"] == str(output_dir / "best.pt")
    assert summary["validation"]["selection_includes_step_zero"]
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "step_0000000.pt").is_file()
    assert [record["step"] for record in validation] == [0, 1, 2]
    assert summary["validation"]["best_loss"] == min(record["loss"] for record in validation)
    assert [record["step"] for record in _read_jsonl(output_dir / "metrics.jsonl")] == [1, 2]
    assert resume_history[0]["checkpoint_step"] == 1
    assert isinstance(resume_history[0]["checkpoint_sha256"], str)

    source.write_text("[experiment]\nname = 'changed'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config.toml does not exactly match"):
        train(config, TinyModel(), TinyTokenizer(), resume=checkpoint)
    source.write_text(source_text, encoding="utf-8")
    (tmp_path / "backbone.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint identity differs: files"):
        train(config, TinyModel(), TinyTokenizer(), resume=checkpoint)


def test_larger_validation_forward_batch_preserves_loss_groups() -> None:
    records = [
        {
            "ids": f"sample-{index}",
            "pixels": torch.full((3, 8, 8), float(index + 1)),
            "captions": f"caption {index}",
        }
        for index in range(5)
    ]
    model = TinyModel().eval()
    tokenizer = TinyTokenizer()
    metric_loader = DataLoader(records, batch_size=2, shuffle=False)
    accelerated_loader = DataLoader(records, batch_size=4, shuffle=False)

    metric = _evaluate_validation(
        model=model,
        tokenizer=tokenizer,
        loader=metric_loader,
        device=torch.device("cpu"),
        precision="fp32",
        loss_batch_size=2,
    )
    accelerated = _evaluate_validation(
        model=model,
        tokenizer=tokenizer,
        loader=accelerated_loader,
        device=torch.device("cpu"),
        precision="fp32",
        loss_batch_size=2,
    )

    assert metric["loss"] == pytest.approx(accelerated["loss"], abs=1e-7)
    assert metric["batches"] == accelerated["batches"] == 3
    assert metric["forward_batches"] == 3
    assert accelerated["forward_batches"] == 2
    assert metric["elapsed_seconds"] < 60
    assert accelerated["elapsed_seconds"] < 60


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
