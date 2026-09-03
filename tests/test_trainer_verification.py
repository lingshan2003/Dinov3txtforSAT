import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from dinotxt_rs.config import Config, DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from dinotxt_rs.training.trainer import train


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
        ),
        train=TrainConfig(
            device="cpu",
            precision="fp32",
            batch_size=2,
            gradient_accumulation=1,
            max_steps=2,
            warmup_steps=0,
            queue_size=0,
            log_every=1,
            checkpoint_every=2,
        ),
        source=source,
    )

    summary_path = train(config, TinyModel(), TinyTokenizer())

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = [
        json.loads(line) for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert summary["steps"] == 2
    assert summary["all_losses_finite"]
    assert summary["all_gradients_finite"]
    assert summary["peak_cuda_allocated_bytes"] is None
    assert isinstance(summary["initial_loss"], float)
    assert isinstance(summary["final_loss"], float)
    assert Path(summary["final_checkpoint"]).is_file()
    assert not list(output_dir.glob("*.part"))
    assert [record["step"] for record in metrics] == [1, 2]
    assert all(record["gradient_norm"] >= 0 for record in metrics)
