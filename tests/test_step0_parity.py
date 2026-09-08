from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from PIL import Image

from dinotxt_rs.config import Config, DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from dinotxt_rs.evaluation.common import EvaluationModel
from dinotxt_rs.evaluation.parity import (
    compare_tensors,
    default_tolerances,
    parameter_state_fingerprint,
    run_step0_parity,
    tensor_sha256,
    trainable_parameter_fingerprint,
)


class _TinyTokenizer:
    def tokenize(self, captions: list[str]) -> torch.Tensor:
        return torch.tensor([[len(caption)] for caption in captions], dtype=torch.float32)


class _TinyParityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_projection = torch.nn.Linear(3, 4, bias=False)
        self.text_projection = torch.nn.Linear(1, 4, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, pixels: torch.Tensor, tokens: torch.Tensor):
        image_features = F.normalize(self.image_projection(pixels.mean(dim=(-1, -2))), dim=-1)
        text_features = F.normalize(self.text_projection(tokens), dim=-1)
        patch_tokens = image_features[:, None, :]
        return (
            image_features,
            text_features,
            self.logit_scale.exp(),
            patch_tokens,
            patch_tokens.clone(),
        )


def test_compare_tensors_reports_exact_and_out_of_tolerance_values() -> None:
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    exact = compare_tensors(reference, reference.clone(), atol=1e-6, rtol=1e-5)
    changed = compare_tensors(reference, reference + 0.1, atol=1e-6, rtol=1e-5)

    assert exact["passed"]
    assert exact["max_absolute_error"] == 0.0
    assert not changed["passed"]
    assert changed["max_absolute_error"] > 0.09


def test_parameter_fingerprint_detects_weight_change() -> None:
    first = torch.nn.Linear(3, 2)
    second = torch.nn.Linear(3, 2)
    second.load_state_dict(first.state_dict())

    first_fingerprint = trainable_parameter_fingerprint(first)
    second_fingerprint = trainable_parameter_fingerprint(second)
    assert first_fingerprint == second_fingerprint

    with torch.no_grad():
        second.weight[0, 0] += 1
    assert trainable_parameter_fingerprint(second) != first_fingerprint


def test_parameter_state_fingerprint_is_order_independent() -> None:
    weight = torch.tensor([[1.0, 2.0]])
    bias = torch.tensor([3.0])

    assert parameter_state_fingerprint({"weight": weight, "bias": bias}) == (
        parameter_state_fingerprint({"bias": bias, "weight": weight})
    )


def test_tensor_sha256_includes_dtype_and_shape() -> None:
    values = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    assert tensor_sha256(values) == tensor_sha256(values.clone())
    assert tensor_sha256(values) != tensor_sha256(values.reshape(2, 2))
    assert tensor_sha256(values) != tensor_sha256(values.to(torch.float32))


def test_default_tolerances_are_precision_aware() -> None:
    fp32 = default_tolerances("fp32")
    bf16 = default_tolerances("bf16")
    assert fp32[0] < bf16[0]
    assert fp32[1] < bf16[1]


def test_run_step0_parity_passes_for_identical_models(tmp_path, monkeypatch) -> None:
    image_paths: list[str] = []
    records = []
    for index, color in enumerate(((20, 40, 60), (80, 100, 120))):
        image_path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (8, 8), color=color).save(image_path)
        image_paths.append(str(image_path))
        records.append(
            {
                "id": f"sample-{index}",
                "image": str(image_path),
                "caption": f"caption {index}",
                "split": "train",
                "source": "test",
            }
        )
    manifest = tmp_path / "fixed.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    source = tmp_path / "config.toml"
    source.write_text("[experiment]\nname='parity-test'\n", encoding="utf-8")
    config = Config(
        experiment=ExperimentConfig(name="parity-test", seed=11, output_dir=tmp_path),
        model=ModelConfig(
            dinov3_repo=tmp_path,
            backbone_domain="web",
            backbone_weights=tmp_path / "backbone.pth",
            dinotxt_weights=tmp_path / "dinotxt.pth",
            bpe_vocab=tmp_path / "vocab.gz",
            image_size=16,
        ),
        data=DataConfig(train_manifest=manifest),
        train=TrainConfig(device="cpu", precision="fp32"),
        source=source,
    )
    official_model = _TinyParityModel()
    restored_model = _TinyParityModel()
    restored_model.load_state_dict(official_model.state_dict())
    for parameter in restored_model.parameters():
        parameter.requires_grad_(False)
    restored_model.register_parameter(
        "adapter_down",
        torch.nn.Parameter(torch.randn(4, 2)),
    )
    official = EvaluationModel(
        official_model,
        _TinyTokenizer(),
        config,
        torch.device("cpu"),
        {"model_variant": "official_without_image_adapter", "checkpoint": None},
    )
    restored = EvaluationModel(
        restored_model,
        _TinyTokenizer(),
        config,
        torch.device("cpu"),
        {"checkpoint": {"step": 0, "sha256": "checkpoint"}},
    )
    checkpoint = tmp_path / "step_0000000.pt"
    torch.save(
        {"trainable_model": {"adapter_down": restored_model.adapter_down.detach().clone()}},
        checkpoint,
    )
    monkeypatch.setattr(
        "dinotxt_rs.evaluation.parity.load_official_reference_model", lambda *args: official
    )
    monkeypatch.setattr(
        "dinotxt_rs.evaluation.parity.load_evaluation_model", lambda *args, **kwargs: restored
    )

    report = run_step0_parity(
        config,
        checkpoint=checkpoint,
        training_output=tmp_path,
        input_manifest=manifest,
        batch_size=2,
    )

    assert report["status"] == "pass"
    assert report["checkpoint_step"] == 0
    assert report["trainable_parameters"]["passed"]
    assert report["trainable_parameters"]["comparison"] == "checkpoint_state_vs_restored_model"
    assert report["official_model"]["model_variant"] == "official_without_image_adapter"
    assert all(value["passed"] for value in report["comparisons"].values())
