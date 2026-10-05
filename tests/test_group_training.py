import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from test_trainer_verification import TinyModel, TinyTokenizer

from dinotxt_rs.config import (
    Config,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    load_config,
    validate_config,
)
from dinotxt_rs.losses import symmetric_contrastive_loss, symmetric_group_contrastive_loss
from dinotxt_rs.training.trainer import train


class FrozenBackboneTiny(TinyModel):
    def __init__(self):
        super().__init__()
        self.visual_model.backbone = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.visual_model.backbone.requires_grad_(False)

    def forward(self, pixels, tokens):
        return super().forward(self.visual_model.backbone(pixels), tokens)


@pytest.mark.parametrize("objective", ["multi_positive", "mask_same_caption"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_singleton_group_loss_and_gradients_equal_original(objective, dtype):
    torch.manual_seed(5)
    image = torch.randn(4, 6, dtype=dtype, requires_grad=True)
    text = torch.randn(4, 6, dtype=dtype, requires_grad=True)
    scale = torch.tensor(2.0, requires_grad=True)
    original = symmetric_contrastive_loss(image, text, scale).loss
    original_grads = torch.autograd.grad(original, (image, text, scale), retain_graph=True)
    grouped = symmetric_group_contrastive_loss(
        image, text, scale, ["a", "b", "c", "d"], objective=objective
    ).loss
    grouped_grads = torch.autograd.grad(grouped, (image, text, scale))
    torch.testing.assert_close(original, grouped)
    for old, new in zip(original_grads, grouped_grads, strict=True):
        torch.testing.assert_close(old, new)


def test_multi_positive_loss_matches_uniform_target_and_attracts_all_positives():
    image = torch.eye(3)
    text = torch.zeros(3, 3, requires_grad=True)
    scale = torch.tensor(1.0)
    result = symmetric_group_contrastive_loss(image, text, scale, ["a", "a", "b"])
    target = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    logits = image @ text.T
    expected = (
        -(target * F.log_softmax(logits, 1)).sum(1).mean()
        -(target * F.log_softmax(logits.T, 1)).sum(1).mean()
    ) / 2
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    assert text.grad[1, 0] < 0  # Off-diagonal positive is attracted.
    assert text.grad[0, 1] < 0
    assert text.grad[2, 0] > 0  # Different group is repelled.


def test_mask_control_excludes_same_caption_from_negative_denominator():
    image = torch.eye(3)
    text = torch.zeros(3, 3, requires_grad=True)
    output = symmetric_group_contrastive_loss(
        image, text, torch.tensor(1.0), ["a", "a", "b"], objective="mask_same_caption"
    )
    assert float(output.loss.detach()) == pytest.approx((2 * torch.log(torch.tensor(2.0))
                                                      + torch.log(torch.tensor(3.0))) / 3)
    output.loss.backward()
    assert text.grad[0, 1] == 0
    assert text.grad[1, 0] == 0
    assert text.grad[2, 0] > 0


@pytest.mark.parametrize("objective", ["multi_positive", "mask_same_caption"])
def test_all_positive_batch_has_finite_loss_and_gradients(objective):
    image = torch.eye(3, requires_grad=True)
    text = torch.zeros(3, 3, requires_grad=True)
    result = symmetric_group_contrastive_loss(
        image, text, torch.tensor(1.0), ["a"] * 3, objective=objective
    )
    expected = torch.log(torch.tensor(3.0)) if objective == "multi_positive" else 0.0
    assert float(result.loss.detach()) == pytest.approx(expected)
    result.loss.backward()
    assert torch.isfinite(image.grad).all() and torch.isfinite(text.grad).all()


@pytest.mark.parametrize("groups", [[], ["a"], ["", "b"], [None, "b"]])
def test_invalid_group_targets_fail(groups):
    with pytest.raises(ValueError, match="one nonempty group"):
        symmetric_group_contrastive_loss(torch.eye(2), torch.eye(2), torch.tensor(1.0), groups)


def _config(tmp_path, objective, images_per_caption):
    records = []
    captions = ["road", "road", "river", "river", "farm", "farm", "lake", "lake"]
    for index, caption in enumerate(captions):
        image = tmp_path / f"train-{index}.png"
        Image.new("RGB", (8, 8), (20 + index * 20, 50, 80)).save(image)
        records.append({"id": f"train-{index}", "image": str(image), "caption": caption,
                        "split": "train", "source": "SkyScript", "group_id": caption})
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    validation_records = []
    for index, caption in enumerate(["airport", "factory"]):
        image = tmp_path / f"val-{index}.png"
        Image.new("RGB", (8, 8), (130, 20 + index * 50, 60)).save(image)
        validation_records.append({"id": f"val-{index}", "image": str(image),
                                   "caption": caption, "split": "val", "source": "SkyScript"})
    val_manifest = tmp_path / "val.jsonl"
    val_manifest.write_text("".join(json.dumps(row) + "\n" for row in validation_records))
    for name in ("backbone.pth", "dinotxt.pth", "vocab.gz"):
        (tmp_path / name).write_bytes(b"tiny")
    source = tmp_path / "config.toml"
    source.write_text("[experiment]\nname = 'group-test'\n")
    return Config(
        experiment=ExperimentConfig("group-test", 11, tmp_path / "full"),
        model=ModelConfig(tmp_path, "web", tmp_path / "backbone.pth",
                          tmp_path / "dinotxt.pth", tmp_path / "vocab.gz", image_size=16),
        data=DataConfig(train_manifest=manifest, val_manifest=val_manifest,
                        validation_batch_size=2, num_workers=0, train_augmentation=False,
                        caption_sampling="caption_group", images_per_caption=images_per_caption),
        train=TrainConfig(device="cpu", precision="fp32", batch_size=4,
                          gradient_accumulation=2, max_steps=3, warmup_steps=0,
                          contrastive_objective=objective, validation_every=2,
                          checkpoint_every=2, log_every=2),
        source=source,
    )


@pytest.mark.parametrize("objective,per_caption", [
    ("single_positive", 1), ("multi_positive", 2), ("mask_same_caption", 2)
])
def test_group_training_exact_resume_and_fixed_validation(tmp_path, objective, per_caption):
    config = _config(tmp_path, objective, per_caption)
    validate_config(config)
    torch.manual_seed(100)
    initial = FrozenBackboneTiny().state_dict()
    full_model = FrozenBackboneTiny()
    full_model.load_state_dict(initial)
    full_summary = json.loads(train(config, full_model, TinyTokenizer()).read_text())
    interrupted = replace(
        config, experiment=replace(config.experiment, output_dir=tmp_path / "part")
    )
    resumed_model = FrozenBackboneTiny()
    resumed_model.load_state_dict(initial)
    train(interrupted, resumed_model, TinyTokenizer(), stop_after_step=1)
    checkpoint = tmp_path / "part" / "latest.pt"
    resumed_model = FrozenBackboneTiny()
    resumed_model.load_state_dict(initial)
    resumed_summary = json.loads(train(interrupted, resumed_model, TinyTokenizer(),
                                       resume=checkpoint).read_text())
    full = torch.load(tmp_path / "full" / "latest.pt", weights_only=False)
    resumed = torch.load(checkpoint, weights_only=False)
    # Compare every saved trainable tensor; the two runs cross group epochs.
    assert full.keys() == resumed.keys()
    state_key = "trainable_model"
    assert state_key in full
    for name, value in full[state_key].items():
        torch.testing.assert_close(value, resumed[state_key][name], rtol=0, atol=0)
    for model in (full_model, resumed_model):
        torch.testing.assert_close(model.visual_model.backbone.weight,
                                   initial["visual_model.backbone.weight"], rtol=0, atol=0)
        assert model.visual_model.backbone.weight.grad is None
        assert not model.visual_model.backbone.training
    assert full["run_state"]["seen_training_indices"] == (
        resumed["run_state"]["seen_training_indices"]
    )
    assert full_summary["validation"]["final_loss"] == resumed_summary["validation"]["final_loss"]
    assert resumed_summary["caption_groups"]["sample_exposures"] == 24
    assert resumed_summary["caption_groups"]["unique_images_seen"] == 8
    assert resumed_summary["caption_groups"]["image_pool_coverage"] == 1.0
    assert full_summary["validation"]["positive_definition"] == "manifest_row_one_to_one"
    assert sorted(path.name for path in (tmp_path / "part").glob("*.pt")) == [
        "best.pt", "latest.pt", "step_0000000.pt"
    ]
    rows = [json.loads(line) for line in
            (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()]
    assert all("caption_group_diagnostics" in row for row in rows)
    expected = 1 if per_caption == 1 else 2
    assert all(row["caption_group_diagnostics"]["mean_positives_per_query"] == expected
               for row in rows)


def test_group_training_rejects_validation_caption_leakage(tmp_path):
    config = _config(tmp_path, "multi_positive", 2)
    rows = [json.loads(line) for line in config.data.val_manifest.read_text().splitlines()]
    rows[0]["caption"] = " ROAD "
    config.data.val_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="caption groups overlap"):
        train(config, TinyModel(), TinyTokenizer())
    assert not config.experiment.output_dir.exists()


def test_group_training_config_guards():
    config = load_config("configs/skyscript_sat_adapter_textlora_3epoch_seed11.toml")
    grouped = replace(config, data=replace(config.data, caption_sampling="caption_group",
                                          images_per_caption=2),
                      train=replace(config.train, contrastive_objective="multi_positive"))
    validate_config(grouped)
    with pytest.raises(ValueError, match="queue_size=0"):
        validate_config(replace(grouped, train=replace(grouped.train, queue_size=16)))
    with pytest.raises(ValueError, match="group-aware objective"):
        validate_config(replace(grouped, train=replace(grouped.train,
                                                      contrastive_objective="single_positive")))
    with pytest.raises(ValueError, match=">=2 images"):
        validate_config(replace(grouped, data=replace(grouped.data, images_per_caption=1)))
    # Old configs retain all defaults and the old sampler/loss protocol.
    assert config.data.caption_sampling == "rows"
    assert config.train.contrastive_objective == "single_positive"


@pytest.mark.parametrize("method,per_caption,objective", [
    ("rotate", 1, "single_positive"), ("multipos", 2, "multi_positive"),
    ("maskpos", 2, "mask_same_caption"),
])
def test_new_configs_match_original_model_budget_and_validation(method, per_caption, objective):
    base = load_config("configs/skyscript_sat_adapter_textlora_3epoch_seed11.toml")
    grouped = load_config(f"configs/skyscript_sat_adapter_textlora_{method}_1710step_seed11.toml")
    assert base.model == grouped.model
    assert base.experiment.seed == grouped.experiment.seed
    assert base.train == replace(grouped.train, contrastive_objective="single_positive")
    assert base.data == replace(
        grouped.data, train_manifest=base.data.train_manifest,
        caption_sampling="rows", images_per_caption=1,
    )
    assert grouped.data.images_per_caption == per_caption
    assert grouped.train.contrastive_objective == objective
