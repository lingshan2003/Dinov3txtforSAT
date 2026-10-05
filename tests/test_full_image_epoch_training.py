from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from PIL import Image
from test_group_training import FrozenBackboneTiny, _config
from test_trainer_verification import TinyTokenizer

from dinotxt_rs.config import validate_config
from dinotxt_rs.losses.contrastive import LossOutput
from dinotxt_rs.training import trainer


def image_epoch_config(tmp_path, output_name: str = "full"):
    config = _config(tmp_path, "multi_positive", 2)
    rows = [json.loads(line) for line in config.data.train_manifest.read_text().splitlines()]
    image_path = tmp_path / "train-8.png"
    Image.new("RGB", (8, 8), (180, 50, 70)).save(image_path)
    rows.append({
        "id": "train-8", "image": str(image_path), "caption": "road",
        "split": "train", "source": "SkyScript", "group_id": "road",
    })
    config.data.train_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return replace(
        config,
        experiment=replace(config.experiment, output_dir=tmp_path / output_name),
        data=replace(config.data, caption_sampling="image_epoch", images_per_caption=2),
        train=replace(config.train, image_epochs=3, max_steps=5, batch_size=4,
                      gradient_accumulation=2, validation_every=2,
                      checkpoint_every=2, log_every=2),
    )


def _initial_model_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(100)
    return {name: value.clone() for name, value in FrozenBackboneTiny().state_dict().items()}


def _load_model(state: dict[str, torch.Tensor]) -> FrozenBackboneTiny:
    model = FrozenBackboneTiny()
    model.load_state_dict(state)
    return model


def test_image_epoch_training_exact_resume_coverage_and_tail_update(tmp_path) -> None:
    config = image_epoch_config(tmp_path)
    validate_config(config)
    initial = _initial_model_state()

    full_model = _load_model(initial)
    full_summary_path = trainer.train(config, full_model, TinyTokenizer())
    full_summary = json.loads(full_summary_path.read_text())

    interrupted = replace(
        config, experiment=replace(config.experiment, output_dir=tmp_path / "part")
    )
    trainer.train(interrupted, _load_model(initial), TinyTokenizer(), stop_after_step=2)
    checkpoint = interrupted.experiment.output_dir / "latest.pt"
    resumed_model = _load_model(initial)
    resumed_summary = json.loads(
        trainer.train(interrupted, resumed_model, TinyTokenizer(), resume=checkpoint).read_text()
    )

    full = torch.load(config.experiment.output_dir / "latest.pt", weights_only=False)
    resumed = torch.load(checkpoint, weights_only=False)
    for name, value in full["trainable_model"].items():
        torch.testing.assert_close(value, resumed["trainable_model"][name], rtol=0, atol=0)
    assert full["run_state"]["sample_exposures"] == 27
    assert resumed["run_state"]["sample_exposures"] == 27
    for model in (full_model, resumed_model):
        torch.testing.assert_close(
            model.visual_model.backbone.weight,
            initial["visual_model.backbone.weight"], rtol=0, atol=0,
        )
        assert model.visual_model.backbone.weight.grad is None
        assert not model.visual_model.backbone.training

    assert full_summary["steps"] == resumed_summary["steps"] == 5
    assert full_summary["completed"] is resumed_summary["completed"] is True
    image_epochs = resumed_summary["image_epochs"]
    assert image_epochs == {
        "target_epochs": 3,
        "completed_epochs": 3,
        "current_epoch_batch_offset": 0,
        "epoch_definition": "one_pass_over_every_training_image",
        "sample_exposures": 27,
        "unique_images_seen": 9,
        "image_pool_coverage": 1.0,
        "micro_batches_per_epoch": 3,
        "target_micro_batches": 9,
    }
    assert resumed_summary["caption_groups"]["unique_images_seen"] == 9
    assert resumed_summary["caption_groups"]["image_pool_coverage"] == 1.0
    assert "completed_group_epochs" not in resumed_summary["caption_groups"]
    # Four full accumulation windows plus the final one-microbatch window.
    metrics = [
        json.loads(line) for line in
        (config.experiment.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in metrics][-1] == 5
    validation = [
        json.loads(line) for line in
        (config.experiment.output_dir / "validation.jsonl").read_text().splitlines()
    ]
    assert validation[0]["step"] == 0
    assert validation[-1]["step"] == 5
    assert sorted(path.name for path in (tmp_path / "part").glob("*.pt")) == [
        "best.pt", "latest.pt", "step_0000000.pt"
    ]


@pytest.mark.parametrize(
    "epochs,accumulation,steps,expected_grad", [(3, 2, 5, 1.0), (1, 4, 1, 3.0)],
)
def test_incomplete_accumulation_window_uses_actual_microbatch_count(
    tmp_path, monkeypatch, epochs, accumulation, steps, expected_grad,
) -> None:
    config = image_epoch_config(tmp_path)
    config = replace(
        config, experiment=replace(config.experiment, output_dir=tmp_path / "scale"),
        train=replace(config.train, image_epochs=epochs, gradient_accumulation=accumulation,
                      max_steps=steps),
    )
    captured_logit_scale_grads: list[float] = []
    real_check = trainer._checked_grad_norm

    def record_grad(*args, **kwargs):
        named_parameters = args[0]
        captured_logit_scale_grads.append(
            float(next(parameter for name, parameter in named_parameters
                       if name == "logit_scale").grad)
        )
        return real_check(*args, **kwargs)

    def parameter_probe(image_features, text_features, logit_scale, group_ids, **kwargs):
        # Keep every trainable path in the graph, with a known logit-scale
        # derivative equal to the number of real examples in this microbatch.
        loss = image_features.square().sum() + text_features.square().sum()
        loss = loss + logit_scale.sum() * image_features.shape[0]
        return LossOutput(loss, loss.detach(), loss.detach())

    monkeypatch.setattr(trainer, "_checked_grad_norm", record_grad)
    monkeypatch.setattr(trainer, "symmetric_group_contrastive_loss", parameter_probe)
    trainer.train(config, _load_model(_initial_model_state()), TinyTokenizer())

    # At 3 epochs / accumulation2, the last window has only the one-row tail:
    # its mean derivative is 1, not .5. At 1 epoch / accumulation4, the actual
    # window has 4+4+1 rows across three batches: derivative 9/3=3, not 9/4.
    assert captured_logit_scale_grads[-1] == pytest.approx(expected_grad)
    assert len(captured_logit_scale_grads) == steps


def test_image_epoch_budget_mismatch_is_rejected_before_output_creation(tmp_path) -> None:
    config = image_epoch_config(tmp_path)
    invalid = replace(config, train=replace(config.train, max_steps=4))
    with pytest.raises(ValueError, match="max_steps"):
        trainer.train(invalid, _load_model(_initial_model_state()), TinyTokenizer())
    assert not invalid.experiment.output_dir.exists()
