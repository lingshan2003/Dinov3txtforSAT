#!/usr/bin/env python3
"""Verify the frozen SAT-only F3 text-projection mechanism-screen configs."""

from __future__ import annotations

from pathlib import Path

from dinotxt_rs.config import Config, load_config

CONFIGS = {
    "f3_adapter256_textproj_lr1e5": Path(
        "configs/skyscript_sat_f3_adapter256_textproj_lr1e5_500step_seed11.toml"
    ),
    "f3_adapter256_textproj_lr5e6": Path(
        "configs/skyscript_sat_f3_adapter256_textproj_lr5e6_500step_seed11.toml"
    ),
}
EXPECTED_PROJECTION_LR = {
    "f3_adapter256_textproj_lr1e5": 1e-5,
    "f3_adapter256_textproj_lr5e6": 5e-6,
}


def _assert_shared_protocol(config: Config, baseline: Config) -> None:
    assert config.experiment.seed == baseline.experiment.seed == 11
    assert config.model.backbone_domain == baseline.model.backbone_domain == "sat"
    for field in (
        "dinov3_repo",
        "backbone_weights",
        "dinotxt_weights",
        "bpe_vocab",
        "image_size",
        "image_adapter_bottleneck",
    ):
        assert getattr(config.model, field) == getattr(baseline.model, field)
    assert config.data == baseline.data
    for field in (
        "device",
        "precision",
        "batch_size",
        "gradient_accumulation",
        "max_steps",
        "warmup_steps",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "queue_size",
        "fixed_monitor_every",
        "validation_every",
        "validation_at_start",
        "log_every",
        "checkpoint_every",
    ):
        assert getattr(config.train, field) == getattr(baseline.train, field)
    assert config.train.max_steps == 500 and config.train.warmup_steps == 50
    assert config.model.image_adapter_bottleneck == 256
    assert config.model.text_last_k == 0
    assert not config.model.train_vision_head
    assert config.model.train_text_projection
    assert not config.model.train_logit_scale
    assert config.train.image_adapter_learning_rate == 1e-4
    assert config.train.image_adapter_weight_decay == 0.01
    assert config.train.vision_head_learning_rate is None
    assert config.train.text_backbone_learning_rate is None
    assert config.train.logit_scale_learning_rate is None


def verify() -> dict[str, Config]:
    baseline = load_config("configs/skyscript_sat_adapter_500step_seed11.toml")
    configs = {name: load_config(path) for name, path in CONFIGS.items()}
    outputs = set()
    for name, config in configs.items():
        _assert_shared_protocol(config, baseline)
        assert config.train.text_projection_learning_rate == EXPECTED_PROJECTION_LR[name]
        assert config.train.text_projection_weight_decay == 0.01
        assert config.experiment.output_dir not in outputs
        outputs.add(config.experiment.output_dir)
    return configs


def main() -> None:
    configs = verify()
    print("m4_f3_sat_config_protocol=verified")
    print("f0=reused_not_retrained")
    print("candidates=" + ",".join(configs))
    print("visual_backbone=permanently_frozen")
    print("vision_head=frozen")


if __name__ == "__main__":
    main()
