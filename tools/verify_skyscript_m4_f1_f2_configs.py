#!/usr/bin/env python3
"""Verify the frozen SAT-only F1/F2 mechanism-screen configuration matrix."""

from __future__ import annotations

from pathlib import Path

from dinotxt_rs.config import Config, load_config

CONFIGS = {
    "f1_visionhead_lr1e5": Path(
        "configs/skyscript_sat_f1_visionhead_lr1e5_500step_seed11.toml"
    ),
    "f1_visionhead_lr5e6": Path(
        "configs/skyscript_sat_f1_visionhead_lr5e6_500step_seed11.toml"
    ),
    "f2_adapter256_visionhead_lr1e5": Path(
        "configs/skyscript_sat_f2_adapter256_visionhead_lr1e5_500step_seed11.toml"
    ),
    "f2_adapter256_visionhead_lr5e6": Path(
        "configs/skyscript_sat_f2_adapter256_visionhead_lr5e6_500step_seed11.toml"
    ),
}
EXPECTED_HEAD_LR = {
    "f1_visionhead_lr1e5": 1e-5,
    "f1_visionhead_lr5e6": 5e-6,
    "f2_adapter256_visionhead_lr1e5": 1e-5,
    "f2_adapter256_visionhead_lr5e6": 5e-6,
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
    assert config.model.text_last_k == 0
    assert not config.model.train_text_projection
    assert not config.model.train_logit_scale


def verify() -> dict[str, Config]:
    baseline = load_config("configs/skyscript_sat_adapter_500step_seed11.toml")
    configs = {name: load_config(path) for name, path in CONFIGS.items()}
    outputs = set()
    for name, config in configs.items():
        _assert_shared_protocol(config, baseline)
        assert config.model.train_vision_head
        assert config.train.vision_head_learning_rate == EXPECTED_HEAD_LR[name]
        assert config.train.vision_head_weight_decay == 0.01
        assert config.experiment.output_dir not in outputs
        outputs.add(config.experiment.output_dir)
        if name.startswith("f1_"):
            assert config.model.image_adapter_bottleneck == 0
            assert config.train.image_adapter_learning_rate is None
        else:
            assert config.model.image_adapter_bottleneck == 256
            assert config.train.image_adapter_learning_rate == 1e-4
            assert config.train.image_adapter_weight_decay == 0.01
    return configs


def main() -> None:
    configs = verify()
    print("m4_f1_f2_sat_config_protocol=verified")
    print("f0=reused_not_retrained")
    print("candidates=" + ",".join(configs))
    print("visual_backbone=permanently_frozen")


if __name__ == "__main__":
    main()
