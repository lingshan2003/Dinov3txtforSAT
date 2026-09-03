from pathlib import Path

from dinotxt_rs.config import load_config


def test_load_mvp_config() -> None:
    config = load_config(Path("configs/train_mvp_web.toml"))
    assert config.model.backbone_domain == "web"
    assert config.model.image_size == 224
    assert config.model.text_last_k == 4
    assert (
        config.data.train_manifest.name
        == "chatearthnet_35_train_10k_seed11_no_nodata_global77.jsonl"
    )
    assert config.data.val_manifest is not None
    assert config.data.val_manifest.name == "chatearthnet_35_val_no_nodata_global77.jsonl"
    assert config.train.queue_size == 4096


def test_load_bounded_web_verification_config() -> None:
    config = load_config(Path("configs/verify_web_10step.toml"))
    assert config.experiment.name == "m3_web_global77_fixed16_10step_seed11"
    assert config.data.train_manifest.name.endswith("fixed16.jsonl")
    assert not config.data.train_augmentation
    assert not config.data.shuffle_train
    assert config.data.num_workers == 0
    assert config.train.max_steps == 10
    assert config.train.gradient_accumulation == 1
    assert config.train.queue_size == 0
    assert config.train.log_every == 1
    assert config.train.checkpoint_every == 10


def test_load_web_loss_trend_config() -> None:
    config = load_config(Path("configs/verify_web_100step.toml"))
    assert config.experiment.name == "m3_web_global77_100step_seed11"
    assert config.data.train_augmentation
    assert config.data.shuffle_train
    assert config.data.num_workers == 8
    assert config.train.max_steps == 100
    assert config.train.warmup_steps == 5
    assert config.train.gradient_accumulation == 4
    assert config.train.queue_size == 4096
    assert config.train.checkpoint_every == 50
