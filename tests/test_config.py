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


def test_load_web_fixed_monitor_config() -> None:
    config = load_config(Path("configs/verify_web_100step_fixed_monitor.toml"))
    assert config.experiment.name == "m3_web_global77_100step_fixedmonitor_seed11"
    assert config.data.fixed_monitor_manifest is not None
    assert config.data.fixed_monitor_manifest.name.endswith("fixed16.jsonl")
    assert config.data.fixed_monitor_batch_size == 16
    assert config.train.fixed_monitor_every == 10


def test_load_validation_resume_configs() -> None:
    web = load_config(Path("configs/verify_web_100step_validation_resume.toml"))
    sat = load_config(Path("configs/verify_sat_100step_validation_resume.toml"))
    assert web.model.backbone_domain == "web"
    assert sat.model.backbone_domain == "sat"
    assert web.data.validation_batch_size == sat.data.validation_batch_size == 16
    assert web.data.num_workers == sat.data.num_workers == 0
    assert web.train.validation_every == sat.train.validation_every == 50
    assert web.train.validation_at_start and sat.train.validation_at_start


def test_load_formal_schedule_500_step_pilot_config() -> None:
    config = load_config(Path("configs/pilot_web_500step_formal_schedule.toml"))
    assert config.experiment.name == "m3_web_global77_formalschedule_500step_pilot_seed11"
    assert config.data.num_workers == 0
    assert config.train.max_steps == 5000
    assert config.train.warmup_steps == 250
    assert config.train.validation_every == 50
    assert config.train.checkpoint_every == 250
