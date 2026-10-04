import tomllib
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


def test_load_accelerated_validation_sat_pilot_config() -> None:
    config = load_config(Path("configs/pilot_sat_500step_formal_schedule_fast_validation.toml"))
    assert config.model.backbone_domain == "sat"
    assert config.data.num_workers == 0
    assert config.data.validation_batch_size == 16
    assert config.data.validation_forward_batch_size == 64
    assert config.data.validation_num_workers == 4
    assert config.data.validation_prefetch_factor == 4
    assert config.train.max_steps == 5000
    assert config.train.warmup_steps == 250


def test_load_m4_abc_configs() -> None:
    a = load_config(Path("configs/m4_web_a_fullscope_lr5e6_100step.toml"))
    b = load_config(Path("configs/m4_web_b_visionhead_lr5e5_100step.toml"))
    c = load_config(Path("configs/m4_web_c_visionhead_lr5e6_100step.toml"))

    assert a.model.text_last_k == 4
    assert a.model.train_text_projection and a.model.train_logit_scale
    assert a.train.learning_rate == 5e-6
    for config in (b, c):
        assert config.model.text_last_k == 0
        assert config.model.train_vision_head
        assert not config.model.train_text_projection
        assert not config.model.train_logit_scale
        assert config.data.validation_forward_batch_size == 64
        assert config.train.max_steps == 5000
    assert b.train.learning_rate == 5e-5
    assert c.train.learning_rate == 5e-6


def test_load_skyscript_adapter_configs() -> None:
    smoke = load_config(Path("configs/skyscript_web_adapter_10step.toml"))
    pilot = load_config(Path("configs/skyscript_web_adapter_100step.toml"))
    seed23 = load_config(Path("configs/skyscript_web_adapter_100step_seed23.toml"))
    seed47 = load_config(Path("configs/skyscript_web_adapter_100step_seed47.toml"))
    for config in (smoke, pilot, seed23, seed47):
        assert config.model.image_adapter_bottleneck == 256
        assert config.model.text_last_k == 0
        assert not config.model.train_vision_head
        assert not config.model.train_text_projection
        assert not config.model.train_logit_scale
        assert not config.data.train_augmentation
        assert config.train.queue_size == 0
    assert smoke.train.max_steps == 10
    assert not smoke.data.shuffle_train
    assert pilot.train.max_steps == 100
    assert pilot.train.validation_every == 25
    assert seed23.experiment.seed == 23
    assert seed47.experiment.seed == 47
    for replication in (seed23, seed47):
        assert replication.data == pilot.data
        assert replication.model == pilot.model
        assert replication.train == pilot.train


def test_load_skyscript_gate_s2_config() -> None:
    baseline = load_config(Path("configs/skyscript_web_adapter_100step.toml"))
    stability = load_config(Path("configs/skyscript_web_adapter_500step_seed11.toml"))

    assert stability.experiment.seed == 11
    assert stability.model == baseline.model
    assert stability.data.train_manifest == baseline.data.train_manifest
    assert stability.data.val_manifest == baseline.data.val_manifest
    assert stability.data.validation_batch_size == baseline.data.validation_batch_size
    assert (
        stability.data.validation_forward_batch_size == baseline.data.validation_forward_batch_size
    )
    assert stability.data.validation_num_workers == baseline.data.validation_num_workers
    assert stability.data.validation_prefetch_factor == baseline.data.validation_prefetch_factor
    assert stability.data.num_workers == baseline.data.num_workers
    assert stability.data.train_augmentation == baseline.data.train_augmentation
    assert stability.data.shuffle_train == baseline.data.shuffle_train
    assert stability.train.max_steps == 500
    assert stability.train.warmup_steps == 50
    assert stability.train.validation_every == 50
    assert stability.train.checkpoint_every == 50
    for field in (
        "device",
        "precision",
        "batch_size",
        "gradient_accumulation",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "queue_size",
        "validation_at_start",
        "log_every",
    ):
        assert getattr(stability.train, field) == getattr(baseline.train, field)


def test_load_new_three_epoch_skyscript_configs() -> None:
    adapter = load_config(Path("configs/skyscript_sat_adapter_3epoch_seed11.toml"))
    vision_head = load_config(Path("configs/skyscript_sat_visionhead_3epoch_seed11.toml"))

    for config in (adapter, vision_head):
        assert config.experiment.seed == 11
        assert (
            config.data.train_manifest.name
            == "skyscript_images23_train_raw_unique36495_seed11_global77.jsonl"
        )
        assert config.data.val_manifest is not None
        assert (
            config.data.val_manifest.name
            == "skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
        )
        assert config.data.validation_batch_size == 16
        assert config.data.validation_forward_batch_size == 64
        assert config.data.num_workers == 0
        assert not config.data.train_augmentation
        assert config.train.batch_size == 16
        assert config.train.gradient_accumulation == 4
        assert config.train.max_steps == 1710
        assert config.train.warmup_steps == 171
        assert config.train.validation_every == 200
        assert config.train.validation_at_start
        assert config.train.checkpoint_every == 200
        assert config.train.checkpoint_policy == "rolling"
        assert config.train.log_every == 10
        assert config.experiment.output_dir.name == config.experiment.name

    assert adapter.experiment.name == "skyscript_sat_adapter_3epoch_seed11"
    assert adapter.model.image_adapter_bottleneck == 256
    assert not adapter.model.train_vision_head
    assert not adapter.model.train_text_projection

    assert vision_head.experiment.name == "skyscript_sat_visionhead_3epoch_seed11"
    assert vision_head.model.image_adapter_bottleneck == 0
    assert vision_head.model.train_vision_head
    assert not vision_head.model.train_text_projection
    assert vision_head.model.vision_head_drop_path == 0.0


def test_historical_configs_keep_numbered_checkpoint_policy() -> None:
    rolling_configs = {
        "skyscript_sat_adapter_3epoch_seed11.toml",
        "skyscript_sat_adapter_3epoch_seed23.toml",
        "skyscript_sat_adapter_3epoch_seed47.toml",
        "skyscript_sat_visionhead_3epoch_seed11.toml",
        "skyscript_sat_adapter_textlora_3epoch_seed23.toml",
        "skyscript_sat_adapter_textlora_3epoch_seed47.toml",
        "skyscript_sat_visionhead_textlora_3epoch_seed23.toml",
        "skyscript_sat_visionhead_textlora_3epoch_seed47.toml",
        "skyscript_sat_textlast2_3epoch_seed11.toml",
        "skyscript_sat_textproj_3epoch_seed11.toml",
        "skyscript_sat_textlora_3epoch_seed11.toml",
        "skyscript_sat_adapter_textlora_3epoch_seed11.toml",
        "skyscript_sat_visionhead_textlora_3epoch_seed11.toml",
        "skyscript_sat_adapter_textproj_3epoch_seed11.toml",
        "skyscript_sat_visionhead_textproj_3epoch_seed11.toml",
    }
    configs = sorted(Path("configs").glob("*.toml"))
    assert {
        path.name for path in configs if load_config(path).train.checkpoint_policy == "rolling"
    } == rolling_configs
    assert all(
        load_config(path).train.checkpoint_policy == "numbered"
        for path in configs
        if path.name not in rolling_configs
    )


def test_three_epoch_sat_seed_replicates_only_change_experiment_identity() -> None:
    configs_dir = Path("configs")
    trials = ("adapter_textlora", "adapter", "visionhead_textlora")
    for trial in trials:
        baseline_path = configs_dir / f"skyscript_sat_{trial}_3epoch_seed11.toml"
        baseline = tomllib.loads(baseline_path.read_text())
        baseline_experiment = baseline.pop("experiment")
        assert baseline_experiment["seed"] == 11
        assert "resume" not in baseline_experiment

        for seed in (23, 47):
            name = f"skyscript_sat_{trial}_3epoch_seed{seed}"
            replication_path = configs_dir / f"{name}.toml"
            replication = tomllib.loads(replication_path.read_text())
            experiment = replication.pop("experiment")

            assert replication == baseline
            assert experiment == {
                "name": name,
                "seed": seed,
                "output_dir": f"outputs/{name}",
            }
            assert "seed11" not in experiment["output_dir"]
            assert "resume" not in experiment

            loaded = load_config(replication_path)
            assert loaded.experiment.seed == seed
            assert loaded.experiment.output_dir.name == name


def test_legacy_phase_runner_configs_are_numbered() -> None:
    # These runners use fixed numbered resume boundaries. Keep their historical
    # configs on numbered retention so a rolling run cannot silently replace them.
    legacy_configs = (
        "skyscript_sat_adapter_500step_seed11.toml",
        "skyscript_sat_adapter_500step_seed23.toml",
        "skyscript_sat_adapter_500step_seed47.toml",
        "skyscript_sat_f1_visionhead_lr1e5_500step_seed11.toml",
        "skyscript_sat_f1_visionhead_lr5e6_500step_seed11.toml",
        "skyscript_sat_f2_adapter256_visionhead_lr1e5_500step_seed11.toml",
        "skyscript_sat_f2_adapter256_visionhead_lr5e6_500step_seed11.toml",
        "skyscript_sat_f3_adapter256_textproj_lr1e5_500step_seed11.toml",
        "skyscript_sat_f3_adapter256_textproj_lr5e6_500step_seed11.toml",
    )
    for name in legacy_configs:
        assert load_config(Path("configs") / name).train.checkpoint_policy == "numbered"
