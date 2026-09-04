from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    dinov3_repo: Path
    backbone_domain: str
    backbone_weights: Path
    dinotxt_weights: Path
    bpe_vocab: Path
    image_size: int = 224
    text_last_k: int = 4
    train_vision_head: bool = True
    train_text_projection: bool = True
    train_logit_scale: bool = True


@dataclass(frozen=True)
class DataConfig:
    train_manifest: Path
    val_manifest: Path | None = None
    # The number of examples in each independent InfoNCE loss.  This is part
    # of the validation metric definition and must stay fixed to compare runs.
    validation_batch_size: int | None = None
    # Evaluation may forward several loss batches together, then split their
    # embeddings back into validation_batch_size groups before computing loss.
    validation_forward_batch_size: int | None = None
    # Validation is deterministic (no shuffle or augmentation), so it can use
    # separate I/O workers without changing train-resume RNG semantics.
    validation_num_workers: int | None = None
    validation_prefetch_factor: int | None = None
    num_workers: int = 8
    train_augmentation: bool = True
    shuffle_train: bool = True
    fixed_monitor_manifest: Path | None = None
    fixed_monitor_batch_size: int | None = None


@dataclass(frozen=True)
class TrainConfig:
    device: str = "cuda"
    precision: str = "bf16"
    batch_size: int = 16
    gradient_accumulation: int = 1
    max_steps: int = 1000
    warmup_steps: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    queue_size: int = 0
    fixed_monitor_every: int = 0
    validation_every: int = 0
    validation_at_start: bool = True
    log_every: int = 10
    checkpoint_every: int = 500


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    source: Path


def _expanded(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: _expanded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expanded(item) for item in value]
    return value


def _path(value: str | None) -> Path | None:
    return None if value is None else Path(value)


def load_config(path: str | Path) -> Config:
    source = Path(path).resolve()
    with source.open("rb") as handle:
        raw = _expanded(tomllib.load(handle))

    exp = raw["experiment"]
    model = raw["model"]
    data = raw["data"]
    train = raw["train"]
    config = Config(
        experiment=ExperimentConfig(
            name=str(exp["name"]), seed=int(exp["seed"]), output_dir=Path(exp["output_dir"])
        ),
        model=ModelConfig(
            dinov3_repo=Path(model["dinov3_repo"]),
            backbone_domain=str(model["backbone_domain"]),
            backbone_weights=Path(model["backbone_weights"]),
            dinotxt_weights=Path(model["dinotxt_weights"]),
            bpe_vocab=Path(model["bpe_vocab"]),
            image_size=int(model.get("image_size", 224)),
            text_last_k=int(model.get("text_last_k", 4)),
            train_vision_head=bool(model.get("train_vision_head", True)),
            train_text_projection=bool(model.get("train_text_projection", True)),
            train_logit_scale=bool(model.get("train_logit_scale", True)),
        ),
        data=DataConfig(
            train_manifest=Path(data["train_manifest"]),
            val_manifest=_path(data.get("val_manifest")),
            validation_batch_size=(
                None
                if data.get("validation_batch_size") is None
                else int(data["validation_batch_size"])
            ),
            validation_forward_batch_size=(
                None
                if data.get("validation_forward_batch_size") is None
                else int(data["validation_forward_batch_size"])
            ),
            validation_num_workers=(
                None
                if data.get("validation_num_workers") is None
                else int(data["validation_num_workers"])
            ),
            validation_prefetch_factor=(
                None
                if data.get("validation_prefetch_factor") is None
                else int(data["validation_prefetch_factor"])
            ),
            num_workers=int(data.get("num_workers", 8)),
            train_augmentation=bool(data.get("train_augmentation", True)),
            shuffle_train=bool(data.get("shuffle_train", True)),
            fixed_monitor_manifest=_path(data.get("fixed_monitor_manifest")),
            fixed_monitor_batch_size=(
                None
                if data.get("fixed_monitor_batch_size") is None
                else int(data["fixed_monitor_batch_size"])
            ),
        ),
        train=TrainConfig(
            device=str(train.get("device", "cuda")),
            precision=str(train.get("precision", "bf16")),
            batch_size=int(train.get("batch_size", 16)),
            gradient_accumulation=int(train.get("gradient_accumulation", 1)),
            max_steps=int(train.get("max_steps", 1000)),
            warmup_steps=int(train.get("warmup_steps", 100)),
            learning_rate=float(train.get("learning_rate", 5e-5)),
            weight_decay=float(train.get("weight_decay", 0.01)),
            max_grad_norm=float(train.get("max_grad_norm", 1.0)),
            queue_size=int(train.get("queue_size", 0)),
            fixed_monitor_every=int(train.get("fixed_monitor_every", 0)),
            validation_every=int(train.get("validation_every", 0)),
            validation_at_start=bool(train.get("validation_at_start", True)),
            log_every=int(train.get("log_every", 10)),
            checkpoint_every=int(train.get("checkpoint_every", 500)),
        ),
        source=source,
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    if config.model.backbone_domain not in {"web", "sat"}:
        raise ValueError("model.backbone_domain must be 'web' or 'sat'")
    if config.model.image_size <= 0 or config.model.image_size % 16:
        raise ValueError("model.image_size must be a positive multiple of the ViT patch size (16)")
    if not 0 <= config.model.text_last_k <= 24:
        raise ValueError("model.text_last_k must be in [0, 24]")
    if config.train.precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("train.precision must be bf16, fp16, or fp32")
    positive = {
        "batch_size": config.train.batch_size,
        "gradient_accumulation": config.train.gradient_accumulation,
        "max_steps": config.train.max_steps,
        "log_every": config.train.log_every,
        "checkpoint_every": config.train.checkpoint_every,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"train.{name} must be positive")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers must be nonnegative")
    has_fixed_monitor = config.data.fixed_monitor_manifest is not None
    if has_fixed_monitor != (config.data.fixed_monitor_batch_size is not None):
        raise ValueError(
            "data.fixed_monitor_manifest and data.fixed_monitor_batch_size must be set together"
        )
    if has_fixed_monitor:
        if config.data.fixed_monitor_batch_size <= 0:
            raise ValueError("data.fixed_monitor_batch_size must be positive")
        if config.train.fixed_monitor_every <= 0:
            raise ValueError(
                "train.fixed_monitor_every must be positive when a fixed monitor is set"
            )
    elif config.train.fixed_monitor_every:
        raise ValueError("train.fixed_monitor_every requires data.fixed_monitor_manifest")
    has_validation = config.data.val_manifest is not None
    if config.train.validation_every:
        if config.train.validation_every <= 0:
            raise ValueError("train.validation_every must be positive")
        if not has_validation:
            raise ValueError("train.validation_every requires data.val_manifest")
        if config.data.validation_batch_size is None:
            raise ValueError("train.validation_every requires data.validation_batch_size")
        if config.data.validation_batch_size <= 0:
            raise ValueError("data.validation_batch_size must be positive")
        forward_batch_size = (
            config.data.validation_batch_size
            if config.data.validation_forward_batch_size is None
            else config.data.validation_forward_batch_size
        )
        if forward_batch_size <= 0:
            raise ValueError("data.validation_forward_batch_size must be positive")
        if forward_batch_size < config.data.validation_batch_size:
            raise ValueError(
                "data.validation_forward_batch_size must be at least validation_batch_size"
            )
        if forward_batch_size % config.data.validation_batch_size:
            raise ValueError(
                "data.validation_forward_batch_size must be a multiple of validation_batch_size"
            )
        validation_num_workers = config.data.validation_num_workers
        if validation_num_workers is not None and validation_num_workers < 0:
            raise ValueError("data.validation_num_workers must be nonnegative")
        prefetch_factor = config.data.validation_prefetch_factor
        if prefetch_factor is not None and prefetch_factor <= 0:
            raise ValueError("data.validation_prefetch_factor must be positive")
        if prefetch_factor is not None and (validation_num_workers or 0) == 0:
            raise ValueError(
                "data.validation_prefetch_factor requires positive validation_num_workers"
            )
    elif config.data.validation_batch_size is not None:
        raise ValueError("data.validation_batch_size requires train.validation_every")
    elif any(
        value is not None
        for value in (
            config.data.validation_forward_batch_size,
            config.data.validation_num_workers,
            config.data.validation_prefetch_factor,
        )
    ):
        raise ValueError("data.validation acceleration settings require train.validation_every")
    if config.train.warmup_steps >= config.train.max_steps:
        raise ValueError("train.warmup_steps must be smaller than train.max_steps")


def required_paths(config: Config) -> list[Path]:
    paths = [
        config.model.dinov3_repo,
        config.model.backbone_weights,
        config.model.dinotxt_weights,
        config.model.bpe_vocab,
        config.data.train_manifest,
    ]
    if config.data.fixed_monitor_manifest is not None:
        paths.append(config.data.fixed_monitor_manifest)
    if config.train.validation_every:
        if config.data.val_manifest is None:  # Defensive guard for callers bypassing validation.
            raise ValueError("train.validation_every requires data.val_manifest")
        paths.append(config.data.val_manifest)
    return paths
