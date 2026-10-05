from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dinotxt_rs.config import Config
from dinotxt_rs.data import ImageTextDataset, collate_image_text, make_transform
from dinotxt_rs.data.image_text import validate_caption_groups
from dinotxt_rs.losses import (
    EmbeddingQueue,
    symmetric_contrastive_loss,
    symmetric_group_contrastive_loss,
)
from dinotxt_rs.models import assert_visual_backbone_frozen, optimizer_parameter_groups
from dinotxt_rs.training.checkpoint import (
    load_checkpoint,
    prepare_resume_artifacts,
    save_best_checkpoint,
    save_checkpoint,
)
from dinotxt_rs.training.provenance import (
    build_provenance,
    run_identity,
    sha256_file,
    write_provenance,
)
from dinotxt_rs.training.sampler import (
    ResumableBatchSampler,
    ResumableCaptionGroupBatchSampler,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _scheduler(optimizer: Any, warmup: int, total: int):
    def factor(step: int) -> float:
        if step < warmup:
            return max(step, 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _assert_finite_loss(loss: torch.Tensor, sample_ids: list[str]) -> None:
    if not torch.isfinite(loss).all():
        preview = ", ".join(sample_ids[:5])
        raise FloatingPointError(f"Non-finite loss for samples: {preview}")


def _checked_grad_norm(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    named_parameter_groups: dict[str, list[tuple[str, torch.nn.Parameter]]],
    max_grad_norm: float,
) -> tuple[float, dict[str, float]]:
    missing_gradients: list[str] = []
    for name, parameter in named_parameters:
        if parameter.grad is None:
            missing_gradients.append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite gradient for trainable parameter: {name}")
    if missing_gradients:
        preview = ", ".join(missing_gradients[:5])
        raise RuntimeError(f"No gradient was produced for trainable parameter(s): {preview}")

    group_norms: dict[str, float] = {}
    for group_name, group_parameters in named_parameter_groups.items():
        if not group_parameters:
            continue
        squared_norm = sum(
            float(parameter.grad.detach().float().norm(2)) ** 2 for _, parameter in group_parameters
        )
        group_norm = math.sqrt(squared_norm)
        if not math.isfinite(group_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm for optimizer group {group_name}: {group_norm}"
            )
        group_norms[group_name] = group_norm

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in named_parameters],
        max_grad_norm,
        error_if_nonfinite=True,
    )
    value = float(gradient_norm)
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite gradient norm: {value}")
    return value, group_norms


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _peak_cuda_allocated_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _set_training_mode(model: Any) -> None:
    model.train()
    # The frozen backbone must remain deterministic while the alignment modules train.
    model.visual_model.backbone.eval()
    # A frozen tower must also stay in eval mode. ``requires_grad=False`` alone does not
    # disable dropout or other training-time behavior, so head-only experiments need an
    # explicit mode policy in addition to the parameter freeze policy.
    vision_head = getattr(model.visual_model, "head", None)
    if vision_head is not None:
        vision_head.train(any(parameter.requires_grad for parameter in vision_head.parameters()))
    text_model = getattr(model, "text_model", None)
    if text_model is None:
        return
    text_model.train(any(parameter.requires_grad for parameter in text_model.parameters()))
    text_backbone = text_model.backbone
    text_backbone.train(any(parameter.requires_grad for parameter in text_backbone.parameters()))
    text_model.head.train(
        any(parameter.requires_grad for parameter in text_model.head.parameters())
    )


def _load_fixed_monitor_batch(config: Config) -> dict[str, Any] | None:
    manifest = config.data.fixed_monitor_manifest
    batch_size = config.data.fixed_monitor_batch_size
    if manifest is None or batch_size is None:
        return None
    dataset = ImageTextDataset(
        manifest,
        make_transform(config.model.image_size, config.model.backbone_domain, train=False),
    )
    if len(dataset) != batch_size:
        raise ValueError(
            "Fixed monitor manifest must contain exactly fixed_monitor_batch_size samples, got "
            f"{len(dataset)} and {batch_size}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.experiment.seed + 2),
        num_workers=0,
        drop_last=False,
        collate_fn=collate_image_text,
    )
    batch = next(iter(loader))
    if len(batch["ids"]) != batch_size:
        raise RuntimeError("Fixed monitor DataLoader did not return the configured full batch")
    return batch


def _evaluate_fixed_monitor(
    *,
    model: Any,
    tokenizer: Any,
    batch: dict[str, Any],
    device: torch.device,
    precision: str,
) -> dict[str, float | int | None]:
    was_training = model.training
    model.eval()
    try:
        pixels = batch["pixels"].to(device, non_blocking=True)
        tokens = tokenizer.tokenize(batch["captions"]).to(device, non_blocking=True)
        if tokens.shape[0] != pixels.shape[0]:
            raise RuntimeError(
                "Fixed monitor tokenizer batch size does not match image batch size for samples: "
                + ", ".join(batch["ids"][:5])
            )
        with torch.no_grad(), _autocast(device, precision):
            image_features, text_features, logit_scale, _, _ = model(pixels, tokens)
            monitor_loss = symmetric_contrastive_loss(
                image_features, text_features, logit_scale, queue=None
            ).loss
        _assert_finite_loss(monitor_loss, batch["ids"])
        return {
            "loss": float(monitor_loss),
            "logit_scale": float(logit_scale.detach()),
            "peak_cuda_allocated_bytes": _peak_cuda_allocated_bytes(device),
        }
    finally:
        if was_training:
            _set_training_mode(model)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _update_group_diagnostics(state: dict[str, Any], groups: list[str]) -> None:
    counts = Counter(groups)
    positive_counts = [counts[group] for group in groups]
    state["micro_batches"] = state.get("micro_batches", 0) + 1
    state["unique_captions"] = state.get("unique_captions", 0) + len(counts)
    state["queries"] = state.get("queries", 0) + len(groups)
    state["positive_pairs"] = state.get("positive_pairs", 0) + sum(positive_counts)
    histogram = state.setdefault("positive_count_histogram", {})
    for count in positive_counts:
        key = str(count)
        histogram[key] = histogram.get(key, 0) + 1


def _group_diagnostics_report(state: dict[str, Any]) -> dict[str, Any]:
    batches, queries = state["micro_batches"], state["queries"]
    pairs = state["positive_pairs"]
    return {
        "micro_batches": batches,
        "mean_unique_captions": state["unique_captions"] / batches,
        "mean_positives_per_query": pairs / queries,
        "mean_off_diagonal_positives": (pairs - queries) / batches,
        "singleton_query_fraction": state["positive_count_histogram"].get("1", 0) / queries,
        "positive_count_histogram": dict(state["positive_count_histogram"]),
    }


def _validation_loader_settings(config: Config) -> tuple[int, int, int | None]:
    """Resolve the independent forward/I/O settings for deterministic validation."""
    loss_batch_size = config.data.validation_batch_size
    if loss_batch_size is None:  # Defensive guard for callers bypassing config validation.
        raise ValueError("Validation was enabled without a validation_batch_size")
    forward_batch_size = config.data.validation_forward_batch_size or loss_batch_size
    num_workers = (
        config.data.num_workers
        if config.data.validation_num_workers is None
        else config.data.validation_num_workers
    )
    return forward_batch_size, num_workers, config.data.validation_prefetch_factor


def _load_validation_loader(
    config: Config, device: torch.device
) -> tuple[ImageTextDataset, DataLoader] | None:
    if not config.train.validation_every:
        return None
    if config.data.val_manifest is None or config.data.validation_batch_size is None:
        raise ValueError("Validation was enabled without a manifest and batch size")
    dataset = ImageTextDataset(
        config.data.val_manifest,
        make_transform(config.model.image_size, config.model.backbone_domain, train=False),
    )
    if not len(dataset):
        raise ValueError("Validation manifest is empty")
    forward_batch_size, num_workers, prefetch_factor = _validation_loader_settings(config)
    loader_kwargs: dict[str, Any] = {
        "batch_size": forward_batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "drop_last": False,
        "collate_fn": collate_image_text,
        "generator": torch.Generator().manual_seed(config.experiment.seed + 1),
    }
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )
    return dataset, loader


def _evaluate_validation(
    *,
    model: Any,
    tokenizer: Any,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    loss_batch_size: int,
) -> dict[str, float | int | None]:
    """Evaluate deterministic groups without queue, batching only the model forward.

    ``loss_batch_size`` defines the published validation metric.  A DataLoader
    batch may contain a multiple of that size solely to run fewer GPU forward
    passes; embeddings are split back into their original groups before the
    InfoNCE loss is calculated.
    """
    if loss_batch_size <= 0:
        raise ValueError("loss_batch_size must be positive")
    forward_batch_size = loader.batch_size
    if not isinstance(forward_batch_size, int) or forward_batch_size % loss_batch_size:
        raise ValueError("Validation forward batch size must be a multiple of loss_batch_size")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0
    loss_batches = 0
    forward_batches = 0
    logit_scale_value: float | None = None
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for batch in loader:
                if not batch["ids"]:
                    raise RuntimeError("Validation DataLoader produced an empty batch")
                pixels = batch["pixels"].to(device, non_blocking=True)
                tokens = tokenizer.tokenize(batch["captions"]).to(device, non_blocking=True)
                if tokens.shape[0] != pixels.shape[0]:
                    raise RuntimeError(
                        "Validation tokenizer batch size does not match image batch size "
                        "for samples: " + ", ".join(batch["ids"][:5])
                    )
                with _autocast(device, precision):
                    image_features, text_features, logit_scale, _, _ = model(pixels, tokens)
                forward_batches += 1
                for group_start in range(0, len(batch["ids"]), loss_batch_size):
                    group_end = group_start + loss_batch_size
                    group_ids = batch["ids"][group_start:group_end]
                    loss = symmetric_contrastive_loss(
                        image_features[group_start:group_end],
                        text_features[group_start:group_end],
                        logit_scale,
                        queue=None,
                    ).loss
                    _assert_finite_loss(loss, group_ids)
                    group_size = len(group_ids)
                    total_loss += float(loss) * group_size
                    total_samples += group_size
                    loss_batches += 1
                logit_scale_value = float(logit_scale.detach())
    finally:
        if was_training:
            _set_training_mode(model)
    if not total_samples or logit_scale_value is None:
        raise RuntimeError("Validation completed without any samples")
    return {
        "loss": total_loss / total_samples,
        "samples": total_samples,
        "batches": loss_batches,
        "forward_batches": forward_batches,
        "loss_batch_size": loss_batch_size,
        "forward_batch_size": forward_batch_size,
        "logit_scale": logit_scale_value,
        "elapsed_seconds": time.monotonic() - start,
        "peak_cuda_allocated_bytes": _peak_cuda_allocated_bytes(device),
    }


def train(
    config: Config,
    model: Any,
    tokenizer: Any,
    *,
    resume: Path | None = None,
    stop_after_step: int | None = None,
) -> Path:
    """Train, optionally from a strictly authenticated checkpoint.

    ``stop_after_step`` is intentionally a CLI execution bound rather than a
    configuration field.  It lets the verification script create a real,
    resumable interruption while preserving one immutable experiment TOML.
    """
    if stop_after_step is not None and not 0 < stop_after_step <= config.train.max_steps:
        raise ValueError("stop_after_step must be in [1, train.max_steps]")
    target_steps = stop_after_step or config.train.max_steps
    seed_everything(config.experiment.seed)
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    if config.model.vision_head_drop_path is not None:
        head = getattr(model.visual_model, "head", None)
        for block in getattr(head, "blocks", []):
            if hasattr(block, "sample_drop_ratio"):
                block.sample_drop_ratio = config.model.vision_head_drop_path
    _set_training_mode(model)
    assert_visual_backbone_frozen(model, require_eval=True)

    transform = make_transform(
        config.model.image_size,
        config.model.backbone_domain,
        train=config.data.train_augmentation,
    )
    dataset = ImageTextDataset(config.data.train_manifest, transform)
    if len(dataset) < config.train.batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, fewer than batch_size={config.train.batch_size}"
        )
    fixed_monitor_batch = _load_fixed_monitor_batch(config)
    validation = _load_validation_loader(config, device)
    group_ids: list[str] | None = None
    if config.data.caption_sampling == "caption_group":
        group_ids = validate_caption_groups(dataset.records, split="train")
        if validation is not None:
            validation_records = validation[0].records
            val_groups = {" ".join(row["caption"].split()).casefold()
                          for row in validation_records}
            if len(val_groups) != len(validation_records):
                raise ValueError("Group training retains the unique-caption validation protocol")
            if set(group_ids) & val_groups:
                raise ValueError("Training and validation caption groups overlap")
            train_images = {Path(row["image"]).resolve() for row in dataset.records}
            val_images = {Path(row["image"]).resolve() for row in validation_records}
            if train_images & val_images or (
                {row["id"] for row in dataset.records}
                & {row["id"] for row in validation_records}
            ):
                raise ValueError("Training and validation images or sample ids overlap")
    rolling_checkpoints = config.train.checkpoint_policy == "rolling"
    if rolling_checkpoints and (validation is None or not config.train.validation_at_start):
        raise ValueError("Rolling checkpoints require validation including step zero")
    if group_ids is None:
        sampler = ResumableBatchSampler(
            dataset_size=len(dataset), batch_size=config.train.batch_size,
            shuffle=config.data.shuffle_train, seed=config.experiment.seed,
        )
    else:
        sampler = ResumableCaptionGroupBatchSampler(
            group_ids=group_ids, batch_size=config.train.batch_size,
            images_per_group=config.data.images_per_caption,
            shuffle=config.data.shuffle_train, seed=config.experiment.seed,
        )
    if not sampler.batches_per_epoch:
        raise ValueError("Sampler cannot produce a full physical batch")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.experiment.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
        collate_fn=collate_image_text,
        generator=loader_generator,
    )
    optimizer_groups, optimizer_group_metadata, named_parameter_groups = optimizer_parameter_groups(
        model, config.train
    )
    named_parameters = [
        named_parameter for group in named_parameter_groups.values() for named_parameter in group
    ]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.99),
    )
    scheduler = _scheduler(optimizer, config.train.warmup_steps, config.train.max_steps)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config.train.precision == "fp16"
    )
    queue = EmbeddingQueue(config.train.queue_size)
    output_dir = config.experiment.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_text = config.source.read_text(encoding="utf-8")
    provenance = build_provenance(config)
    provenance["optimizer_parameter_groups"] = optimizer_group_metadata
    text_lora_metadata = getattr(model, "text_lora_metadata", None)
    if text_lora_metadata is not None:
        provenance["text_lora"] = text_lora_metadata
        print("text_lora=" + json.dumps(text_lora_metadata, ensure_ascii=False), flush=True)
    identity = run_identity(config_text, provenance)
    config_snapshot_path = output_dir / "config.toml"
    metrics_path = output_dir / "metrics.jsonl"
    fixed_monitor_path = output_dir / "fixed_monitor.jsonl"
    validation_path = output_dir / "validation.jsonl"
    resume_history_path = output_dir / "resume_history.jsonl"
    optimizer_groups_path = output_dir / "optimizer_groups.json"
    print(
        "optimizer_parameter_groups=" + json.dumps(optimizer_group_metadata, ensure_ascii=False),
        flush=True,
    )

    if resume is None:
        existing = [
            path.name
            for path in (
                config_snapshot_path,
                metrics_path,
                output_dir / "provenance.json",
                optimizer_groups_path,
            )
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite an existing training run without --resume: "
                + ", ".join(existing)
            )
        config_snapshot_path.write_text(config_text, encoding="utf-8")
        metrics_path.touch()
        provenance_path = write_provenance(config, provenance)
        _write_json_atomic(optimizer_groups_path, optimizer_group_metadata)
        print(f"provenance={provenance_path}", flush=True)
    else:
        if (
            not config_snapshot_path.is_file()
            or config_snapshot_path.read_text(encoding="utf-8") != config_text
        ):
            raise ValueError("Refusing to resume because output config.toml does not exactly match")
        if (
            not metrics_path.is_file()
            or not (output_dir / "provenance.json").is_file()
            or not optimizer_groups_path.is_file()
        ):
            raise ValueError(
                "Refusing to resume without existing metrics, provenance, and optimizer groups"
            )
        stored_optimizer_groups = json.loads(optimizer_groups_path.read_text(encoding="utf-8"))
        if stored_optimizer_groups != optimizer_group_metadata:
            raise ValueError("Refusing to resume because optimizer parameter groups differ")

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_in_batch_loss = 0.0
    running_log_steps = 0
    group_diagnostics: dict[str, Any] = {}
    seen_training_indices: set[int] = set()
    optimizer_step_loss = 0.0
    optimizer_step_in_batch_loss = 0.0
    initial_loss: float | None = None
    final_loss: float | None = None
    initial_in_batch_loss: float | None = None
    final_in_batch_loss: float | None = None
    initial_fixed_monitor_loss: float | None = None
    final_fixed_monitor_loss: float | None = None
    initial_validation_loss: float | None = None
    final_validation_loss: float | None = None
    validation_evaluations = 0
    last_validation_step: int | None = None
    skipped_optimizer_steps = 0
    consecutive_overflows = 0
    best_validation_loss: float | None = None
    best_validation_step: int | None = None
    best_checkpoint: Path | None = None
    last_gradient_norm: float | None = None
    last_group_gradient_norms: dict[str, float] | None = None
    last_checkpoint_step = 0
    resumed_from: Path | None = None

    if resume is not None:
        resumed_from = resume.resolve()
        global_step, run_state, identity_check = load_checkpoint(
            resumed_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            queue=queue,
            sampler=sampler,
            loader_generator=loader_generator,
            device=device,
            expected_identity=identity,
            expected_optimizer_parameter_groups=optimizer_group_metadata,
        )
        if global_step >= target_steps:
            raise ValueError(
                "Resume checkpoint is already at step "
                f"{global_step}, not below target {target_steps}"
            )
        micro_step = int(
            run_state.get("micro_step", global_step * config.train.gradient_accumulation)
        )
        if micro_step % config.train.gradient_accumulation:
            raise ValueError("Resume checkpoint was taken during gradient accumulation")
        running_loss = float(run_state.get("running_loss", 0.0))
        running_in_batch_loss = float(run_state.get("running_in_batch_loss", 0.0))
        running_log_steps = int(
            run_state.get("running_log_steps", global_step % config.train.log_every)
        )
        group_diagnostics = run_state.get("group_diagnostics", {})
        if not isinstance(group_diagnostics, dict):
            raise ValueError("Checkpoint group diagnostics are invalid")
        if group_ids is not None:
            seen_indices = run_state.get("seen_training_indices")
            if not isinstance(seen_indices, list) or any(
                type(index) is not int or not 0 <= index < len(dataset)
                for index in seen_indices
            ):
                raise ValueError("Checkpoint seen training indices are invalid")
            seen_training_indices = set(seen_indices)
        initial_loss = run_state.get("initial_loss")
        final_loss = run_state.get("final_loss")
        initial_in_batch_loss = run_state.get("initial_in_batch_loss")
        final_in_batch_loss = run_state.get("final_in_batch_loss")
        initial_fixed_monitor_loss = run_state.get("initial_fixed_monitor_loss")
        final_fixed_monitor_loss = run_state.get("final_fixed_monitor_loss")
        initial_validation_loss = run_state.get("initial_validation_loss")
        final_validation_loss = run_state.get("final_validation_loss")
        validation_evaluations = int(run_state.get("validation_evaluations", 0))
        last_validation_step = run_state.get("last_validation_step")
        skipped_optimizer_steps = int(run_state.get("skipped_optimizer_steps", 0))
        if validation is not None and run_state.get("validation_generator_state") is not None:
            validation[1].generator.set_state(run_state["validation_generator_state"])
        best_validation_loss = run_state.get("best_validation_loss")
        best_validation_step = run_state.get("best_validation_step")
        if best_validation_loss is not None:
            best_validation_loss = float(best_validation_loss)
        if best_validation_step is not None:
            best_validation_step = int(best_validation_step)
            best_checkpoint = output_dir / "best.pt"
        last_gradient_norm = run_state.get("last_gradient_norm")
        if last_gradient_norm is not None:
            last_gradient_norm = float(last_gradient_norm)
        restored_group_norms = run_state.get("last_group_gradient_norms")
        if restored_group_norms is not None:
            if not isinstance(restored_group_norms, dict):
                raise ValueError("Checkpoint group gradient norms are invalid")
            last_group_gradient_norms = {
                str(name): float(value) for name, value in restored_group_norms.items()
            }
        last_checkpoint_step = global_step
        artifact_recovery = prepare_resume_artifacts(
            output_dir, global_step, run_state, resumed_from, identity
        )
        _append_jsonl(
            resume_history_path,
            {
                "format_version": 1,
                "checkpoint": str(resumed_from),
                "checkpoint_sha256": sha256_file(resumed_from),
                "checkpoint_step": global_step,
                "target_steps": config.train.max_steps,
                "run_identity": identity,
                "identity_check": identity_check,
                "artifact_recovery": artifact_recovery,
            },
        )
        print(f"resumed_from={resumed_from} step={global_step}", flush=True)
    else:
        if fixed_monitor_batch is not None:
            fixed_monitor_record = _evaluate_fixed_monitor(
                model=model,
                tokenizer=tokenizer,
                batch=fixed_monitor_batch,
                device=device,
                precision=config.train.precision,
            )
            fixed_monitor_record["step"] = 0
            _append_jsonl(fixed_monitor_path, fixed_monitor_record)
            initial_fixed_monitor_loss = float(fixed_monitor_record["loss"])
            final_fixed_monitor_loss = initial_fixed_monitor_loss
        if validation is not None and config.train.validation_at_start:
            _, validation_loader = validation
            validation_record = _evaluate_validation(
                model=model,
                tokenizer=tokenizer,
                loader=validation_loader,
                device=device,
                precision=config.train.precision,
                loss_batch_size=config.data.validation_batch_size,
            )
            validation_record["step"] = 0
            _append_jsonl(validation_path, validation_record)
            initial_validation_loss = float(validation_record["loss"])
            final_validation_loss = initial_validation_loss
            validation_evaluations = 1
            last_validation_step = 0

    def checkpoint_run_state() -> dict[str, Any]:
        return {
            "micro_step": micro_step,
            "running_loss": running_loss,
            "running_in_batch_loss": running_in_batch_loss,
            "running_log_steps": running_log_steps,
            "group_diagnostics": group_diagnostics,
            "seen_training_indices": sorted(seen_training_indices),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "initial_in_batch_loss": initial_in_batch_loss,
            "final_in_batch_loss": final_in_batch_loss,
            "initial_fixed_monitor_loss": initial_fixed_monitor_loss,
            "final_fixed_monitor_loss": final_fixed_monitor_loss,
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "validation_evaluations": validation_evaluations,
            "last_validation_step": last_validation_step,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "validation_generator_state": (
                None if validation is None else validation[1].generator.get_state()
            ),
            "best_validation_loss": best_validation_loss,
            "best_validation_step": best_validation_step,
            "last_gradient_norm": last_gradient_norm,
            "last_group_gradient_norms": last_group_gradient_norms,
        }

    def save_current_checkpoint(name: str | None = None) -> Path:
        return save_checkpoint(
            output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            queue=queue,
            sampler_state=sampler.state_dict(),
            loader_generator_state=loader_generator.get_state(),
            run_state=checkpoint_run_state(),
            run_identity=identity,
            optimizer_parameter_groups=optimizer_group_metadata,
            step=global_step,
            config_text=config_text,
            name=name,
        )

    if resume is None and validation is not None and initial_validation_loss is not None:
        # The unmodified official initialization is a valid model-selection
        # candidate.  Without it, a run that harms validation could still label
        # its least harmful training step as "best".
        best_validation_loss = initial_validation_loss
        best_validation_step = 0
        best_checkpoint = output_dir / "best.pt"
        initial_checkpoint = save_current_checkpoint()
        best_checkpoint = save_best_checkpoint(output_dir, initial_checkpoint)
        print(f"checkpoint={initial_checkpoint} best_checkpoint={best_checkpoint}", flush=True)
        if rolling_checkpoints:
            save_current_checkpoint("latest.pt")
    elif resume is not None and rolling_checkpoints:
        # A rollback starts a new continuation at the restored state. Do not
        # leave latest.pt pointing into the discarded future branch.
        save_current_checkpoint("latest.pt")

    start = time.monotonic()
    while global_step < target_steps:
        for batch in loader:
            if not batch["ids"]:
                raise RuntimeError("DataLoader produced an empty batch")
            pixels = batch["pixels"].to(device, non_blocking=True)
            tokens = tokenizer.tokenize(batch["captions"]).to(device, non_blocking=True)
            if tokens.shape[0] != pixels.shape[0]:
                raise RuntimeError(
                    "Tokenizer batch size does not match image batch size for samples: "
                    + ", ".join(batch["ids"][:5])
                )
            with _autocast(device, config.train.precision):
                image_features, text_features, logit_scale, _, _ = model(pixels, tokens)
                if config.train.contrastive_objective == "single_positive":
                    loss_output = symmetric_contrastive_loss(
                        image_features, text_features, logit_scale, queue=queue
                    )
                else:
                    loss_output = symmetric_group_contrastive_loss(
                        image_features, text_features, logit_scale, batch["group_ids"],
                        objective=config.train.contrastive_objective,
                    )
                _assert_finite_loss(loss_output.loss, batch["ids"])
                loss = loss_output.loss / config.train.gradient_accumulation
            with torch.no_grad():
                if config.train.contrastive_objective == "single_positive":
                    in_batch_loss = symmetric_contrastive_loss(
                        image_features, text_features, logit_scale, queue=None
                    ).loss
                else:
                    in_batch_loss = symmetric_group_contrastive_loss(
                        image_features, text_features, logit_scale, batch["group_ids"],
                        objective=config.train.contrastive_objective,
                    ).loss
                _assert_finite_loss(in_batch_loss, batch["ids"])
            scaler.scale(loss).backward()
            queue.enqueue(image_features, text_features)
            sampler.advance()
            if group_ids is not None:
                _update_group_diagnostics(group_diagnostics, batch["group_ids"])
                seen_training_indices.update(batch["indices"])
            optimizer_step_loss += float(loss_output.loss.detach())
            optimizer_step_in_batch_loss += float(in_batch_loss)
            micro_step += 1
            if micro_step % config.train.gradient_accumulation:
                continue

            scaler.unscale_(optimizer)
            used_learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
            assert_visual_backbone_frozen(model, require_eval=True)
            if scaler.is_enabled() and any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for _, parameter in named_parameters
            ):
                missing = [name for name, parameter in named_parameters if parameter.grad is None]
                if missing:
                    raise RuntimeError("No gradient was produced for: " + ", ".join(missing[:5]))
                # GradScaler recorded the overflow during unscale_. Let it skip
                # the update and reduce its scale without advancing the schedule.
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                skipped_optimizer_steps += 1
                consecutive_overflows += 1
                optimizer_step_loss = 0.0
                optimizer_step_in_batch_loss = 0.0
                print(
                    json.dumps(
                        {
                            "event": "fp16_overflow",
                            "step": global_step,
                            "skipped_optimizer_steps": skipped_optimizer_steps,
                            "scale": scaler.get_scale(),
                        }
                    ),
                    flush=True,
                )
                if consecutive_overflows >= 32:
                    raise FloatingPointError("32 consecutive FP16 optimizer updates overflowed")
                continue
            last_gradient_norm, last_group_gradient_norms = _checked_grad_norm(
                named_parameters,
                named_parameter_groups,
                config.train.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            consecutive_overflows = 0
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            with torch.no_grad():
                if model.logit_scale.requires_grad:
                    model.logit_scale.clamp_(0.0, math.log(100.0))
            global_step += 1
            step_loss = optimizer_step_loss / config.train.gradient_accumulation
            step_in_batch_loss = optimizer_step_in_batch_loss / config.train.gradient_accumulation
            optimizer_step_loss = 0.0
            optimizer_step_in_batch_loss = 0.0
            if initial_loss is None:
                initial_loss = step_loss
                initial_in_batch_loss = step_in_batch_loss
            final_loss = step_loss
            final_in_batch_loss = step_in_batch_loss
            running_loss += step_loss
            running_in_batch_loss += step_in_batch_loss
            running_log_steps += 1

            if (
                fixed_monitor_batch is not None
                and global_step % config.train.fixed_monitor_every == 0
            ):
                fixed_monitor_record = _evaluate_fixed_monitor(
                    model=model,
                    tokenizer=tokenizer,
                    batch=fixed_monitor_batch,
                    device=device,
                    precision=config.train.precision,
                )
                fixed_monitor_record["step"] = global_step
                _append_jsonl(fixed_monitor_path, fixed_monitor_record)
                final_fixed_monitor_loss = float(fixed_monitor_record["loss"])

            improved_validation = False
            if validation is not None and (
                global_step % config.train.validation_every == 0
                or global_step == config.train.max_steps
            ):
                _, validation_loader = validation
                validation_record = _evaluate_validation(
                    model=model,
                    tokenizer=tokenizer,
                    loader=validation_loader,
                    device=device,
                    precision=config.train.precision,
                    loss_batch_size=config.data.validation_batch_size,
                )
                validation_record["step"] = global_step
                _append_jsonl(validation_path, validation_record)
                final_validation_loss = float(validation_record["loss"])
                if initial_validation_loss is None:
                    initial_validation_loss = final_validation_loss
                validation_evaluations += 1
                last_validation_step = global_step
                if best_validation_loss is None or final_validation_loss < best_validation_loss:
                    best_validation_loss = final_validation_loss
                    best_validation_step = global_step
                    best_checkpoint = output_dir / "best.pt"
                    improved_validation = True
                print("validation=" + json.dumps(validation_record, ensure_ascii=False), flush=True)

            if global_step % config.train.log_every == 0 or global_step == target_steps:
                elapsed = time.monotonic() - start
                record = {
                    "step": global_step,
                    "loss": running_loss / running_log_steps,
                    "in_batch_loss": running_in_batch_loss / running_log_steps,
                    "averaged_optimizer_steps": running_log_steps,
                    "lr": next(iter(used_learning_rates.values())),
                    "learning_rates": used_learning_rates,
                    "next_learning_rates": {
                        group["name"]: learning_rate
                        for group, learning_rate in zip(
                            optimizer.param_groups, scheduler.get_last_lr(), strict=True
                        )
                    },
                    "logit_scale": float(model.logit_scale.exp().detach()),
                    "gradient_norm": last_gradient_norm,
                    "gradient_norms": last_group_gradient_norms,
                    "queue_size": len(queue),
                    "skipped_optimizer_steps": skipped_optimizer_steps,
                    "peak_cuda_allocated_bytes": _peak_cuda_allocated_bytes(device),
                    "elapsed_seconds": elapsed,
                }
                if group_diagnostics:
                    record["caption_group_diagnostics"] = _group_diagnostics_report(
                        group_diagnostics
                    )
                    record["unique_training_images_seen"] = len(seen_training_indices)
                    record["training_image_pool_coverage"] = (
                        len(seen_training_indices) / len(dataset)
                    )
                _append_jsonl(metrics_path, record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                running_loss = 0.0
                running_in_batch_loss = 0.0
                running_log_steps = 0
                group_diagnostics = {}
                start = time.monotonic()

            if improved_validation:
                path = save_current_checkpoint("latest.pt" if rolling_checkpoints else None)
                best_checkpoint = save_best_checkpoint(output_dir, path)
                print(f"checkpoint={path} best_checkpoint={best_checkpoint}", flush=True)
                last_checkpoint_step = global_step
            elif global_step % config.train.checkpoint_every == 0:
                path = save_current_checkpoint("latest.pt" if rolling_checkpoints else None)
                print(f"checkpoint={path}", flush=True)
                last_checkpoint_step = global_step
            if global_step >= target_steps:
                break

    if global_step != last_checkpoint_step:
        final_checkpoint = save_current_checkpoint("latest.pt" if rolling_checkpoints else None)
    else:
        final_checkpoint = output_dir / (
            "latest.pt" if rolling_checkpoints else f"step_{global_step:07d}.pt"
        )
    if (
        initial_loss is None
        or final_loss is None
        or initial_in_batch_loss is None
        or final_in_batch_loss is None
        or last_gradient_norm is None
        or last_group_gradient_norms is None
    ):
        raise RuntimeError("Training completed without an optimizer step")
    summary: dict[str, Any] = {
        "format_version": 2,
        "steps": global_step,
        "device": str(device),
        "log_every": config.train.log_every,
        "target_steps": config.train.max_steps,
        "completed": global_step == config.train.max_steps,
        "resumed_from": None if resumed_from is None else str(resumed_from),
        "samples_in_manifest": len(dataset),
        "batch_size": config.train.batch_size,
        "gradient_accumulation": config.train.gradient_accumulation,
        "micro_steps": micro_step,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "checkpoint_policy": config.train.checkpoint_policy,
        "log_at_end": True,
        "vision_head_drop_path": config.model.vision_head_drop_path,
        "train_augmentation": config.data.train_augmentation,
        "shuffle_train": config.data.shuffle_train,
        "queue_size": config.train.queue_size,
        "contrastive_objective": config.train.contrastive_objective,
        "caption_sampling": config.data.caption_sampling,
        "images_per_caption": config.data.images_per_caption,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_in_batch_loss": initial_in_batch_loss,
        "final_in_batch_loss": final_in_batch_loss,
        "last_gradient_norm": last_gradient_norm,
        "last_group_gradient_norms": last_group_gradient_norms,
        "optimizer_parameter_groups": {
            **optimizer_group_metadata,
            "final_learning_rates": {
                group["name"]: learning_rate
                for group, learning_rate in zip(
                    optimizer.param_groups, scheduler.get_last_lr(), strict=True
                )
            },
        },
        "visual_backbone_permanently_frozen": True,
        "all_losses_finite": True,
        "all_gradients_finite": skipped_optimizer_steps == 0,
        "all_applied_gradients_finite": True,
        "peak_cuda_allocated_bytes": _peak_cuda_allocated_bytes(device),
        "final_checkpoint": str(final_checkpoint),
        "text_lora": text_lora_metadata,
    }
    if group_ids is not None:
        group_counts = Counter(group_ids)
        summary["caption_groups"] = {
            "count": len(group_counts),
            "singleton_groups": sum(count == 1 for count in group_counts.values()),
            "max_images_per_group": max(group_counts.values()),
            "epoch_definition": "one_visit_per_caption_group_not_all_images",
            "completed_group_epochs": sampler.epoch,
            "current_epoch_batch_offset": sampler.batch_offset,
            "sample_exposures": micro_step * config.train.batch_size,
            "unique_images_seen": len(seen_training_indices),
            "image_pool_coverage": len(seen_training_indices) / len(dataset),
            "unique_caption_groups_seen": len({group_ids[index]
                                              for index in seen_training_indices}),
        }
    if fixed_monitor_batch is not None:
        if initial_fixed_monitor_loss is None or final_fixed_monitor_loss is None:
            raise RuntimeError("Fixed monitor was configured but did not produce a result")
        summary["fixed_monitor"] = {
            "samples": len(fixed_monitor_batch["ids"]),
            "every": config.train.fixed_monitor_every,
            "initial_loss": initial_fixed_monitor_loss,
            "final_loss": final_fixed_monitor_loss,
        }
    if validation is not None:
        validation_dataset, _ = validation
        summary["validation"] = {
            "samples": len(validation_dataset),
            "batch_size": config.data.validation_batch_size,
            "forward_batch_size": _validation_loader_settings(config)[0],
            "num_workers": _validation_loader_settings(config)[1],
            "every": config.train.validation_every,
            "evaluations": validation_evaluations,
            "last_step": last_validation_step,
            "validation_at_end": True,
            "initial_loss": initial_validation_loss,
            "final_loss": final_validation_loss,
            "best_loss": best_validation_loss,
            "best_step": best_validation_step,
            "best_checkpoint": None if best_checkpoint is None else str(best_checkpoint),
            "selection_includes_step_zero": config.train.validation_at_start,
            "all_losses_finite": True,
            "positive_definition": "manifest_row_one_to_one",
        }
    summary_path = _write_json_atomic(output_dir / "training_summary.json", summary)
    return summary_path
