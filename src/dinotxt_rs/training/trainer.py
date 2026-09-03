from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dinotxt_rs.config import Config
from dinotxt_rs.data import ImageTextDataset, collate_image_text, make_transform
from dinotxt_rs.losses import EmbeddingQueue, symmetric_contrastive_loss
from dinotxt_rs.training.checkpoint import save_checkpoint
from dinotxt_rs.training.provenance import write_provenance


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
    named_parameters: list[tuple[str, torch.nn.Parameter]], max_grad_norm: float
) -> float:
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

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in named_parameters],
        max_grad_norm,
        error_if_nonfinite=True,
    )
    value = float(gradient_norm)
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite gradient norm: {value}")
    return value


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


def train(config: Config, model: Any, tokenizer: Any) -> Path:
    seed_everything(config.experiment.seed)
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    _set_training_mode(model)

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
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.experiment.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=config.data.shuffle_train,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
        drop_last=True,
        collate_fn=collate_image_text,
        generator=loader_generator,
    )
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = _scheduler(optimizer, config.train.warmup_steps, config.train.max_steps)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config.train.precision == "fp16"
    )
    queue = EmbeddingQueue(config.train.queue_size)
    output_dir = config.experiment.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = write_provenance(config)
    print(f"provenance={provenance_path}", flush=True)
    metrics_path = output_dir / "metrics.jsonl"
    fixed_monitor_path = output_dir / "fixed_monitor.jsonl"
    config_text = config.source.read_text(encoding="utf-8")
    (output_dir / "config.toml").write_text(config_text, encoding="utf-8")

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_in_batch_loss = 0.0
    optimizer_step_loss = 0.0
    optimizer_step_in_batch_loss = 0.0
    initial_loss: float | None = None
    final_loss: float | None = None
    initial_in_batch_loss: float | None = None
    final_in_batch_loss: float | None = None
    initial_fixed_monitor_loss: float | None = None
    final_fixed_monitor_loss: float | None = None
    last_gradient_norm: float | None = None
    last_checkpoint_step = 0
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
    start = time.monotonic()
    while global_step < config.train.max_steps:
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
                loss_output = symmetric_contrastive_loss(
                    image_features, text_features, logit_scale, queue=queue
                )
                _assert_finite_loss(loss_output.loss, batch["ids"])
                loss = loss_output.loss / config.train.gradient_accumulation
            with torch.no_grad():
                in_batch_loss = symmetric_contrastive_loss(
                    image_features, text_features, logit_scale, queue=None
                ).loss
                _assert_finite_loss(in_batch_loss, batch["ids"])
            scaler.scale(loss).backward()
            queue.enqueue(image_features, text_features)
            optimizer_step_loss += float(loss_output.loss.detach())
            optimizer_step_in_batch_loss += float(in_batch_loss)
            micro_step += 1
            if micro_step % config.train.gradient_accumulation:
                continue

            scaler.unscale_(optimizer)
            last_gradient_norm = _checked_grad_norm(named_parameters, config.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            with torch.no_grad():
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

            if global_step % config.train.log_every == 0:
                elapsed = time.monotonic() - start
                record = {
                    "step": global_step,
                    "loss": running_loss / config.train.log_every,
                    "in_batch_loss": running_in_batch_loss / config.train.log_every,
                    "lr": scheduler.get_last_lr()[0],
                    "logit_scale": float(model.logit_scale.exp().detach()),
                    "gradient_norm": last_gradient_norm,
                    "queue_size": len(queue),
                    "peak_cuda_allocated_bytes": _peak_cuda_allocated_bytes(device),
                    "elapsed_seconds": elapsed,
                }
                _append_jsonl(metrics_path, record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                running_loss = 0.0
                running_in_batch_loss = 0.0
                start = time.monotonic()

            if global_step % config.train.checkpoint_every == 0:
                path = save_checkpoint(
                    output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    config_text=config_text,
                )
                print(f"checkpoint={path}", flush=True)
                last_checkpoint_step = global_step
            if global_step >= config.train.max_steps:
                break

    if global_step != last_checkpoint_step:
        final_checkpoint = save_checkpoint(
            output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            config_text=config_text,
        )
    else:
        final_checkpoint = output_dir / f"step_{global_step:07d}.pt"
    if (
        initial_loss is None
        or final_loss is None
        or initial_in_batch_loss is None
        or final_in_batch_loss is None
        or last_gradient_norm is None
    ):
        raise RuntimeError("Training completed without an optimizer step")
    summary = {
        "format_version": 1,
        "steps": global_step,
        "samples_in_manifest": len(dataset),
        "batch_size": config.train.batch_size,
        "gradient_accumulation": config.train.gradient_accumulation,
        "train_augmentation": config.data.train_augmentation,
        "shuffle_train": config.data.shuffle_train,
        "queue_size": config.train.queue_size,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_in_batch_loss": initial_in_batch_loss,
        "final_in_batch_loss": final_in_batch_loss,
        "last_gradient_norm": last_gradient_norm,
        "all_losses_finite": True,
        "all_gradients_finite": True,
        "peak_cuda_allocated_bytes": _peak_cuda_allocated_bytes(device),
        "final_checkpoint": str(final_checkpoint),
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
    summary_path = _write_json_atomic(output_dir / "training_summary.json", summary)
    return summary_path
