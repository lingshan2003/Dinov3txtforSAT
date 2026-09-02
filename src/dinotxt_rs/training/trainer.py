from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
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


def train(config: Config, model: Any, tokenizer: Any) -> None:
    seed_everything(config.experiment.seed)
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model.to(device)
    model.train()
    # The backbone stays deterministic even while the trainable alignment heads are in train mode.
    model.visual_model.backbone.eval()

    transform = make_transform(config.model.image_size, config.model.backbone_domain, train=True)
    dataset = ImageTextDataset(config.data.train_manifest, transform)
    if len(dataset) < config.train.batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, fewer than batch_size={config.train.batch_size}"
        )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
        drop_last=True,
        collate_fn=collate_image_text,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = _scheduler(optimizer, config.train.warmup_steps, config.train.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.precision == "fp16")
    queue = EmbeddingQueue(config.train.queue_size)
    output_dir = config.experiment.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = write_provenance(config)
    print(f"provenance={provenance_path}", flush=True)
    metrics_path = output_dir / "metrics.jsonl"
    config_text = config.source.read_text(encoding="utf-8")
    (output_dir / "config.toml").write_text(config_text, encoding="utf-8")

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    start = time.monotonic()
    while global_step < config.train.max_steps:
        for batch in loader:
            pixels = batch["pixels"].to(device, non_blocking=True)
            tokens = tokenizer.tokenize(batch["captions"]).to(device, non_blocking=True)
            with _autocast(device, config.train.precision):
                image_features, text_features, logit_scale, _, _ = model(pixels, tokens)
                loss_output = symmetric_contrastive_loss(
                    image_features, text_features, logit_scale, queue=queue
                )
                loss = loss_output.loss / config.train.gradient_accumulation
            scaler.scale(loss).backward()
            queue.enqueue(image_features, text_features)
            running_loss += float(loss_output.loss.detach())
            micro_step += 1
            if micro_step % config.train.gradient_accumulation:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0.0, math.log(100.0))
            global_step += 1

            if global_step % config.train.log_every == 0:
                elapsed = time.monotonic() - start
                record = {
                    "step": global_step,
                    "loss": running_loss
                    / (config.train.log_every * config.train.gradient_accumulation),
                    "lr": scheduler.get_last_lr()[0],
                    "logit_scale": float(model.logit_scale.exp().detach()),
                    "queue_size": len(queue),
                    "elapsed_seconds": elapsed,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps(record, ensure_ascii=False), flush=True)
                running_loss = 0.0
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
            if global_step >= config.train.max_steps:
                break

    save_checkpoint(
        output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        config_text=config_text,
    )
