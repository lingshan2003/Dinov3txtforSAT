from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from dinotxt_rs.models.official_dinotxt import trainable_state_dict


def save_checkpoint(
    output_dir: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    step: int,
    config_text: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"step_{step:07d}.pt"
    temporary = destination.with_suffix(".pt.part")
    payload = {
        "format_version": 1,
        "step": step,
        "trainable_model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config_toml": config_text,
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination

