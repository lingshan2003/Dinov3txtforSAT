from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """Keep an original linear map and add an unmerged, initially zero low-rank update."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0 or rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA rank must be positive and fit both linear dimensions")
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA requires finite positive alpha and dropout in [0, 1)")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        options = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, **options))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, **options))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + update * self.scaling


def add_text_lora(
    model: Any, *, rank: int, alpha: float, dropout: float, include_projection: bool
) -> Any:
    """Adapt every official text block's QKV/output/MLP maps, optionally its final projection.

    Targets are explicit for the pinned dino.txt architecture. Validate all of them before
    replacing anything, so an upstream mismatch cannot silently train only some layers.
    """
    if rank == 0:
        return model
    if rank < 0 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("Invalid text LoRA rank, alpha, or dropout")
    blocks = model.text_model.backbone.blocks
    if not blocks:
        raise ValueError("Text LoRA requires at least one Transformer block")
    targets = []
    for index, block in enumerate(blocks):
        for parent_name, names in (
            ("attention", ("qkv", "proj")),
            ("feed_forward", ("fc1", "fc2")),
        ):
            parent = getattr(block, parent_name, None)
            for name in names:
                label = f"text_model.backbone.blocks.{index}.{parent_name}.{name}"
                module = getattr(parent, name, None)
                if not isinstance(module, nn.Linear):
                    raise TypeError(f"Expected an original nn.Linear at {label}")
                targets.append((parent, name, module, label))
    if include_projection:
        parent = model.text_model.head
        module = getattr(parent, "linear_projection", None)
        if not isinstance(module, nn.Linear):
            raise TypeError("Expected nn.Linear at text_model.head.linear_projection")
        targets.append((parent, "linear_projection", module, "text_model.head.linear_projection"))
    if any(rank > min(module.in_features, module.out_features) for _, _, module, _ in targets):
        raise ValueError("Text LoRA rank exceeds a target linear dimension")
    for parent, name, module, _ in targets:
        setattr(parent, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
    model.text_lora_metadata = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "blocks": len(blocks),
        "include_projection": include_projection,
        "modules": [label for _, _, _, label in targets],
        "parameters": sum(
            rank * (module.in_features + module.out_features) for _, _, module, _ in targets
        ),
    }
    return model
