from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ResidualEmbeddingAdapter(nn.Module):
    """A zero-initialized bottleneck adapter for normalized embeddings."""

    def __init__(self, embedding_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("embedding_dim and bottleneck_dim must be positive")
        self.norm = nn.LayerNorm(embedding_dim)
        self.down = nn.Linear(embedding_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, embedding_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(features))))
        return F.normalize(features + residual, dim=-1)


class DINOtxtWithImageAdapter(nn.Module):
    """Apply a small residual adapter after the frozen dino.txt image embedding.

    Patch tokens remain the official model outputs.  The wrapper exposes the
    official tower attributes because the trainer uses them to enforce eval mode
    for frozen modules.
    """

    def __init__(self, base_model: nn.Module, *, embedding_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.image_adapter = ResidualEmbeddingAdapter(embedding_dim, bottleneck_dim)

    @property
    def visual_model(self) -> Any:
        return self.base_model.visual_model

    @property
    def text_model(self) -> Any:
        return self.base_model.text_model

    @property
    def logit_scale(self) -> torch.nn.Parameter:
        return self.base_model.logit_scale

    def forward(self, pixels: torch.Tensor, tokens: torch.Tensor):
        image_features, text_features, logit_scale, patch_tokens, backbone_patch_tokens = (
            self.base_model(pixels, tokens)
        )
        image_features = self.image_adapter(image_features)
        return image_features, text_features, logit_scale, patch_tokens, backbone_patch_tokens


def add_image_embedding_adapter(
    model: nn.Module, *, bottleneck_dim: int, embedding_dim: int = 2048
) -> nn.Module:
    if bottleneck_dim < 0:
        raise ValueError("bottleneck_dim must be nonnegative")
    if bottleneck_dim == 0:
        return model
    return DINOtxtWithImageAdapter(
        model, embedding_dim=embedding_dim, bottleneck_dim=bottleneck_dim
    )
