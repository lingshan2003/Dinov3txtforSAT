from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def model_paths(config: Any) -> list[Path]:
    return [
        config.model.dinov3_repo,
        config.model.backbone_weights,
        config.model.dinotxt_weights,
        config.model.bpe_vocab,
    ]


def assert_finite(name: str, value: Any) -> None:
    import torch

    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def assert_shapes(
    image_features: Any,
    text_features: Any,
    scale: Any,
    patch_tokens: Any,
    backbone_patch_tokens: Any,
    batch_size: int,
) -> None:
    expected_features = (batch_size, 2048)
    if tuple(image_features.shape) != expected_features:
        raise RuntimeError(f"Unexpected image feature shape: {tuple(image_features.shape)}")
    if tuple(text_features.shape) != expected_features:
        raise RuntimeError(f"Unexpected text feature shape: {tuple(text_features.shape)}")
    token_outputs = (
        ("patch_tokens", patch_tokens),
        ("backbone_patch_tokens", backbone_patch_tokens),
    )
    for name, value in token_outputs:
        if value.ndim != 3 or tuple(value.shape[:1]) != (batch_size,) or value.shape[-1] != 1024:
            raise RuntimeError(f"Unexpected {name} shape: {tuple(value.shape)}")
    if scale.numel() != 1:
        raise RuntimeError(f"Expected scalar logit scale, got shape {tuple(scale.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load dino.txt and run one synthetic forward pass")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Synthetic forward batch size; use this to preflight evaluation memory",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    import torch

    from dinotxt_rs.config import load_config
    from dinotxt_rs.models import configure_trainable_parameters, load_official_dinotxt

    config = load_config(args.config)
    missing = [str(path) for path in model_paths(config) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required model paths:\n" + "\n".join(missing))
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA but torch.cuda.is_available() is False")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, tokenizer = load_official_dinotxt(
        config.model.dinov3_repo,
        config.model.backbone_weights,
        config.model.dinotxt_weights,
        config.model.bpe_vocab,
    )
    counts = configure_trainable_parameters(
        model,
        text_last_k=config.model.text_last_k,
        train_vision_head=config.model.train_vision_head,
        train_text_projection=config.model.train_text_projection,
        train_logit_scale=config.model.train_logit_scale,
    )
    model.to(device).eval()
    prompt_templates = ["a satellite image of forest", "a satellite image of water"]
    prompts = [prompt_templates[index % len(prompt_templates)] for index in range(args.batch_size)]
    pixels = torch.randn(
        len(prompts),
        3,
        config.model.image_size,
        config.model.image_size,
        device=device,
    )
    tokens = tokenizer.tokenize(prompts).to(device)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and config.train.precision == "bf16"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        image_features, text_features, scale, patch_tokens, backbone_patch_tokens = model(
            pixels, tokens
        )
    assert_shapes(
        image_features,
        text_features,
        scale,
        patch_tokens,
        backbone_patch_tokens,
        batch_size=len(prompts),
    )
    for name, value in (
        ("image_features", image_features),
        ("text_features", text_features),
        ("logit_scale", scale),
        ("patch_tokens", patch_tokens),
        ("backbone_patch_tokens", backbone_patch_tokens),
    ):
        assert_finite(name, value)
    if float(scale) <= 0:
        raise RuntimeError(f"Expected positive logit scale, got {float(scale)}")

    memory: dict[str, int] = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    print(
        json.dumps(
            {
                "backbone_domain": config.model.backbone_domain,
                "device": str(device),
                "parameters": counts,
                "image_features": tuple(image_features.shape),
                "text_features": tuple(text_features.shape),
                "patch_tokens": tuple(patch_tokens.shape),
                "backbone_patch_tokens": tuple(backbone_patch_tokens.shape),
                "logit_scale": float(scale),
                "cuda_memory": memory,
            }
        )
    )


if __name__ == "__main__":
    main()
