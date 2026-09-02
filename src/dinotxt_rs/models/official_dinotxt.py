from __future__ import annotations

from pathlib import Path
from typing import Any


def load_official_dinotxt(
    dinov3_repo: str | Path,
    backbone_weights: str | Path,
    dinotxt_weights: str | Path,
    bpe_vocab: str | Path,
) -> tuple[Any, Any]:
    """Load Meta's exact ViT-L dino.txt implementation from a pinned local checkout."""
    import torch

    repo = Path(dinov3_repo).resolve()
    for path in (repo, Path(backbone_weights), Path(dinotxt_weights), Path(bpe_vocab)):
        if not path.exists():
            raise FileNotFoundError(path)
    model, tokenizer = torch.hub.load(
        str(repo),
        "dinov3_vitl16_dinotxt_tet1280d20h24l",
        source="local",
        pretrained=True,
        weights=str(Path(dinotxt_weights).resolve()),
        backbone_weights=str(Path(backbone_weights).resolve()),
        bpe_path_or_url=str(Path(bpe_vocab).resolve()),
        check_hash=True,
    )
    return model, tokenizer


def configure_trainable_parameters(
    model: Any,
    *,
    text_last_k: int,
    train_vision_head: bool,
    train_text_projection: bool,
    train_logit_scale: bool,
) -> dict[str, int]:
    """Apply one explicit freeze policy and return total/trainable parameter counts."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if train_vision_head:
        for parameter in model.visual_model.head.parameters():
            parameter.requires_grad_(True)

    text_backbone = model.text_model.backbone
    blocks = text_backbone.blocks
    if text_last_k > len(blocks):
        raise ValueError(f"text_last_k={text_last_k} exceeds text depth={len(blocks)}")
    if text_last_k:
        for block in blocks[-text_last_k:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in text_backbone.ln_final.parameters():
            parameter.requires_grad_(True)

    if train_text_projection:
        for parameter in model.text_model.head.parameters():
            parameter.requires_grad_(True)
    model.logit_scale.requires_grad_(train_logit_scale)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise ValueError("Freeze policy left no trainable parameters")
    return {"total": total, "trainable": trainable}


def trainable_state_dict(model: Any) -> dict[str, Any]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in names
    }
