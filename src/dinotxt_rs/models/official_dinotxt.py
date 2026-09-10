from __future__ import annotations

from pathlib import Path
from typing import Any

OPTIMIZER_GROUP_NAMES = (
    "image_adapter",
    "vision_head",
    "text_projection",
    "text_backbone",
    "logit_scale",
)


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
    train_image_adapter: bool = False,
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

    image_adapter = getattr(model, "image_adapter", None)
    if train_image_adapter:
        if image_adapter is None:
            raise ValueError("train_image_adapter requires a configured image adapter")
        for parameter in image_adapter.parameters():
            parameter.requires_grad_(True)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise ValueError("Freeze policy left no trainable parameters")
    assert_visual_backbone_frozen(model)
    return {"total": total, "trainable": trainable}


def assert_visual_backbone_frozen(model: Any, *, require_eval: bool = False) -> None:
    """Enforce the permanent visual-backbone freeze invariant."""
    backbone = model.visual_model.backbone
    named_parameters = getattr(backbone, "named_parameters", None)
    parameters = (
        list(named_parameters())
        if named_parameters is not None
        else [(str(index), parameter) for index, parameter in enumerate(backbone.parameters())]
    )
    trainable = [name for name, parameter in parameters if parameter.requires_grad]
    if trainable:
        raise RuntimeError(
            "Visual backbone must remain permanently frozen; trainable parameters: "
            + ", ".join(trainable[:5])
        )
    gradients = [
        name
        for name, parameter in parameters
        if getattr(parameter, "grad", None) is not None
    ]
    if gradients:
        raise RuntimeError(
            "Frozen visual backbone unexpectedly received gradients: "
            + ", ".join(gradients[:5])
        )
    if require_eval and getattr(backbone, "training", False):
        raise RuntimeError("Frozen visual backbone must remain in eval mode")


def optimizer_parameter_groups(
    model: Any,
    train_config: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[tuple[str, Any]]]]:
    """Build disjoint named AdamW groups for the allowed alignment/text modules."""
    assert_visual_backbone_frozen(model)
    modules = {
        "image_adapter": getattr(model, "image_adapter", None),
        "vision_head": model.visual_model.head,
        "text_projection": model.text_model.head,
        "text_backbone": model.text_model.backbone,
    }
    parameter_membership: dict[int, str] = {}
    for group_name, module in modules.items():
        if module is None:
            continue
        for parameter in module.parameters():
            key = id(parameter)
            previous = parameter_membership.get(key)
            if previous is not None and previous != group_name:
                raise RuntimeError(
                    f"Optimizer parameter belongs to both {previous!r} and {group_name!r}"
                )
            parameter_membership[key] = group_name
    parameter_membership[id(model.logit_scale)] = "logit_scale"

    grouped: dict[str, list[tuple[str, Any]]] = {
        name: [] for name in OPTIMIZER_GROUP_NAMES
    }
    unclassified: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = parameter_membership.get(id(parameter))
        if group_name is None:
            unclassified.append(name)
        else:
            grouped[group_name].append((name, parameter))
    if unclassified:
        raise RuntimeError(
            "Trainable parameters are outside the approved alignment/text modules: "
            + ", ".join(unclassified[:5])
        )

    optimizer_groups: list[dict[str, Any]] = []
    metadata_groups: list[dict[str, Any]] = []
    covered: set[int] = set()
    for group_name in OPTIMIZER_GROUP_NAMES:
        named_parameters = grouped[group_name]
        if not named_parameters:
            continue
        parameter_ids = {id(parameter) for _, parameter in named_parameters}
        if covered & parameter_ids:
            raise RuntimeError(f"Optimizer group {group_name!r} overlaps an earlier group")
        covered.update(parameter_ids)
        learning_rate = getattr(train_config, f"{group_name}_learning_rate")
        if learning_rate is None:
            learning_rate = train_config.learning_rate
        weight_decay = getattr(train_config, f"{group_name}_weight_decay")
        if weight_decay is None:
            weight_decay = train_config.weight_decay
        names = [name for name, _ in named_parameters]
        parameter_count = sum(parameter.numel() for _, parameter in named_parameters)
        optimizer_groups.append(
            {
                "name": group_name,
                "params": [parameter for _, parameter in named_parameters],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
        metadata_groups.append(
            {
                "name": group_name,
                "initial_learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "parameter_tensors": len(named_parameters),
                "parameters": parameter_count,
                "parameter_names": names,
            }
        )
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if not expected or covered != expected:
        raise RuntimeError("Optimizer groups do not exactly cover all trainable parameters")
    metadata = {
        "format_version": 1,
        "visual_backbone_permanently_frozen": True,
        "groups": metadata_groups,
        "trainable_parameter_tensors": len(expected),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    return optimizer_groups, metadata, grouped


def trainable_state_dict(model: Any) -> dict[str, Any]:
    assert_visual_backbone_frozen(model)
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in names
    }
