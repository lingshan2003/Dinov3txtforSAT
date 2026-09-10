import pytest
import torch

from dinotxt_rs.config import TrainConfig
from dinotxt_rs.models.official_dinotxt import (
    assert_visual_backbone_frozen,
    configure_trainable_parameters,
    optimizer_parameter_groups,
)


class FakeParameter:
    def __init__(self, size: int = 1) -> None:
        self.size = size
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> "FakeParameter":
        self.requires_grad = value
        return self

    def numel(self) -> int:
        return self.size


class FakeModule:
    def __init__(self, *parameters: FakeParameter, **children: object) -> None:
        self._parameters = list(parameters)
        for name, child in children.items():
            setattr(self, name, child)

    def parameters(self):
        yield from self._parameters
        for name, value in vars(self).items():
            if name == "_parameters":
                continue
            if isinstance(value, FakeParameter):
                yield value
            elif isinstance(value, FakeModule):
                yield from value.parameters()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, FakeModule):
                        yield from item.parameters()


def test_explicit_last_k_freeze_policy() -> None:
    backbone_parameter = FakeParameter(10)
    vision_head_parameter = FakeParameter(2)
    text_blocks = [FakeModule(FakeParameter(3)) for _ in range(4)]
    text_norm_parameter = FakeParameter(1)
    text_head_parameter = FakeParameter(2)
    scale = FakeParameter(1)
    model = FakeModule(
        visual_model=FakeModule(
            backbone=FakeModule(backbone_parameter), head=FakeModule(vision_head_parameter)
        ),
        text_model=FakeModule(
            backbone=FakeModule(blocks=text_blocks, ln_final=FakeModule(text_norm_parameter)),
            head=FakeModule(text_head_parameter),
        ),
    )
    model.logit_scale = scale

    counts = configure_trainable_parameters(
        model,
        text_last_k=2,
        train_vision_head=True,
        train_text_projection=True,
        train_logit_scale=True,
    )

    assert not backbone_parameter.requires_grad
    assert vision_head_parameter.requires_grad
    assert not text_blocks[0]._parameters[0].requires_grad
    assert not text_blocks[1]._parameters[0].requires_grad
    assert text_blocks[2]._parameters[0].requires_grad
    assert text_blocks[3]._parameters[0].requires_grad
    assert text_norm_parameter.requires_grad
    assert text_head_parameter.requires_grad
    assert scale.requires_grad
    assert counts == {"total": 28, "trainable": 12}


class TorchPolicyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual_model = torch.nn.Module()
        self.visual_model.backbone = torch.nn.Linear(2, 2)
        self.visual_model.head = torch.nn.Linear(2, 2)
        self.text_model = torch.nn.Module()
        self.text_model.backbone = torch.nn.Module()
        self.text_model.backbone.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.text_model.backbone.ln_final = torch.nn.LayerNorm(2)
        self.text_model.head = torch.nn.Linear(2, 2)
        self.image_adapter = torch.nn.Linear(2, 2)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))


def test_named_optimizer_groups_are_disjoint_and_keep_visual_backbone_frozen() -> None:
    model = TorchPolicyModel()
    configure_trainable_parameters(
        model,
        text_last_k=0,
        train_vision_head=True,
        train_text_projection=False,
        train_logit_scale=False,
        train_image_adapter=True,
    )
    config = TrainConfig(
        image_adapter_learning_rate=1e-4,
        vision_head_learning_rate=1e-5,
    )

    groups, metadata, named = optimizer_parameter_groups(model, config)

    assert [group["name"] for group in groups] == ["image_adapter", "vision_head"]
    assert [group["lr"] for group in groups] == [1e-4, 1e-5]
    assert [group["name"] for group in metadata["groups"]] == [
        "image_adapter",
        "vision_head",
    ]
    assert metadata["visual_backbone_permanently_frozen"]
    assert not any(
        parameter.requires_grad for parameter in model.visual_model.backbone.parameters()
    )
    assert not named["text_projection"] and not named["text_backbone"]


def test_visual_backbone_freeze_violation_is_rejected() -> None:
    model = TorchPolicyModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    next(model.visual_model.backbone.parameters()).requires_grad_(True)

    with pytest.raises(RuntimeError, match="permanently frozen"):
        assert_visual_backbone_frozen(model)
