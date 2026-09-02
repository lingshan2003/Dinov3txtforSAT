from dinotxt_rs.models.official_dinotxt import configure_trainable_parameters


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
