import pytest
import torch
import torch.nn.functional as F
from test_text_lora import TinyTextModel

from dinotxt_rs.config import load_config
from dinotxt_rs.models import (
    add_image_embedding_adapter,
    add_text_lora,
    configure_trainable_parameters,
    optimizer_parameter_groups,
)


class PairedModel(TinyTextModel):
    def forward(self, pixels, tokens):
        image = self.visual_model.head(self.visual_model.backbone(pixels))
        text = super().forward(tokens)
        return F.normalize(image, dim=-1), F.normalize(text, dim=-1), \
            self.logit_scale.exp(), None, None


@pytest.mark.parametrize("visual", ["adapter", "visionhead"])
@pytest.mark.parametrize("text", ["textproj", "textlora"])
def test_joint_config_updates_both_sides_and_preserves_frozen_parameters(visual, text):
    config = load_config(f"configs/skyscript_sat_{visual}_{text}_3epoch_seed11.toml")
    torch.manual_seed(11)
    model = PairedModel(depth=2)
    model = add_image_embedding_adapter(
        model, bottleneck_dim=config.model.image_adapter_bottleneck, embedding_dim=12
    )
    model = add_text_lora(
        model, rank=config.model.text_lora_rank, alpha=config.model.text_lora_alpha,
        dropout=config.model.text_lora_dropout,
        include_projection=config.model.text_lora_include_projection,
    )
    configure_trainable_parameters(
        model, text_last_k=0, train_vision_head=config.model.train_vision_head,
        train_text_projection=config.model.train_text_projection, train_logit_scale=False,
        train_image_adapter=visual == "adapter", train_text_lora=text == "textlora",
    )
    groups, _, named = optimizer_parameter_groups(model, config.train)
    expected = {"image_adapter" if visual == "adapter" else "vision_head", "text_projection"}
    if text == "textlora":
        expected.add("text_backbone")
    assert {group["name"] for group in groups} == expected
    frozen = {name: p.detach().clone()
              for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(groups)
    pixels, tokens = torch.randn(3, 8), torch.randint(0, 20, (3, 7))
    before = {name: p.detach().clone()
              for name, p in model.named_parameters() if p.requires_grad}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        image, text_features, scale, _, _ = model(pixels, tokens)
        logits = scale * image @ text_features.T
        labels = torch.arange(3)
        ((F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2).backward()
        for group_name in expected:
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for _, p in named[group_name])
            assert sum(float(p.grad.abs().sum()) for _, p in named[group_name]) > 0
        optimizer.step()
    parameters = dict(model.named_parameters())
    assert all(torch.equal(parameters[name], value) for name, value in frozen.items())
    for group_name in expected:
        assert any(not torch.equal(parameters[name], before[name]) for name, _ in named[group_name])
