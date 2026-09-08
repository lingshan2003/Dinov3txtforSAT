import torch
import torch.nn.functional as F

from dinotxt_rs.models import add_image_embedding_adapter, configure_trainable_parameters


class TinyOfficialModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual_model = torch.nn.Module()
        self.visual_model.backbone = torch.nn.Linear(3, 4, bias=False)
        self.visual_model.head = torch.nn.Linear(4, 8, bias=False)
        self.text_model = torch.nn.Module()
        self.text_model.backbone = torch.nn.Module()
        self.text_model.backbone.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2, bias=False)]
        )
        self.text_model.backbone.ln_final = torch.nn.LayerNorm(2)
        self.text_model.head = torch.nn.Linear(2, 8, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, pixels: torch.Tensor, tokens: torch.Tensor):
        image = F.normalize(self.visual_model.head(self.visual_model.backbone(pixels)), dim=-1)
        text = F.normalize(self.text_model.head(tokens), dim=-1)
        return image, text, self.logit_scale.exp(), "patch", "backbone-patch"


def test_zero_initialized_adapter_preserves_initial_embeddings() -> None:
    base = TinyOfficialModel()
    pixels = torch.randn(3, 3)
    tokens = torch.randn(3, 2)
    expected = base(pixels, tokens)
    model = add_image_embedding_adapter(base, bottleneck_dim=2, embedding_dim=8)

    actual = model(pixels, tokens)

    torch.testing.assert_close(actual[0], expected[0], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    assert actual[2:] == expected[2:]


def test_adapter_only_policy_freezes_the_official_model() -> None:
    model = add_image_embedding_adapter(TinyOfficialModel(), bottleneck_dim=2, embedding_dim=8)

    counts = configure_trainable_parameters(
        model,
        text_last_k=0,
        train_vision_head=False,
        train_text_projection=False,
        train_logit_scale=False,
        train_image_adapter=True,
    )

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {
        "image_adapter.norm.weight",
        "image_adapter.norm.bias",
        "image_adapter.down.weight",
        "image_adapter.down.bias",
        "image_adapter.up.weight",
        "image_adapter.up.bias",
    }
    assert counts["trainable"] == 58
    assert counts["trainable"] < counts["total"]
