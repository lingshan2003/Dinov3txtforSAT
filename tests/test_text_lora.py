import copy
import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from dinotxt_rs.config import load_config, validate_config
from dinotxt_rs.models import add_text_lora, configure_trainable_parameters
from dinotxt_rs.models.official_dinotxt import optimizer_parameter_groups, trainable_state_dict
from dinotxt_rs.models.text_lora import LoRALinear
from dinotxt_rs.training import trainer
from dinotxt_rs.training.checkpoint import _restore_trainable_model


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, 2, c // 2).unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = Attention(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Module()
        self.feed_forward.fc1 = nn.Linear(dim, dim * 4)
        self.feed_forward.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward.fc2(F.gelu(self.feed_forward.fc1(self.ffn_norm(x))))


class TinyTextModel(nn.Module):
    """Small model with the pinned upstream text tower's module names and causal attention."""

    def __init__(self, depth=24):
        super().__init__()
        self.visual_model = nn.Module()
        self.visual_model.backbone = nn.Linear(8, 8)
        self.visual_model.head = nn.Linear(8, 12)
        self.text_model = nn.Module()
        self.text_model.backbone = nn.Module()
        self.text_model.backbone.token_embedding = nn.Embedding(20, 8)
        self.text_model.backbone.positional_embedding = nn.Parameter(torch.randn(7, 8))
        self.text_model.backbone.blocks = nn.ModuleList([Block(8) for _ in range(depth)])
        self.text_model.backbone.ln_final = nn.LayerNorm(8)
        self.text_model.head = nn.Module()
        self.text_model.head.linear_projection = nn.Linear(8, 12, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens):
        backbone = self.text_model.backbone
        x = backbone.token_embedding(tokens) + backbone.positional_embedding[:tokens.shape[1]]
        for block in backbone.blocks:
            x = block(x)
        return self.text_model.head.linear_projection(backbone.ln_final(x))[:, -1]


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    yield
    torch.set_num_threads(previous)


def adapted(model):
    model = add_text_lora(model, rank=2, alpha=4, dropout=0, include_projection=True)
    configure_trainable_parameters(
        model, text_last_k=0, train_vision_head=False, train_text_projection=False,
        train_logit_scale=False, train_text_lora=True,
    )
    return model


def test_zero_delta_matches_original_and_every_block_is_covered():
    model = TinyTextModel().eval()
    reference = copy.deepcopy(model)
    tokens = torch.randint(0, 20, (3, 7))
    model = adapted(model).eval()
    torch.testing.assert_close(model(tokens), reference(tokens), rtol=0, atol=0)
    assert model.text_lora_metadata["blocks"] == 24
    assert len(model.text_lora_metadata["modules"]) == 24 * 4 + 1
    assert model.text_lora_metadata["parameters"] == 24 * 2 * 16 * 8 + 2 * (8 + 12)
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert len(names) == 2 * (24 * 4 + 1)
    assert all(name.endswith(("lora_A", "lora_B")) for name in names)


def test_all_layers_learn_and_original_visual_and_text_weights_stay_fixed():
    model = adapted(TinyTextModel())
    frozen = {
        name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad
    }
    config = load_config("configs/skyscript_sat_textlora_3epoch_seed11.toml")
    groups, metadata, _ = optimizer_parameter_groups(model, config.train)
    assert [group["name"] for group in groups] == ["text_projection", "text_backbone"]
    assert metadata["trainable_parameters"] == model.text_lora_metadata["parameters"]
    optimizer = torch.optim.AdamW(groups)
    tokens = torch.randint(0, 20, (3, 7))
    target = torch.randn(3, 12)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(model(tokens), target).backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None
            else:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                # B=0 gives A a zero first-step gradient; after B's update both factors learn.
                if step or name.endswith("lora_B"):
                    assert torch.count_nonzero(parameter.grad) > 0, name
        optimizer.step()
    parameters = dict(model.named_parameters())
    assert all(torch.equal(parameters[name], old) for name, old in frozen.items())


def test_saved_lora_reconstructs_exact_outputs_and_names(tmp_path):
    base = TinyTextModel(depth=2)
    restored_base = copy.deepcopy(base)
    model = adapted(base)
    tokens = torch.randint(0, 20, (3, 7))
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    F.mse_loss(model(tokens), torch.randn(3, 12)).backward()
    optimizer.step()
    expected = model(tokens).detach()
    path = tmp_path / "lora.pt"
    torch.save(trainable_state_dict(model), path)
    state = torch.load(path, weights_only=True)
    assert all(name.endswith(("lora_A", "lora_B")) for name in state)
    restored = adapted(restored_base)
    _restore_trainable_model(restored, state)
    torch.testing.assert_close(restored(tokens), expected, rtol=0, atol=0)
    incomplete = dict(state)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="parameter names"):
        _restore_trainable_model(restored, incomplete)


def test_architecture_mismatch_is_rejected_before_partial_injection():
    model = TinyTextModel(depth=2)
    model.text_model.backbone.blocks[1].feed_forward.fc2 = nn.Identity()
    with pytest.raises(TypeError, match="feed_forward.fc2"):
        add_text_lora(model, rank=2, alpha=4, dropout=0, include_projection=True)
    assert not any(isinstance(module, LoRALinear) for module in model.modules())


def test_lora_training_resume_and_evaluation_reconstruct_same_outputs(tmp_path, monkeypatch):
    from dinotxt_rs.evaluation import common
    from dinotxt_rs.training.provenance import run_identity

    class PairedModel(TinyTextModel):
        def forward(self, pixels, tokens):
            image = self.visual_model.head(
                self.visual_model.backbone(F.pad(pixels.mean(dim=(-1, -2)), (0, 5)))
            )
            text = super().forward(tokens)
            return F.normalize(image, dim=-1), F.normalize(text, dim=-1), \
                self.logit_scale.exp(), None, None

    class Tokenizer:
        def tokenize(self, captions):
            return torch.tensor([
                [(ord(caption[-1]) + offset) % 20 for offset in range(7)] for caption in captions
            ])

    records = []
    for index in range(4):
        path = tmp_path / f"image{index}.png"
        Image.new("RGB", (8, 8), (20 + index * 30, 40, 60)).save(path)
        records.append({"id": str(index), "image": str(path), "caption": f"caption {index}",
                        "split": "train", "source": "fixture"})
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    config = load_config("configs/skyscript_sat_textlora_3epoch_seed11.toml")
    source = tmp_path / "experiment.toml"
    source.write_text(config.source.read_text())
    config = replace(
        config,
        source=source,
        experiment=replace(config.experiment, output_dir=tmp_path / "full"),
        model=replace(config.model, image_size=16, text_lora_rank=2, text_lora_dropout=0.2),
        data=replace(config.data, train_manifest=manifest, val_manifest=manifest,
                     num_workers=0, validation_num_workers=0,
                     validation_batch_size=2, validation_forward_batch_size=2,
                     validation_prefetch_factor=None),
        train=replace(config.train, device="cpu", precision="fp32", max_steps=5,
                      warmup_steps=0, batch_size=2, gradient_accumulation=1,
                      validation_every=2, checkpoint_every=2, log_every=2),
    )
    hashes = {name: "fixture" for name in (
        "backbone_weights", "dinotxt_weights", "bpe_vocab", "train_manifest", "val_manifest"
    )}
    provenance = {"project_commit": None, "dinov3_commit": "fixture",
                  "files": {name: {"sha256": value} for name, value in hashes.items()}}
    monkeypatch.setattr(trainer, "build_provenance", lambda config: copy.deepcopy(provenance))
    monkeypatch.setattr(trainer, "sha256_file", lambda path: "fixture")
    base = PairedModel(depth=2)

    def construct():
        model = add_text_lora(
            copy.deepcopy(base), rank=2, alpha=16, dropout=0.2, include_projection=True
        )
        configure_trainable_parameters(
            model, text_last_k=0, train_vision_head=False, train_text_projection=False,
            train_logit_scale=False, train_text_lora=True,
        )
        return model

    trainer.seed_everything(123)
    full = construct()
    trainer.train(config, full, Tokenizer())
    resume_config = replace(
        config, experiment=replace(config.experiment, output_dir=tmp_path / "resume")
    )
    trainer.seed_everything(123)
    trainer.train(resume_config, construct(), Tokenizer(), stop_after_step=3)
    restored = construct()
    trainer.train(resume_config, restored, Tokenizer(), resume=tmp_path / "resume/latest.pt")
    for name, parameter in full.named_parameters():
        assert torch.equal(parameter, dict(restored.named_parameters())[name]), name
    assert len(list((tmp_path / "resume").glob("*.pt"))) == 3

    # Exercise the actual evaluation loader, including reinjection and exact parameter-name checks.
    monkeypatch.setattr(common, "_config_input_hashes", lambda config: hashes)
    monkeypatch.setattr(
        common, "load_official_dinotxt", lambda *args: (copy.deepcopy(base), Tokenizer())
    )
    monkeypatch.setattr(common, "sha256_file", lambda path: "fixture")
    monkeypatch.setattr(common, "git_commit", lambda path: None)
    evaluation = common.load_evaluation_model(
        config, checkpoint=tmp_path / "full/latest.pt", training_output=tmp_path / "full"
    )
    full.eval()
    pixels = torch.randn(3, 3, 16, 16)
    tokens = torch.randint(0, 20, (3, 7))
    torch.testing.assert_close(evaluation.model(pixels, tokens)[1], full(pixels, tokens)[1],
                               rtol=0, atol=0)
    payload = torch.load(tmp_path / "full/latest.pt", weights_only=False)
    assert payload["run_identity"] == run_identity(source.read_text(), provenance)


@pytest.mark.parametrize("change", [
    {"text_last_k": 2}, {"train_text_projection": True}, {"text_lora_rank": -1},
    {"text_lora_alpha": float("nan")}, {"text_lora_dropout": 1.0},
    {"text_lora_rank": 0},
])
def test_config_rejects_conflicting_or_invalid_lora(change):
    config = load_config("configs/skyscript_sat_textlora_3epoch_seed11.toml")
    with pytest.raises(ValueError):
        validate_config(replace(config, model=replace(config.model, **change)))
