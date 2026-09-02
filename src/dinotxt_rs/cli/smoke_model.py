from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Load dino.txt and run one synthetic forward pass")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import torch

    from dinotxt_rs.config import load_config, required_paths
    from dinotxt_rs.models import configure_trainable_parameters, load_official_dinotxt

    config = load_config(args.config)
    missing = [str(path) for path in required_paths(config) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
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
    device = torch.device(config.train.device)
    model.to(device).eval()
    pixels = torch.randn(2, 3, config.model.image_size, config.model.image_size, device=device)
    prompts = ["a satellite image of forest", "a satellite image of water"]
    tokens = tokenizer.tokenize(prompts).to(device)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        image_features, text_features, scale, patch_tokens, backbone_patch_tokens = model(
            pixels, tokens
        )
    print(
        {
            "parameters": counts,
            "image_features": tuple(image_features.shape),
            "text_features": tuple(text_features.shape),
            "patch_tokens": tuple(patch_tokens.shape),
            "backbone_patch_tokens": tuple(backbone_patch_tokens.shape),
            "logit_scale": float(scale),
        }
    )


if __name__ == "__main__":
    main()
