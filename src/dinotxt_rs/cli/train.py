from __future__ import annotations

import argparse

from dinotxt_rs.config import load_config, required_paths
from dinotxt_rs.models import configure_trainable_parameters, load_official_dinotxt
from dinotxt_rs.training.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DINOv3 dino.txt for remote sensing")
    parser.add_argument("--config", required=True, help="Experiment TOML file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    ratio = counts["trainable"] / counts["total"]
    print(
        f"parameters total={counts['total']:,} "
        f"trainable={counts['trainable']:,} ratio={ratio:.3%}"
    )
    train(config, model, tokenizer)


if __name__ == "__main__":
    main()
