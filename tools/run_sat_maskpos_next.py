#!/usr/bin/env python3
"""Run separate full-maskpos tests of candidate batch, image epochs and trainable scope."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinotxt_rs.config import load_config  # noqa: E402
from tools import run_sat_full_image_epoch as shared  # noqa: E402
from tools.run_sat_multi_positive import Trial  # noqa: E402

REPORT_DIR = Path("outputs/sat_maskpos_next_seed11")
VARIANTS = {
    "batch32_1epoch": (
        "skyscript_sat_adapter_textlora_maskpos_batch32_fullimage1epoch_seed11", 32, 2, 1, False, 0,
    ),
    "epochs3_batch16": (
        "skyscript_sat_adapter_textlora_maskpos_fullimage3epoch_seed11", 16, 4, 3, False, 0,
    ),
    "head_lora_1epoch": (
        "skyscript_sat_head_adapter_textlora_maskpos_batch32_fullimage1epoch_seed11",
        32, 2, 1, True, 0,
    ),
    "fulltext4_1epoch": (
        "skyscript_sat_head_adapter_textlast4_maskpos_batch32_fullimage1epoch_seed11",
        32, 2, 1, True, 4,
    ),
}


def run_pipeline(
    root: Path, python: str, *, mode: str = "all", include_strong: bool = False,
) -> Path | None:
    root = root.resolve()
    trials = []
    reference = load_config(
        root / "configs/skyscript_sat_adapter_textlora_maskpos_fullimage1epoch_seed11.toml"
    )
    for method, (name, batch, accumulation, epochs, head, fulltext) in VARIANTS.items():
        if fulltext and not include_strong:
            continue
        trial = Trial(method, root / "configs" / f"{name}.toml", root / "outputs" / name)
        config = load_config(trial.config)
        model = config.model
        if (
            config.experiment.seed != 11
            or config.experiment.name != name
            or (root / config.experiment.output_dir).resolve() != trial.output
            or config.data != reference.data
            or model.backbone_domain != "sat"
            or any(getattr(model, field) != getattr(reference.model, field) for field in (
                "dinov3_repo", "backbone_weights", "dinotxt_weights", "bpe_vocab", "image_size",
            ))
            or model.image_adapter_bottleneck != 256
            or model.train_vision_head != head
            or model.text_last_k != fulltext
            or model.train_text_projection != bool(fulltext)
            or model.text_lora_rank != (0 if fulltext else 8)
            or model.text_lora_include_projection != (not fulltext)
            or model.train_logit_scale
            or config.train.batch_size != batch
            or config.train.gradient_accumulation != accumulation
            or config.train.image_epochs != epochs
            or config.train.queue_size != 0
            or config.train.contrastive_objective != "mask_same_caption"
        ):
            raise ValueError(f"Unexpected full-maskpos follow-up protocol: {trial.config}")
        expected_rates = {
            "image_adapter_learning_rate": 1e-4,
            "text_projection_learning_rate": 1e-5 if fulltext else 1e-4,
            "text_backbone_learning_rate": 5e-6 if fulltext else 1e-4,
        }
        if head:
            expected_rates["vision_head_learning_rate"] = 1e-5
        if any(getattr(config.train, key) != value for key, value in expected_rates.items()):
            raise ValueError(f"Unexpected per-module learning rates: {trial.config}")
        trials.append(trial)
    return shared.run_pipeline(
        root, python, mode=mode, trials=trials, report_dir=REPORT_DIR,
        series="sat_maskpos_next_seed11", require_matching_recipes=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-strong", action="store_true",
        help="Also fully update the final four text blocks and projection (no text LoRA).",
    )
    modes = parser.add_mutually_exclusive_group()
    for name in ("train-only", "evaluate-only", "preflight-only"):
        modes.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    mode = next((name for name in ("train-only", "evaluate-only", "preflight-only")
                 if getattr(args, name.replace("-", "_"))), "all")
    run_pipeline(Path(__file__).resolve().parents[1], sys.executable, mode=mode,
                 include_strong=args.include_strong)


if __name__ == "__main__":
    main()
