from __future__ import annotations

import argparse
from pathlib import Path

from dinotxt_rs.config import load_config
from dinotxt_rs.evaluation.common import load_evaluation_model, write_json_atomic
from dinotxt_rs.evaluation.eurosat import evaluate_eurosat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DINOtxt-RS on EuroSAT zero-shot classification"
    )
    parser.add_argument("--config", required=True, help="Exact model/training TOML")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output}")
    model = load_evaluation_model(
        load_config(args.config),
        checkpoint=args.checkpoint,
        training_output=args.training_output,
    )
    report = evaluate_eurosat(
        model,
        args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    write_json_atomic(args.output, report)
    print(f"evaluation_report={args.output}")


if __name__ == "__main__":
    main()
