from __future__ import annotations

import argparse
from pathlib import Path

from dinotxt_rs.config import load_config
from dinotxt_rs.evaluation.common import (
    load_evaluation_model,
    load_official_reference_model,
    write_json_atomic,
)
from dinotxt_rs.evaluation.retrieval import evaluate_rsicd_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINOtxt-RS retrieval on one RSICD split")
    parser.add_argument("--config", required=True, help="Exact model/training TOML")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument(
        "--official", action="store_true",
        help="Evaluate unmodified official weights without project adapters or LoRA.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--retrieval-chunk-size", type=int, default=256)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args()
    if args.official and (args.checkpoint is not None or args.training_output is not None):
        parser.error("--official cannot be combined with --checkpoint or --training-output")
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output}")
    config = load_config(args.config)
    model = (
        load_official_reference_model(config) if args.official
        else load_evaluation_model(
            config, checkpoint=args.checkpoint, training_output=args.training_output,
        )
    )
    report = evaluate_rsicd_retrieval(
        model,
        args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        retrieval_chunk_size=args.retrieval_chunk_size,
        split=args.split,
    )
    write_json_atomic(args.output, report)
    print(f"evaluation_report={args.output}")


if __name__ == "__main__":
    main()
