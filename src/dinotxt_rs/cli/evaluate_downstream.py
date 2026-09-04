from __future__ import annotations

import argparse
from pathlib import Path

from dinotxt_rs.config import load_config
from dinotxt_rs.evaluation.common import load_evaluation_model, write_json_atomic
from dinotxt_rs.evaluation.eurosat import evaluate_eurosat
from dinotxt_rs.evaluation.retrieval import evaluate_rsicd_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one baseline or checkpoint on EuroSAT and RSICD"
    )
    parser.add_argument("--config", required=True, help="Exact model/training TOML")
    parser.add_argument("--eurosat-manifest", required=True, type=Path)
    parser.add_argument("--rsicd-manifest", required=True, type=Path)
    parser.add_argument("--eurosat-output", required=True, type=Path)
    parser.add_argument("--rsicd-output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--retrieval-chunk-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eurosat_output.exists() or args.rsicd_output.exists():
        raise FileExistsError("Refusing to overwrite an existing downstream evaluation report")
    model = load_evaluation_model(
        load_config(args.config),
        checkpoint=args.checkpoint,
        training_output=args.training_output,
    )
    eurosat = evaluate_eurosat(
        model,
        args.eurosat_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    write_json_atomic(args.eurosat_output, eurosat)
    rsicd = evaluate_rsicd_retrieval(
        model,
        args.rsicd_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        retrieval_chunk_size=args.retrieval_chunk_size,
    )
    write_json_atomic(args.rsicd_output, rsicd)
    print(f"eurosat_report={args.eurosat_output}")
    print(f"rsicd_report={args.rsicd_output}")


if __name__ == "__main__":
    main()
