from __future__ import annotations

import argparse
from pathlib import Path

from dinotxt_rs.config import load_config
from dinotxt_rs.evaluation.common import write_json_atomic
from dinotxt_rs.evaluation.parity import run_step0_parity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official initialization with an authenticated step-0 checkpoint"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-output", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite parity report: {args.output}")
    report = run_step0_parity(
        load_config(args.config),
        checkpoint=args.checkpoint,
        training_output=args.training_output,
        input_manifest=args.input_manifest,
        batch_size=args.batch_size,
        atol=args.atol,
        rtol=args.rtol,
    )
    write_json_atomic(args.output, report)
    print(f"parity_report={args.output} status={report['status']}", flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
