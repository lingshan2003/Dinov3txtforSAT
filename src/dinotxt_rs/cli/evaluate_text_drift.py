from __future__ import annotations

import argparse
from pathlib import Path

from dinotxt_rs.config import load_config
from dinotxt_rs.evaluation.common import write_json_atomic
from dinotxt_rs.evaluation.text_drift import evaluate_text_embedding_drift


def _named_path(value: str, *, label: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError(f"{label} must use NAME=PATH")
    return name, Path(raw_path)


def _checkpoint(value: str) -> tuple[int, Path]:
    raw_step, separator, raw_path = value.partition("=")
    if not separator or not raw_step or not raw_path:
        raise argparse.ArgumentTypeError("--checkpoint must use STEP=PATH")
    try:
        step = int(raw_step)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Checkpoint STEP must be an integer") from exc
    if step < 0:
        raise argparse.ArgumentTypeError("Checkpoint STEP must be nonnegative")
    return step, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure checkpoint text-embedding drift from frozen official dino.txt"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--training-output", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        required=True,
        action="append",
        help="Named caption manifest as NAME=PATH; repeat for multiple datasets",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        action="append",
        help="Authenticated checkpoint as STEP=PATH; repeat for multiple stages",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifests = [_named_path(value, label="--manifest") for value in args.manifest]
    checkpoints = [_checkpoint(value) for value in args.checkpoint]
    if len({name for name, _ in manifests}) != len(manifests):
        parser.error("--manifest names must be unique")
    if len({step for step, _ in checkpoints}) != len(checkpoints):
        parser.error("--checkpoint steps must be unique")
    args.manifest = dict(manifests)
    args.checkpoint = dict(checkpoints)
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite text drift report: {args.output}")
    report = evaluate_text_embedding_drift(
        load_config(args.config),
        manifests=args.manifest,
        checkpoints=args.checkpoint,
        training_output=args.training_output,
        batch_size=args.batch_size,
    )
    write_json_atomic(args.output, report)
    print(f"text_drift_report={args.output}")


if __name__ == "__main__":
    main()
