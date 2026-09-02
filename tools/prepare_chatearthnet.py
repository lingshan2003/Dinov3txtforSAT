#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ChatEarthNet JSON files to canonical JSONL"
    )
    parser.add_argument("--annotations", required=True, type=Path, help="Official split JSON file")
    parser.add_argument(
        "--images-root", required=True, type=Path, help="Directory containing RGB PNG files"
    )
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def caption_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            raise ValueError("Empty caption list")
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid caption: {value!r}")
    return " ".join(value.split())


def main() -> None:
    args = parse_args()
    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("Expected the official ChatEarthNet annotation root to be a list")
    random.Random(args.seed).shuffle(records)
    if args.limit is not None:
        records = records[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for item in records:
            filename = item["image"]
            image_path = (args.images_root / filename).resolve()
            if not image_path.is_file():
                missing += 1
                if args.allow_missing:
                    continue
                raise FileNotFoundError(f"Missing image: {image_path}")
            output = {
                "id": f"chatearthnet:{filename}",
                "image": str(image_path),
                "caption": caption_text(item["caption"]),
                "split": args.split,
                "source": "ChatEarthNet",
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"written": written, "missing": missing, "output": str(args.output)}))


if __name__ == "__main__":
    main()
