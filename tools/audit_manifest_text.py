#!/usr/bin/env python3
"""Audit canonical manifest captions with the exact local dino.txt tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dinov3-repo", required=True, type=Path)
    parser.add_argument("--bpe-vocab", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context-length", type=int, default=77)
    parser.add_argument(
        "--require-fit",
        action="store_true",
        help="Exit nonzero after writing the audit when any caption exceeds context length",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def audit_manifest(manifest: Path, tokenizer: Any, context_length: int) -> dict[str, Any]:
    if context_length < 3:
        raise ValueError("context_length must be at least 3")
    source = manifest.resolve()
    counts: Counter[str] = Counter()
    token_lengths: list[int] = []
    overflow_examples: list[dict[str, Any]] = []
    records = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{source}:{line_number}: blank lines are not allowed")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            sample_id = value.get("id")
            caption = value.get("caption")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{source}:{line_number}: missing non-empty id")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"{source}:{line_number}: missing non-empty caption")
            caption = " ".join(caption.split())
            length = len(tokenizer.encode(caption)) + 2
            records += 1
            counts[caption.casefold()] += 1
            token_lengths.append(length)
            if length > context_length and len(overflow_examples) < 20:
                overflow_examples.append(
                    {"id": sample_id, "tokens": length, "caption": caption}
                )
    if not records:
        raise ValueError(f"Manifest is empty: {source}")
    overflow_count = sum(length > context_length for length in token_lengths)
    duplicate_records = sum(count - 1 for count in counts.values())
    return {
        "format_version": 1,
        "manifest": str(source),
        "manifest_sha256": sha256_file(source),
        "records": records,
        "context_length": context_length,
        "token_lengths_including_boundary_tokens": {
            "p50": _percentile(token_lengths, 0.50),
            "p95": _percentile(token_lengths, 0.95),
            "max": max(token_lengths),
        },
        "overflow_count": overflow_count,
        "overflow_fraction": overflow_count / records,
        "overflow_examples": overflow_examples,
        "unique_normalized_captions": len(counts),
        "duplicate_records": duplicate_records,
        "duplicate_fraction": duplicate_records / records,
        "top_normalized_captions": counts.most_common(20),
        "status": "clear" if overflow_count == 0 else "context_overflow",
    }


def load_tokenizer(dinov3_repo: Path, bpe_vocab: Path) -> Any:
    repo = dinov3_repo.resolve()
    vocab = bpe_vocab.resolve()
    if not repo.is_dir():
        raise NotADirectoryError(repo)
    if not vocab.is_file():
        raise FileNotFoundError(vocab)
    sys.path.insert(0, str(repo))
    from dinov3.eval.text.tokenizer import get_tokenizer

    return get_tokenizer(str(vocab))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.dinov3_repo, args.bpe_vocab)
    report = audit_manifest(args.manifest, tokenizer, args.context_length)
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    if args.require_fit and report["status"] != "clear":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
