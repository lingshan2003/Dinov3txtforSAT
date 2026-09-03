#!/usr/bin/env python3
"""Derive a canonical manifest with sentence-aware captions fitting dino.txt's context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace canonical captions with first complete sentences fitted to dino.txt's context."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Source canonical JSONL manifest")
    parser.add_argument("--output", required=True, type=Path, help="New global77 JSONL manifest")
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--dinov3-repo", required=True, type=Path)
    parser.add_argument("--bpe-vocab", required=True, type=Path)
    parser.add_argument("--context-length", type=int, default=77)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_complete_sentence(caption: str) -> str:
    """Return text through the first ., !, or ? followed by whitespace/end, never a comma clause."""
    sentence = SENTENCE_BOUNDARY.split(caption.strip(), maxsplit=1)[0].strip()
    if not sentence:
        raise ValueError("Caption is empty after stripping whitespace")
    return sentence


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text)) + 2  # start-of-text and end-of-text tokens


def fit_complete_words(tokenizer: Any, text: str, context_length: int) -> tuple[str, bool]:
    """Fit text without splitting a word; return whether word-backoff was necessary."""
    if context_length < 3:
        raise ValueError("context_length must allow start token, one text token, and end token")
    fitted = text.strip()
    needed_backoff = False
    while token_count(tokenizer, fitted) > context_length:
        words = fitted.rsplit(maxsplit=1)
        if len(words) != 2:
            raise ValueError(
                "A single whitespace-delimited word exceeds the configured tokenizer context length"
            )
        fitted = words[0].rstrip(" ,;:")
        needed_backoff = True
    if not fitted:
        raise ValueError("Word backoff removed the complete caption")
    return fitted, needed_backoff


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def derive_records(
    input_path: Path, tokenizer: Any, context_length: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    original_token_lengths: list[int] = []
    global_token_lengths: list[int] = []
    word_backoff: list[dict[str, Any]] = []

    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" not in record or "caption" not in record:
                raise ValueError(
                    f"{input_path}:{line_number}: canonical record needs id and caption"
                )
            caption = record["caption"]
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"{input_path}:{line_number}: invalid caption for {record['id']}")

            sentence = first_complete_sentence(caption)
            global_caption, used_backoff = fit_complete_words(tokenizer, sentence, context_length)
            original_token_lengths.append(token_count(tokenizer, caption))
            global_token_lengths.append(token_count(tokenizer, global_caption))
            if used_backoff:
                word_backoff.append(
                    {
                        "id": record["id"],
                        "first_sentence_tokens": token_count(tokenizer, sentence),
                        "global77_tokens": token_count(tokenizer, global_caption),
                        "global77_caption": global_caption,
                    }
                )
            records.append({**record, "caption": global_caption})

    if not records:
        raise ValueError(f"Input manifest is empty: {input_path}")
    audit = {
        "strategy": "first_complete_sentence_then_complete_word_backoff",
        "context_length": context_length,
        "input_manifest": str(input_path.resolve()),
        "input_manifest_sha256": sha256_file(input_path),
        "records": len(records),
        "original_caption_tokens": {
            "p50": percentile(original_token_lengths, 0.50),
            "p95": percentile(original_token_lengths, 0.95),
            "max": max(original_token_lengths),
        },
        "global77_caption_tokens": {
            "p50": percentile(global_token_lengths, 0.50),
            "p95": percentile(global_token_lengths, 0.95),
            "max": max(global_token_lengths),
        },
        "word_backoff_records": len(word_backoff),
        "word_backoff_examples": word_backoff[:20],
    }
    return records, audit


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def load_tokenizer(dinov3_repo: Path, bpe_vocab: Path) -> Any:
    repo = dinov3_repo.resolve()
    vocab = bpe_vocab.resolve()
    if not repo.is_dir():
        raise NotADirectoryError(f"DINOv3 repository does not exist: {repo}")
    if not vocab.is_file():
        raise FileNotFoundError(f"BPE vocabulary does not exist: {vocab}")
    sys.path.insert(0, str(repo))
    from dinov3.eval.text.tokenizer import get_tokenizer

    return get_tokenizer(str(vocab))


def main() -> None:
    args = parse_args()
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive")
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("--output must differ from --input so the raw manifest remains immutable")
    tokenizer = load_tokenizer(args.dinov3_repo, args.bpe_vocab)
    records, audit = derive_records(input_path, tokenizer, args.context_length)
    write_jsonl_atomic(output_path, records)
    audit["output_manifest"] = str(output_path)
    audit["output_manifest_sha256"] = sha256_file(output_path)
    write_json_atomic(args.audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
