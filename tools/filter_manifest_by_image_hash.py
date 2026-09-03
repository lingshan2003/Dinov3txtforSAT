#!/usr/bin/env python3
"""Create a derived manifest excluding known image-content hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a derived JSONL manifest excluding exact image SHA-256 hashes."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source canonical JSONL manifest")
    parser.add_argument("--output", required=True, type=Path, help="New filtered JSONL manifest")
    parser.add_argument(
        "--exclude-sha256",
        required=True,
        action="append",
        dest="excluded_hashes",
        help="64-character SHA-256 of image content to exclude; repeat for each hash.",
    )
    parser.add_argument("--audit-output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_hashes(values: list[str]) -> set[str]:
    hashes = set(values)
    if not hashes:
        raise ValueError("At least one --exclude-sha256 value is required")
    invalid = sorted(
        value
        for value in hashes
        if len(value) != 64 or not all(char in "0123456789abcdef" for char in value)
    )
    if invalid:
        raise ValueError(f"Invalid SHA-256 values: {invalid}")
    return hashes


def filter_manifest(
    input_path: Path, excluded_hashes: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    excluded = Counter[str]()
    input_records = 0
    lines = input_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        input_records += 1
        record = json.loads(line)
        if "id" not in record or "image" not in record:
            raise ValueError(f"{input_path}:{line_number}: canonical record needs id and image")
        image_path = Path(record["image"])
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{input_path}:{line_number}: missing image for {record['id']}: {image_path}"
            )
        digest = sha256_file(image_path)
        if digest in excluded_hashes:
            excluded[digest] += 1
            continue
        records.append(record)
    if not records:
        raise ValueError(f"Filtering removed every record from {input_path}")
    return records, {
        "input_manifest": str(input_path.resolve()),
        "input_manifest_sha256": sha256_file(input_path),
        "input_records": input_records,
        "kept_records": len(records),
        "removed_records": input_records - len(records),
        "excluded_hashes": sorted(excluded_hashes),
        "removed_by_sha256": dict(sorted(excluded.items())),
    }


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


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError(
            "--output must differ from --input so the source manifest remains immutable"
        )
    excluded_hashes = validate_hashes(args.excluded_hashes)
    records, audit = filter_manifest(input_path, excluded_hashes)
    write_jsonl_atomic(output_path, records)
    audit["output_manifest"] = str(output_path)
    audit["output_manifest_sha256"] = sha256_file(output_path)
    write_json_atomic(args.audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
