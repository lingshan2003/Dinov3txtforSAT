#!/usr/bin/env python3
"""Derive a small, immutable prefix manifest for bounded training verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("id", "image", "caption", "split", "source")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed prefix of a canonical training JSONL manifest"
    )
    parser.add_argument("--input", required=True, type=Path, help="Canonical source JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Derived JSONL path")
    parser.add_argument(
        "--limit", required=True, type=int, help="Exact number of records to retain"
    )
    parser.add_argument("--audit-output", required=True, type=Path, help="JSON audit path")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_canonical_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing required fields: {missing}")
            if not isinstance(record["id"], str) or not record["id"]:
                raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
            if record["id"] in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id: {record['id']}")
            seen_ids.add(record["id"])
            records.append(record)
    if not records:
        raise ValueError(f"Source manifest is empty: {path}")
    return records


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, destination)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def prepare_fixed_manifest(
    *, source: Path, output: Path, audit_output: Path, limit: int
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    resolved_source = source.resolve()
    records = read_canonical_records(resolved_source)
    if len(records) < limit:
        raise ValueError(
            f"Source manifest has {len(records)} records, fewer than requested limit={limit}"
        )
    selected = records[:limit]
    if {record["split"] for record in selected} != {"train"}:
        raise ValueError("Fixed verification manifests must contain only split='train' records")
    _write_jsonl_atomic(output, selected)
    resolved_output = output.resolve()
    audit = {
        "source_manifest": str(resolved_source),
        "source_sha256": sha256_file(resolved_source),
        "output_manifest": str(resolved_output),
        "output_sha256": sha256_file(resolved_output),
        "source_records": len(records),
        "selected_records": len(selected),
        "selection": "first_n_in_existing_manifest_order",
        "sample_ids": [record["id"] for record in selected],
    }
    _write_json_atomic(audit_output, audit)
    return audit


def main() -> None:
    args = parse_args()
    audit = prepare_fixed_manifest(
        source=args.input,
        output=args.output,
        audit_output=args.audit_output,
        limit=args.limit,
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
