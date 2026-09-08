#!/usr/bin/env python3
"""Split a unique-caption SkyScript selection CSV into deterministic train/val CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--group-field", default="title_raw")
    parser.add_argument("--path-field", default="filepath")
    parser.add_argument("--val-count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(value.split()).strip().casefold()


def priority(seed: int, group: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{group}".encode()).digest()
    return int.from_bytes(digest[:16], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_unique_rows(
    path: Path, *, group_field: str, path_field: str
) -> tuple[list[dict[str, str]], list[str]]:
    source = path.resolve()
    rows: list[dict[str, str]] = []
    seen_groups: set[str] = set()
    seen_paths: set[str] = set()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted({group_field, path_field} - set(fieldnames))
        if missing:
            raise ValueError(f"Selection CSV is missing columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{source}:{line_number}: row has more values than header")
            group = normalize_text(row.get(group_field, ""))
            filepath = (row.get(path_field) or "").strip().replace("\\", "/")
            if not group or not filepath:
                raise ValueError(f"{source}:{line_number}: empty group or filepath")
            if group in seen_groups:
                raise ValueError(
                    f"{source}:{line_number}: duplicate normalized {group_field}: {group!r}"
                )
            if filepath in seen_paths:
                raise ValueError(f"{source}:{line_number}: duplicate filepath: {filepath}")
            seen_groups.add(group)
            seen_paths.add(filepath)
            rows.append({name: row.get(name, "") for name in fieldnames})
    if not rows:
        raise ValueError(f"Selection CSV is empty: {source}")
    return rows, fieldnames


def split_rows(
    rows: list[dict[str, str]], *, group_field: str, val_count: int, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < val_count < len(rows):
        raise ValueError("val_count must be positive and smaller than the number of rows")
    ranked = sorted(
        rows,
        key=lambda row: (
            priority(seed, normalize_text(row[group_field])),
            normalize_text(row[group_field]),
        ),
    )
    val_groups = {normalize_text(row[group_field]) for row in ranked[:val_count]}
    train = [row for row in rows if normalize_text(row[group_field]) not in val_groups]
    val = [row for row in rows if normalize_text(row[group_field]) in val_groups]
    if len(val) != val_count or len(train) + len(val) != len(rows):
        raise RuntimeError("Deterministic SkyScript split produced inconsistent counts")
    return train, val


def write_csv_atomic(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite split CSV: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite split audit: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    destinations = {
        args.train_output.resolve(),
        args.val_output.resolve(),
        args.audit_output.resolve(),
    }
    if len(destinations) != 3 or args.input.resolve() in destinations:
        raise ValueError("Input, train, validation, and audit paths must all differ")
    existing = sorted(path for path in destinations if path.exists())
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing split output(s): "
            + ", ".join(str(path) for path in existing)
        )
    rows, fieldnames = read_unique_rows(
        args.input, group_field=args.group_field, path_field=args.path_field
    )
    train, val = split_rows(
        rows, group_field=args.group_field, val_count=args.val_count, seed=args.seed
    )
    write_csv_atomic(args.train_output, train, fieldnames)
    write_csv_atomic(args.val_output, val, fieldnames)
    report = {
        "format_version": 1,
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input.resolve()),
        "group_field": args.group_field,
        "seed": args.seed,
        "records": len(rows),
        "train_records": len(train),
        "val_records": len(val),
        "train_output": str(args.train_output.resolve()),
        "train_sha256": sha256_file(args.train_output.resolve()),
        "val_output": str(args.val_output.resolve()),
        "val_sha256": sha256_file(args.val_output.resolve()),
        "group_overlap": 0,
        "selection": "lowest_sha256_group_priority_to_validation",
    }
    write_json_atomic(args.audit_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
