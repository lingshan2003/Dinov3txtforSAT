#!/usr/bin/env python3
"""Create a deterministic all-image EuroSAT zero-shot evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

EUROSAT_CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
IMAGE_SUFFIXES = {".jpeg", ".jpg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sorted EuroSAT manifest without modifying the raw dataset"
    )
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_records(images_root: Path, *, expected_total: int = 27_000) -> list[dict[str, str]]:
    images_root = images_root.resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"EuroSAT images root does not exist: {images_root}")
    paths = sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No JPEG images found under {images_root}")
    records: list[dict[str, str]] = []
    for path in paths:
        label = path.parent.name
        if label not in EUROSAT_CLASSES:
            raise ValueError(f"Unexpected EuroSAT class directory for {path}: {label!r}")
        relative = path.relative_to(images_root).as_posix()
        records.append(
            {
                "id": f"eurosat:{relative}",
                "image": str(path.resolve()),
                "label": label,
                "split": "test",
                "source": "EuroSAT",
            }
        )
    counts = Counter(record["label"] for record in records)
    missing = sorted(set(EUROSAT_CLASSES) - set(counts))
    if missing:
        raise ValueError(f"EuroSAT is missing expected class directories: {missing}")
    if len(records) != expected_total:
        raise ValueError(f"Expected {expected_total:,} EuroSAT images, found {len(records)}")
    return records


def write_manifest(
    records: list[dict[str, str]], output: Path, audit_output: Path, images_root: Path
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    audit = {
        "format_version": 1,
        "dataset": "EuroSAT",
        "images_root": str(images_root.resolve()),
        "records": len(records),
        "classes": list(EUROSAT_CLASSES),
        "class_counts": dict(sorted(Counter(record["label"] for record in records).items())),
        "manifest": {"path": str(output.resolve()), "sha256": sha256_file(output)},
    }
    temporary_audit = audit_output.with_suffix(audit_output.suffix + ".part")
    temporary_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_audit, audit_output)
    return audit


def main() -> None:
    args = parse_args()
    records = build_records(args.images_root)
    audit = write_manifest(records, args.output, args.audit_output, args.images_root)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
