#!/usr/bin/env python3
"""Convert ChatEarthNet annotations into the project's canonical JSONL contract."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_FIELDS = ("image", "image_path", "file_name", "filename", "path")
CAPTION_FIELDS = ("caption", "captions", "text", "description")
SPLIT_FIELDS = ("split", "partition", "set")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ChatEarthNet JSON annotations to canonical JSONL"
    )
    parser.add_argument("--annotations", required=True, type=Path, help="Official annotation JSON")
    parser.add_argument(
        "--images-root",
        required=True,
        type=Path,
        help="Actual extracted RGB root; nested image directories are searched.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        help="Explicit split written to every record; otherwise the annotation must contain one.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write only resolvable records and report the omitted images instead of failing.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Optional JSON file recording input, sampling, and missing-image statistics.",
    )
    return parser.parse_args()


def caption_text(value: Any) -> str:
    """Select the first official caption and normalize whitespace explicitly."""
    if isinstance(value, list):
        if not value:
            raise ValueError("Empty caption list")
        value = value[0]
    if isinstance(value, dict):
        for field in ("raw", "caption", "text"):
            if field in value:
                return caption_text(value[field])
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid caption: {value!r}")
    return " ".join(value.split())


def records_from_root(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        records = next(
            (value[key] for key in ("annotations", "images", "data", "records") if key in value),
            None,
        )
    else:
        records = None
    if not isinstance(records, list):
        raise TypeError(
            "Expected a list of records or an object with annotations/images/data/records; "
            f"got {type(value).__name__}"
        )
    invalid = [index for index, item in enumerate(records) if not isinstance(item, dict)]
    if invalid:
        raise TypeError(f"Annotation records must be objects; invalid indexes: {invalid[:5]}")
    return records


def required_field(record: dict[str, Any], names: tuple[str, ...], label: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"Record has no {label} field; expected one of {names}, got {sorted(record)}")


def image_reference(record: dict[str, Any]) -> str:
    value = required_field(record, IMAGE_FIELDS, "image")
    if isinstance(value, dict):
        value = required_field(value, ("path", "file_name", "filename", "name"), "image path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid image reference: {value!r}")
    reference = value.replace("\\", "/")
    while reference.startswith("./"):
        reference = reference[2:]
    return reference


def record_split(record: dict[str, Any], explicit_split: str | None) -> str:
    if explicit_split is not None:
        return explicit_split
    value = required_field(record, SPLIT_FIELDS, "split")
    if not isinstance(value, str) or value not in {"train", "val", "test"}:
        raise ValueError(
            f"Invalid split {value!r}; pass --split to map the official split explicitly"
        )
    return value


def build_image_index(images_root: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    root = images_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image root does not exist or is not a directory: {root}")
    paths = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    index: dict[str, list[Path]] = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        for key in (relative, path.name):
            index.setdefault(key, []).append(path)
    return index, paths


def resolve_image(reference: str, images_root: Path, index: dict[str, list[Path]]) -> Path | None:
    raw_path = Path(reference)
    direct = raw_path if raw_path.is_absolute() else images_root / raw_path
    if direct.is_file():
        return direct.resolve()
    candidates = index.get(reference, [])
    if not candidates:
        candidates = index.get(raw_path.name, [])
    if len(candidates) > 1:
        choices = ", ".join(str(path) for path in candidates[:3])
        raise ValueError(f"Ambiguous image reference {reference!r}; candidates include {choices}")
    return candidates[0] if candidates else None


def prepare_manifest(
    *,
    annotations: Path,
    images_root: Path,
    split: str | None,
    limit: int | None,
    seed: int,
    allow_missing: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    records = records_from_root(json.loads(annotations.read_text(encoding="utf-8")))
    index, indexed_paths = build_image_index(images_root)
    converted: list[dict[str, str]] = []
    missing: list[str] = []
    seen_ids: set[str] = set()
    split_counts: Counter[str] = Counter()

    for position, record in enumerate(records):
        reference = image_reference(record)
        resolved = resolve_image(reference, images_root, index)
        sample_id = f"chatearthnet:{reference}"
        if resolved is None:
            missing.append(reference)
            continue
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id at annotation index {position}: {sample_id}")
        seen_ids.add(sample_id)
        sample_split = record_split(record, split)
        split_counts[sample_split] += 1
        converted.append(
            {
                "id": sample_id,
                "image": str(resolved),
                "caption": caption_text(required_field(record, CAPTION_FIELDS, "caption")),
                "split": sample_split,
                "source": "ChatEarthNet",
            }
        )

    if missing and not allow_missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} annotation image references could not be resolved below "
            f"{images_root.resolve()}; first: {preview}"
        )

    random.Random(seed).shuffle(converted)
    selected = converted if limit is None else converted[:limit]
    audit = {
        "annotations": str(annotations.resolve()),
        "images_root": str(images_root.resolve()),
        "annotation_records": len(records),
        "indexed_images": len(indexed_paths),
        "resolved_records": len(converted),
        "selected_records": len(selected),
        "missing_count": len(missing),
        "missing_references": missing,
        "split_counts_before_sampling": dict(sorted(split_counts.items())),
        "seed": seed,
        "limit": limit,
    }
    return selected, audit


def write_json_atomic(path: Path, value: Any) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def write_jsonl_atomic(path: Path, records: list[dict[str, str]]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    records, audit = prepare_manifest(
        annotations=args.annotations,
        images_root=args.images_root,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        allow_missing=args.allow_missing,
    )
    write_jsonl_atomic(args.output, records)
    if args.audit_output is not None:
        write_json_atomic(args.audit_output, audit)
    print(json.dumps({**audit, "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
