#!/usr/bin/env python3
"""Expand the official RSICD test captions into a deterministic retrieval manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpeg", ".jpg"}
FILENAME_FIELDS = ("filename", "file_name", "image", "image_path", "filepath", "path")
IDENTIFIER_FIELDS = ("imgid", "image_id", "id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an expanded RSICD test retrieval manifest from official annotations"
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _annotation_images(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("images")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("RSICD annotation must be an object with an images list")
    return value


def _image_index(images_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in images_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        for key in (path.name, path.relative_to(images_root).as_posix()):
            index.setdefault(key, []).append(path.resolve())
    if not index:
        raise ValueError(f"No RSICD JPEG images found under {images_root}")
    return index


def _first_string(record: dict[str, Any], fields: tuple[str, ...], label: str) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"RSICD image record has no nonempty {label}: {record!r}")


def _first_identifier(record: dict[str, Any]) -> str:
    for field in IDENTIFIER_FIELDS:
        value = record.get(field)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    raise ValueError(f"RSICD image record has no usable image identifier: {record!r}")


def _resolve_image(reference: str, images_root: Path, index: dict[str, list[Path]]) -> Path:
    normalized = reference.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    candidate = Path(normalized)
    direct = candidate if candidate.is_absolute() else images_root / candidate
    if direct.is_file():
        return direct.resolve()
    matches = index.get(normalized, []) or index.get(candidate.name, [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"RSICD annotation image cannot be resolved: {reference}")
    raise ValueError(f"RSICD annotation image is ambiguous: {reference} -> {matches}")


def _caption_text(sentence: Any) -> str:
    if not isinstance(sentence, dict):
        raise ValueError(f"RSICD sentence is not an object: {sentence!r}")
    for field in ("raw", "caption", "text"):
        value = sentence.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    tokens = sentence.get("tokens")
    if isinstance(tokens, list) and all(isinstance(token, str) for token in tokens):
        text = " ".join(token.strip() for token in tokens if token.strip())
        if text:
            return text
    raise ValueError(f"RSICD sentence has no usable caption text: {sentence!r}")


def build_records(
    annotations: Path, images_root: Path, split: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    annotations = annotations.resolve()
    images_root = images_root.resolve()
    if not annotations.is_file():
        raise FileNotFoundError(f"RSICD annotation does not exist: {annotations}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"RSICD image root does not exist: {images_root}")
    annotation_images = _annotation_images(json.loads(annotations.read_text(encoding="utf-8")))
    image_index = _image_index(images_root)
    records: list[dict[str, str]] = []
    image_ids: set[str] = set()
    selected_images = 0
    for image_record in annotation_images:
        record_split = image_record.get("split")
        if not isinstance(record_split, str) or record_split.lower() != split.lower():
            continue
        native_id = _first_identifier(image_record)
        image_id = f"rsicd:{native_id}"
        if image_id in image_ids:
            raise ValueError(f"Duplicate RSICD image identifier: {image_id}")
        image_ids.add(image_id)
        image_path = _resolve_image(
            _first_string(image_record, FILENAME_FIELDS, "filename"), images_root, image_index
        )
        sentences = image_record.get("sentences")
        if not isinstance(sentences, list) or not sentences:
            raise ValueError(f"RSICD image {image_id} has no sentences")
        selected_images += 1
        for caption_index, sentence in enumerate(sentences):
            records.append(
                {
                    "id": f"{image_id}:caption{caption_index:02d}",
                    "image_id": image_id,
                    "image": str(image_path),
                    "caption": _caption_text(sentence),
                    "split": split.lower(),
                    "source": "RSICD",
                }
            )
    if not records:
        raise ValueError(f"RSICD has no records with split={split!r}")
    records.sort(key=lambda record: record["id"])
    audit = {
        "format_version": 1,
        "dataset": "RSICD",
        "split": split.lower(),
        "annotations": {"path": str(annotations), "sha256": sha256_file(annotations)},
        "images_root": str(images_root),
        "images": selected_images,
        "captions": len(records),
        "caption_count_histogram": dict(
            sorted(Counter(Counter(record["image_id"] for record in records).values()).items())
        ),
    }
    return records, audit


def write_manifest(
    records: list[dict[str, str]], audit: dict[str, Any], output: Path, audit_output: Path
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    audit["manifest"] = {"path": str(output.resolve()), "sha256": sha256_file(output)}
    temporary_audit = audit_output.with_suffix(audit_output.suffix + ".part")
    temporary_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_audit, audit_output)
    return audit


def main() -> None:
    args = parse_args()
    records, audit = build_records(args.annotations, args.images_root, args.split)
    print(
        json.dumps(
            write_manifest(records, audit, args.output, args.audit_output), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
