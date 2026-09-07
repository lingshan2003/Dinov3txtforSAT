#!/usr/bin/env python3
"""Audit exact-file and exact-decoded-pixel overlap between canonical manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        value = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{value.width}x{value.height}:".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _unique_images(manifest: Path) -> list[dict[str, str]]:
    images: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{manifest}:{line_number}: expected a JSON object")
        image = record.get("image")
        identifier = record.get("image_id", record.get("id"))
        if not isinstance(image, str) or not image or not isinstance(identifier, str):
            raise ValueError(f"{manifest}:{line_number}: record needs image and id/image_id")
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"{manifest}:{line_number}: missing image: {path}")
        resolved = str(path.resolve())
        existing = images.get(identifier)
        current = {"id": identifier, "path": resolved}
        if existing is not None and existing != current:
            raise ValueError(f"{manifest}:{line_number}: inconsistent image for {identifier}")
        images[identifier] = current
    if not images:
        raise ValueError(f"Manifest contains no images: {manifest}")
    return [images[key] for key in sorted(images)]


def _fingerprints(records: list[dict[str, str]]) -> list[dict[str, str]]:
    by_path: dict[str, tuple[str, str]] = {}
    result: list[dict[str, str]] = []
    for record in records:
        path = record["path"]
        if path not in by_path:
            by_path[path] = (sha256_file(Path(path)), decoded_pixel_sha256(Path(path)))
        file_hash, pixel_hash = by_path[path]
        result.append({**record, "file_sha256": file_hash, "pixel_sha256": pixel_hash})
    return result


def _overlaps(
    left: list[dict[str, str]], right: list[dict[str, str]], key: str
) -> list[dict[str, str]]:
    left_by_hash: dict[str, list[dict[str, str]]] = {}
    for record in left:
        left_by_hash.setdefault(record[key], []).append(record)
    result: list[dict[str, str]] = []
    for right_record in right:
        for left_record in left_by_hash.get(right_record[key], []):
            result.append(
                {
                    "sha256": right_record[key],
                    "left_id": left_record["id"],
                    "left_path": left_record["path"],
                    "right_id": right_record["id"],
                    "right_path": right_record["path"],
                }
            )
    return result


def audit_overlap(left_manifest: Path, right_manifest: Path) -> dict[str, Any]:
    left = _fingerprints(_unique_images(left_manifest))
    right = _fingerprints(_unique_images(right_manifest))
    file_overlaps = _overlaps(left, right, "file_sha256")
    pixel_overlaps = _overlaps(left, right, "pixel_sha256")
    return {
        "format_version": 1,
        "left_manifest": {
            "path": str(left_manifest.resolve()),
            "sha256": sha256_file(left_manifest),
            "unique_images": len(left),
        },
        "right_manifest": {
            "path": str(right_manifest.resolve()),
            "sha256": sha256_file(right_manifest),
            "unique_images": len(right),
        },
        "exact_file_overlap_count": len(file_overlaps),
        "exact_decoded_pixel_overlap_count": len(pixel_overlaps),
        "exact_file_overlaps": file_overlaps,
        "exact_decoded_pixel_overlaps": pixel_overlaps,
        "status": "clear" if not file_overlaps and not pixel_overlaps else "overlap_detected",
        "scope_note": (
            "This detects byte-identical files and images with identical decoded RGB dimensions "
            "and pixels; it does not claim perceptual near-duplicate detection."
        ),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-manifest", required=True, type=Path)
    parser.add_argument("--right-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-zero-overlap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite overlap audit: {args.output}")
    report = audit_overlap(args.left_manifest, args.right_manifest)
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_zero_overlap and report["status"] != "clear":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
