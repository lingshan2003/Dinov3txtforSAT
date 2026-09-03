#!/usr/bin/env python3
"""Create a machine-readable, read-only inventory of the server-side assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_REFERENCE_FIELDS = {"file_name", "filename", "filepath", "image", "image_path", "path"}
CHECKPOINTS = (
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    "bpe_simple_vocab_16e6.txt.gz",
)
ARCHIVE_SUFFIXES = {".7z", ".gz", ".rar", ".tar", ".tgz", ".xz", ".zip"}
DIRECTORY_HINTS = {
    "chatearthnet": ("chatearth", "json_files", "s2_rgb"),
    "eurosat": ("eurosat", "2750"),
    "rsicd": ("rsicd",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory DINOtxt-RS checkpoints and extracted datasets without mutating them."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project checkout containing assets/ (default: current directory)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="JSON report path; parent directories are created if needed",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Open and verify every discovered image; this can take several minutes.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_record(path: Path, *, include_hash: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
    }
    if include_hash:
        record["sha256"] = sha256(path)
    return record


def files_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def image_files(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES]


def image_inventory(paths: list[Path], verify: bool) -> dict[str, Any]:
    suffixes = Counter(path.suffix.lower() for path in paths)
    result: dict[str, Any] = {
        "count": len(paths),
        "by_suffix": dict(sorted(suffixes.items())),
    }
    if not verify:
        return result

    from PIL import Image

    unreadable: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            unreadable.append(str(path.resolve()))
    result["verified"] = len(paths) - len(unreadable)
    result["unreadable"] = unreadable
    return result


def annotation_summary(path: Path) -> dict[str, Any]:
    record = path_record(path, include_hash=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
        return record

    if isinstance(value, list):
        record["root_type"] = "list"
        record["records"] = len(value)
        dictionaries = [item for item in value if isinstance(item, dict)]
        record["dictionary_records"] = len(dictionaries)
        record["field_counts"] = dict(
            sorted(Counter(key for item in dictionaries for key in item).items())
        )
    elif isinstance(value, dict):
        record["root_type"] = "object"
        record["keys"] = sorted(value)
        list_fields = {
            key: len(item) for key, item in value.items() if isinstance(item, list)
        }
        if list_fields:
            record["list_fields"] = dict(sorted(list_fields.items()))
    else:
        record["root_type"] = type(value).__name__
    return record


def annotation_image_references(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key.lower() in IMAGE_REFERENCE_FIELDS
                and isinstance(item, str)
                and Path(item).suffix.lower() in IMAGE_SUFFIXES
            ):
                yield item
            elif isinstance(item, (dict, list)):
                yield from annotation_image_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from annotation_image_references(item)


def image_reference_inventory(
    annotation: Path, dataset_root: Path, images: list[Path]
) -> dict[str, Any]:
    try:
        value = json.loads(annotation.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"parse_error": f"{type(exc).__name__}: {exc}"}

    index: dict[str, list[Path]] = {}
    for image in images:
        for key in (image.name, image.relative_to(dataset_root).as_posix()):
            index.setdefault(key, []).append(image)

    references = list(annotation_image_references(value))
    missing: set[str] = set()
    ambiguous: set[str] = set()
    resolved = 0
    for reference in references:
        normalized = reference.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        candidate = Path(normalized)
        direct = candidate if candidate.is_absolute() else dataset_root / candidate
        matches = [direct] if direct.is_file() else index.get(normalized, [])
        if not matches:
            matches = index.get(candidate.name, [])
        if len(matches) == 1:
            resolved += 1
        elif len(matches) == 0:
            missing.add(reference)
        else:
            ambiguous.add(reference)
    return {
        "references": len(references),
        "unique_references": len(set(references)),
        "resolved_references": resolved,
        "missing_references": len(missing),
        "missing_reference_examples": sorted(missing)[:20],
        "ambiguous_references": len(ambiguous),
        "ambiguous_reference_examples": sorted(ambiguous)[:20],
    }


def dataset_inventory(raw_root: Path, name: str, verify_images: bool) -> dict[str, Any]:
    dataset_root = raw_root / name
    if not dataset_root.exists():
        return {"expected_root": str(dataset_root.resolve()), "exists": False}

    files = files_under(dataset_root)
    images = image_files(files)
    annotations = [path for path in files if path.suffix.lower() == ".json"]
    archives = [path for path in files if path.suffix.lower() in ARCHIVE_SUFFIXES]
    return {
        "expected_root": str(dataset_root.resolve()),
        "exists": True,
        "file_count": len(files),
        "image_inventory": image_inventory(images, verify_images),
        "image_parent_directories": sorted({str(path.parent.resolve()) for path in images}),
        "annotation_candidates": [
            {
                **annotation_summary(path),
                "image_reference_inventory": image_reference_inventory(path, dataset_root, images),
            }
            for path in annotations
        ],
        "archives": [path_record(path, include_hash=True) for path in archives],
    }


def eurosat_classes(raw_root: Path) -> dict[str, Any]:
    root = raw_root / "eurosat"
    if not root.is_dir():
        return {"exists": False}
    image_paths = image_files(files_under(root))
    by_parent = Counter(path.parent.name for path in image_paths)
    return {"exists": True, "classes_by_leaf_directory": dict(sorted(by_parent.items()))}


def candidate_directories(raw_root: Path, files: list[Path], dataset: str) -> list[str]:
    hints = DIRECTORY_HINTS[dataset]
    candidates: set[Path] = set()
    for file in files:
        for directory in (file.parent, *file.parents):
            if directory == raw_root.parent:
                break
            if any(hint in directory.name.lower() for hint in hints):
                candidates.add(directory.resolve())
            if directory == raw_root:
                break
    return [str(path) for path in sorted(candidates)]


def build_report(project_root: Path, verify_images: bool) -> dict[str, Any]:
    root = project_root.resolve()
    checkpoints_root = root / "assets" / "checkpoints"
    raw_root = root / "assets" / "data" / "raw"
    all_raw_files = files_under(raw_root)
    raw_archives = [path for path in all_raw_files if path.suffix.lower() in ARCHIVE_SUFFIXES]
    return {
        "project_root": str(root),
        "raw_directory_entries": (
            [str(path.resolve()) for path in sorted(raw_root.iterdir()) if path.is_dir()]
            if raw_root.is_dir()
            else []
        ),
        "checkpoints": [
            {
                "name": name,
                "exists": (path := checkpoints_root / name).is_file(),
                **(path_record(path, include_hash=True) if path.is_file() else {}),
            }
            for name in CHECKPOINTS
        ],
        "raw_archives": [path_record(path, include_hash=True) for path in raw_archives],
        "dataset_directory_candidates": {
            dataset: candidate_directories(raw_root, all_raw_files, dataset)
            for dataset in DIRECTORY_HINTS
        },
        "datasets": {
            "chatearthnet": dataset_inventory(raw_root, "chatearthnet", verify_images),
            "eurosat": {
                **dataset_inventory(raw_root, "eurosat", verify_images),
                "class_inventory": eurosat_classes(raw_root),
            },
            "rsicd": dataset_inventory(raw_root, "rsicd", verify_images),
        },
    }


def main() -> None:
    args = parse_args()
    report = build_report(args.project_root, args.verify_images)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(report_path.suffix + ".part")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, report_path)
    print(
        json.dumps({"report": str(report_path), "datasets": report["datasets"]}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
