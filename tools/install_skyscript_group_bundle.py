#!/usr/bin/env python3
"""Install only reference train/val groups from a polished top30 candidate ZIP.

The original representative paths determine the images root. Existing images
are reused only when byte size and ZIP CRC match; references are never remapped.
The bundle remains on disk after installation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    from extract_skyscript_subset import _crc32_file, extract_members, index_selected_members
    from prepare_skyscript_groups import normalize_group, prepare_grouped_manifests
else:
    from tools.extract_skyscript_subset import (
        _crc32_file,
        extract_members,
        index_selected_members,
    )
    from tools.prepare_skyscript_groups import normalize_group, prepare_grouped_manifests


DEFAULT_TRAIN = Path(
    "assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl"
)
DEFAULT_VAL = Path(
    "assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
)
DEFAULT_OUTPUT = Path("assets/data/manifests/skyscript_caption_groups_v1")
METADATA_NAMES = {"_metadata/candidates.csv", "_metadata/audit.json", "_metadata/README.txt"}


def _safe_name(value: str, *, image: bool = False) -> str:
    path = PurePosixPath(value)
    if (
        not value or "\\" in value or path.is_absolute() or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"Unsafe or noncanonical bundle path: {value!r}")
    if image and (
        len(path.parts) < 2 or path.parts[0] not in {"images2", "images3"}
        or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    ):
        raise ValueError(f"Unsupported bundle image path: {value!r}")
    return value


def _references(
    train_reference: Path, val_reference: Path,
) -> tuple[Path, dict[str, str], dict[str, tuple[str, str, str]]]:
    roots: set[Path] = set()
    groups: dict[str, str] = {}
    representatives: dict[str, tuple[str, str, str]] = {}
    ids: set[str] = set()
    for split, reference in (("train", train_reference), ("val", val_reference)):
        count = 0
        with reference.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{reference}:{line_number}: reference must be an object")
                for field in ("id", "image", "caption"):
                    if not isinstance(record.get(field), str) or not record[field].strip():
                        raise ValueError(f"{reference}:{line_number}: missing {field}")
                if record.get("source") != "SkyScript" or record.get("split") != split:
                    raise ValueError(
                        f"{reference}:{line_number}: invalid reference source or split"
                    )
                image = Path(record["image"])
                indices = [
                    i for i, part in enumerate(image.parts) if part in {"images2", "images3"}
                ]
                if not image.is_absolute() or ".." in image.parts or len(indices) != 1:
                    raise ValueError(f"Cannot infer images root from reference path: {image}")
                index = indices[0]
                root = Path(*image.parts[:index]).resolve()
                relative = _safe_name(PurePosixPath(*image.parts[index:]).as_posix(), image=True)
                resolved = image.resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError(f"Reference path escapes images root: {image}")
                group = normalize_group(record["caption"])
                if (
                    not group or group in groups or relative in representatives
                    or record["id"] in ids
                ):
                    raise ValueError("Reference groups, image paths or IDs overlap/duplicate")
                roots.add(root)
                groups[group] = split
                representatives[relative] = (split, group, record["id"])
                ids.add(record["id"])
                count += 1
        if not count:
            raise ValueError(f"Empty {split} reference manifest")
    if len(roots) != 1:
        raise ValueError("Reference images must share one existing images root")
    root = next(iter(roots))
    resolved_paths = [(root / relative).resolve() for relative in representatives]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Reference image paths resolve to duplicate images")
    return root, groups, representatives


def install_group_bundle(
    *, bundle: Path, train_reference: Path = DEFAULT_TRAIN,
    val_reference: Path = DEFAULT_VAL, output_dir: Path = DEFAULT_OUTPUT,
    max_member_bytes: int = 100 * 1024 * 1024,
) -> dict[str, Any]:
    destination = output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output directory already exists: {destination}")
    if max_member_bytes <= 0:
        raise ValueError("max_member_bytes must be positive")
    root, groups, representatives = _references(train_reference, val_reference)
    members: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(bundle) as archive:
        for info in archive.infolist():
            name = _safe_name(info.filename)
            if name in members:
                raise ValueError(f"Duplicate bundle member: {name}")
            member_type = stat.S_IFMT(info.external_attr >> 16)
            if info.is_dir() or member_type not in {0, stat.S_IFREG}:
                raise ValueError(f"Bundle member must be a regular file: {name}")
            if name not in METADATA_NAMES:
                _safe_name(name, image=True)
            if info.file_size > max_member_bytes:
                raise ValueError(f"Bundle member exceeds max size: {name}")
            members[name] = info
        if missing_metadata := METADATA_NAMES - members.keys():
            raise ValueError(f"Missing bundle metadata: {sorted(missing_metadata)}")
        producer_audit = json.loads(archive.read("_metadata/audit.json"))
        if not isinstance(producer_audit, dict):
            raise ValueError("Bundle audit must be an object")
        archive.read("_metadata/README.txt").decode("utf-8")
        csv_bytes = archive.read("_metadata/candidates.csv")
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"), newline=""))
    if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
        raise ValueError("Candidate CSV header contains duplicate columns")
    if not {"filepath", "title_raw"} <= set(reader.fieldnames or []):
        raise ValueError("Candidate CSV must contain filepath and title_raw")
    csv_groups: dict[str, str] = {}
    selected: set[str] = set()
    resolved_paths: set[Path] = set()
    for row in reader:
        if None in row:
            raise ValueError("Candidate CSV row has more values than its header")
        relative = _safe_name(row.get("filepath") or "", image=True)
        group = normalize_group(row.get("title_raw") or "")
        if not group or relative in csv_groups:
            raise ValueError(f"Duplicate CSV image or empty caption: {relative}")
        csv_groups[relative] = group
        image = (root / relative).resolve()
        if not image.is_relative_to(root) or image in resolved_paths:
            raise ValueError(f"CSV image escapes root or aliases another image: {relative}")
        resolved_paths.add(image)
        if relative in representatives and representatives[relative][1] != group:
            raise ValueError(f"Reference representative has conflicting CSV caption: {relative}")
        if group in groups:
            selected.add(relative)
    if set(csv_groups) != members.keys() - METADATA_NAMES:
        raise ValueError("Bundle images and candidate CSV paths do not match exactly")
    if missing_representatives := representatives.keys() - selected:
        raise ValueError(
            f"Missing reference representatives: {sorted(missing_representatives)[:5]}"
        )
    reference_ids = {record[2] for record in representatives.values()}
    if any(
        f"skyscript:{relative}" in reference_ids
        for relative in selected - representatives.keys()
    ):
        raise ValueError("Restored image ID conflicts with an original representative ID")
    matches, _ = index_selected_members([bundle], selected, max_member_bytes=max_member_bytes)
    # Preflight every existing file before extracting any new image.
    for relative, member in matches.items():
        image = (root / relative).resolve()
        if image.exists() and not (
            image.is_file() and image.stat().st_size == member.file_size
            and _crc32_file(image) == member.crc
        ):
            raise FileExistsError(f"Refusing to overwrite non-matching extracted file: {image}")
    candidates_csv = (root / "_metadata/group_candidates_v1.csv").resolve()
    if not candidates_csv.is_relative_to(root):
        raise ValueError("Candidate metadata path escapes images root")
    if candidates_csv.exists() and not (
        candidates_csv.is_file() and candidates_csv.read_bytes() == csv_bytes
    ):
        raise FileExistsError(f"Refusing to overwrite different candidate CSV: {candidates_csv}")
    extraction = extract_members(matches, root)
    if not candidates_csv.exists():
        candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".group_candidates_v1.csv.part-", dir=candidates_csv.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(csv_bytes)
            if candidates_csv.exists():
                if candidates_csv.read_bytes() != csv_bytes:
                    raise FileExistsError(
                        f"Candidate CSV appeared with different contents: {candidates_csv}"
                    )
                temporary.unlink()
            else:
                os.replace(temporary, candidates_csv)
        finally:
            temporary.unlink(missing_ok=True)
    audit = prepare_grouped_manifests(
        csv_path=candidates_csv, images_root=root, train_reference=train_reference,
        val_reference=val_reference, output_dir=destination,
    )
    return {
        "status": "complete", "bundle": str(bundle.resolve()), "images_root": str(root),
        "output_dir": str(destination), "candidate_csv": str(candidates_csv),
        "candidate_images": len(csv_groups), "eligible_images": len(selected),
        "excluded_unknown_group_images": len(csv_groups) - len(selected),
        **extraction, "splits": audit["splits"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--train-reference", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-reference", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-member-bytes", type=int, default=100 * 1024 * 1024)
    args = parser.parse_args()
    print(json.dumps(install_group_bundle(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
