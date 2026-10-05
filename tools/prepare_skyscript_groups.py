#!/usr/bin/env python3
"""Restore SkyScript images without changing an existing caption-group holdout.

No new split or representative selection is performed. The unique train/val
manifests are authoritative, including their exact canonical caption wording.
Unknown caption groups and unselected image prefixes are excluded explicitly.
The input CSV must already use the original experiment's quality filter (e.g.
polished top30). This tool does not filter scores or select a top30 subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    from prepare_skyscript import safe_relative_path
else:
    from tools.prepare_skyscript import safe_relative_path


def normalize_group(caption: str) -> str:
    return " ".join(caption.split()).casefold()


def _image_path(root: Path, relative: str, prefixes: set[str]) -> Path:
    reference = safe_relative_path(relative)
    if reference.parts[0] not in prefixes:
        raise ValueError(f"Reference image uses an unselected prefix: {relative}")
    image = (root / Path(*reference.parts)).resolve()
    if not image.is_relative_to(root):
        raise ValueError(f"Image path escapes images root: {relative}")
    return image


def _read_reference(
    path: Path, split: str, root: Path, prefixes: set[str]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    seen_images: set[str] = set()
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            for field in ("id", "image", "caption"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"{path}:{line_number}: missing or empty {field}")
            if record.get("split") != split:
                raise ValueError(f"{path}:{line_number}: expected split={split!r}")
            if record.get("source") != "SkyScript":
                raise ValueError(f"{path}:{line_number}: expected source='SkyScript'")
            image = Path(record["image"]).resolve()
            if not image.is_relative_to(root):
                raise ValueError(f"Reference image is outside images root: {image}")
            relative = image.relative_to(root).as_posix()
            _image_path(root, relative, prefixes)
            if not image.is_file():
                raise FileNotFoundError(f"Reference representative is missing: {image}")
            group = normalize_group(record["caption"])
            if not group:
                raise ValueError(f"{path}:{line_number}: empty normalized caption")
            if group in groups or str(image) in seen_images or record["id"] in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate reference group, image or id")
            groups[group] = record
            seen_images.add(str(image))
            seen_ids.add(record["id"])
    if not groups:
        raise ValueError(f"Reference manifest is empty: {path}")
    return groups


def _group_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["group_id"] for record in records)
    sizes = sorted(counts.values())
    return {
        "records": len(records),
        "caption_groups": len(counts),
        "singleton_groups": sum(size == 1 for size in sizes),
        "group_size_histogram": dict(sorted(Counter(sizes).items())),
        "group_size_p50": sizes[round((len(sizes) - 1) * 0.5)],
        "group_size_p95": sizes[round((len(sizes) - 1) * 0.95)],
        "group_size_max": sizes[-1],
        "image_prefix_counts": dict(
            sorted(Counter(record["csv_filepath"].split("/")[0] for record in records).items())
        ),
    }


def prepare_grouped_manifests(
    *,
    csv_path: Path,
    images_root: Path,
    train_reference: Path,
    val_reference: Path,
    output_dir: Path,
    caption_field: str = "title_raw",
    allow_prefixes: tuple[str, ...] = ("images2", "images3"),
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Validate fully, then publish two manifests and their audit together."""
    root = images_root.resolve()
    destination = output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output directory already exists: {destination}")
    if not root.is_dir():
        raise NotADirectoryError(root)
    prefixes = set(allow_prefixes)
    if not prefixes or any(
        not prefix or prefix in {".", ".."} or "/" in prefix or "\\" in prefix
        for prefix in prefixes
    ):
        raise ValueError("Allowed prefixes must be nonempty single directory names")
    references = {
        "train": _read_reference(train_reference, "train", root, prefixes),
        "val": _read_reference(val_reference, "val", root, prefixes),
    }
    if references["train"].keys() & references["val"].keys():
        raise ValueError("Train and val reference caption groups overlap")
    image_membership: dict[str, tuple[str, str]] = {}
    reference_by_image: dict[str, dict[str, Any]] = {}
    reference_ids: set[str] = set()
    for split, groups in references.items():
        for group, record in groups.items():
            image = str(Path(record["image"]).resolve())
            if image in image_membership or record["id"] in reference_ids:
                raise ValueError("Train and val reference images or IDs overlap")
            image_membership[image] = (split, group)
            reference_by_image[image] = record
            reference_ids.add(record["id"])
    group_split = {group: split for split, groups in references.items() for group in groups}
    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    seen_paths: set[str] = set()
    seen_resolved: set[str] = set()
    seen_ids: set[str] = set()
    retained_representatives: set[str] = set()
    counts: Counter[str] = Counter()
    missing_preview: list[str] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filepath", caption_field}
        if missing_columns := required - set(reader.fieldnames or []):
            raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")
        for line_number, row in enumerate(reader, start=2):
            counts["csv_rows"] += 1
            try:
                relative = safe_relative_path(row.get("filepath") or "")
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{line_number}: {exc}") from exc
            if relative.parts[0] not in prefixes:
                counts["excluded_prefix_rows"] += 1
                continue
            relative_text = relative.as_posix()
            image = _image_path(root, relative_text, prefixes)
            image_text = str(image)
            if relative_text in seen_paths or image_text in seen_resolved:
                raise ValueError(f"Duplicate or aliased CSV image: {relative_text}")
            seen_paths.add(relative_text)
            seen_resolved.add(image_text)
            group = normalize_group(row.get(caption_field) or "")
            if not group:
                raise ValueError(f"{csv_path}:{line_number}: empty caption")
            if image_text in image_membership and image_membership[image_text][1] != group:
                raise ValueError(f"Reference image has conflicting CSV caption: {relative_text}")
            if group not in group_split:
                counts["excluded_unknown_group_rows"] += 1
                continue
            counts["eligible_rows"] += 1
            if not image.is_file():
                counts["missing_eligible_rows"] += 1
                if len(missing_preview) < 100:
                    missing_preview.append(relative_text)
                continue
            split = group_split[group]
            representative = references[split][group]
            is_representative = image_text in reference_by_image
            record = (
                dict(reference_by_image[image_text])
                if is_representative
                else {
                    "id": f"skyscript:{relative_text}",
                    "image": image_text,
                    "split": split,
                    "source": "SkyScript",
                }
            )
            if record["id"] in seen_ids or (
                not is_representative and record["id"] in reference_ids
            ):
                raise ValueError(f"Duplicate or conflicting restored ID: {record['id']}")
            seen_ids.add(record["id"])
            record.update(
                caption=representative["caption"],
                group_id=group,
                csv_filepath=relative_text,
                reference_representative=is_representative,
            )
            records[split].append(record)
            if is_representative:
                retained_representatives.add(image_text)
    if counts["missing_eligible_rows"] and not allow_missing:
        raise FileNotFoundError(
            f"{counts['missing_eligible_rows']} eligible images are missing; "
            f"first references: {missing_preview[:5]}"
        )
    if missing_representatives := reference_by_image.keys() - retained_representatives:
        raise ValueError(
            "Reference representatives are absent from eligible CSV output: "
            f"{sorted(missing_representatives)[:5]}"
        )
    for split, groups in references.items():
        if {record["group_id"] for record in records[split]} != groups.keys():
            raise ValueError(f"Not every {split} reference group was retained")
        # Keep the original reference order and representative first within each group.
        order = {group: index for index, group in enumerate(groups)}
        records[split].sort(
            key=lambda record: (
                order[record["group_id"]],
                not record["reference_representative"],
                record["csv_filepath"],
            )
        )
    audit = {
        "format_version": 1,
        "positive_definition": "normalized_full_caption_group",
        "normalization": "whitespace_collapse_then_casefold",
        "source_csv": str(csv_path.resolve()),
        "images_root": str(root),
        "train_reference": str(train_reference.resolve()),
        "val_reference": str(val_reference.resolve()),
        "output_dir": str(destination),
        "caption_field": caption_field,
        "allowed_prefixes": sorted(prefixes),
        "allow_missing": allow_missing,
        "row_counts": {key: counts[key] for key in (
            "csv_rows", "eligible_rows", "excluded_prefix_rows",
            "excluded_unknown_group_rows", "missing_eligible_rows",
        )},
        "missing_references_preview": missing_preview,
        "splits": {split: _group_statistics(rows) for split, rows in records.items()},
        "reference_representatives_retained": len(retained_representatives),
        "caption_group_overlap": 0,
        "image_path_overlap": 0,
        "sample_id_overlap": 0,
        "notes": [
            "All siblings of each reference caption group keep its original split.",
            "Exact reference caption strings and representative records are preserved.",
            "Path/ID checks do not establish pixel, geographic or semantic independence.",
            "No content hashes, new sampling or new train/val split were computed.",
            "Input CSV must preserve the original quality filter; no score filtering occurs.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.part-", dir=destination.parent))
    try:
        for split, rows in records.items():
            with (temporary / f"{split}_grouped.jsonl").open("w", encoding="utf-8") as handle:
                for record in rows:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        (temporary / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Output directory already exists: {destination}")
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", required=True, type=Path,
        help="Already quality-filtered polished top30 CSV; this tool does not filter scores",
    )
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--train-reference", required=True, type=Path)
    parser.add_argument("--val-reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--caption-field", default="title_raw")
    parser.add_argument(
        "--allow-prefix", action="append", help="Repeat to replace default images2/images3 prefixes"
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    audit = prepare_grouped_manifests(
        csv_path=args.csv, images_root=args.images_root,
        train_reference=args.train_reference, val_reference=args.val_reference,
        output_dir=args.output_dir, caption_field=args.caption_field,
        allow_prefixes=tuple(args.allow_prefix or ("images2", "images3")),
        allow_missing=args.allow_missing,
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
