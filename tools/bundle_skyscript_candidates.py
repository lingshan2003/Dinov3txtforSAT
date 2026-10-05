#!/usr/bin/env python3
"""Stream only polished top30 images2/3 candidates into a portable upload ZIP.

No extraction directory, new train/val split, caption deduplication, or SHA
verification is needed. JPEG bytes are preserved and stored without recompression.
The server's existing unique manifests remain authoritative for the final split.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    from extract_skyscript_subset import index_selected_members, safe_relative_path
else:
    from tools.extract_skyscript_subset import index_selected_members, safe_relative_path


def read_candidates(csv_path: Path) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    groups: Counter[str] = Counter()
    prefixes: Counter[str] = Counter()
    total = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"filepath", "title_raw"} <= set(reader.fieldnames or []):
            raise ValueError("Input must contain filepath and title_raw (polished top30 CSV)")
        for line_number, row in enumerate(reader, start=2):
            total += 1
            if None in row:
                raise ValueError(f"{csv_path}:{line_number}: extra CSV fields")
            relative = safe_relative_path(row.get("filepath") or "")
            prefix = relative.split("/")[0]
            if prefix not in {"images2", "images3"}:
                continue
            caption = row.get("title_raw") or ""
            group = " ".join(caption.split()).casefold()
            if not group:
                raise ValueError(f"{csv_path}:{line_number}: empty caption")
            if relative in seen:
                raise ValueError(f"Duplicate selected filepath: {relative}")
            seen.add(relative)
            selected.append((relative, caption))
            prefixes[prefix] += 1
            groups[group] += 1
    if not selected:
        raise ValueError("No images2/images3 candidates found")
    return selected, {
        "csv_rows": total,
        "candidate_images": len(selected),
        "caption_groups": len(groups),
        "images_by_prefix": dict(prefixes),
        "singleton_groups": sum(count == 1 for count in groups.values()),
        "largest_group": max(groups.values()),
        "group_size_histogram": dict(sorted(Counter(groups.values()).items())),
    }


def bundle_candidates(
    *, csv_path: Path, archives: list[Path], output: Path, dry_run: bool = False
) -> dict[str, Any]:
    destination = output.resolve()
    inputs = {csv_path.resolve(), *(path.resolve() for path in archives)}
    if destination in inputs:
        raise ValueError("Bundle output must differ from source CSV and archives")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {destination}")
    selected, statistics = read_candidates(csv_path)
    matches, archive_audit = index_selected_members(
        archives, {relative for relative, _ in selected}, max_member_bytes=100 * 1024 * 1024
    )
    audit = {
        "format_version": 1,
        "source_csv": str(csv_path.resolve()),
        "source_csv_name": csv_path.name,
        "selection": "all_polished_top30_rows_in_images2_images3",
        "caption_field": "title_raw",
        "positive_definition": "normalized_complete_caption",
        **statistics,
        **archive_audit,
        "image_bytes": sum(member.file_size for member in matches.values()),
        "split_assignment": "deferred_to_existing_server_train_val_manifests",
        "image_encoding": "original_bytes_no_resize_or_reencode",
        "zip_image_compression": "stored",
        "notes": [
            "CSV must already be the original polished top30 quality-filtered version.",
            "All selected caption siblings are retained; no one-image-per-caption sampling.",
            "ZIP CRC is checked while streaming; no SHA verification is performed.",
            "Server installer preserves the existing representative paths and caption split.",
        ],
    }
    print("bundle_plan=" + json.dumps(audit, ensure_ascii=False), flush=True)
    if dry_run:
        return audit
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"Previous partial bundle exists: {temporary}; inspect it first")
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(("filepath", "title_raw"))
    writer.writerows(selected)
    by_archive: dict[Path, list[tuple[str, str]]] = {}
    for relative, member in matches.items():
        by_archive.setdefault(member.archive, []).append((relative, member.member))
    copied = 0
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as out:
            out.writestr("_metadata/candidates.csv", text.getvalue(),
                         compress_type=zipfile.ZIP_DEFLATED)
            out.writestr("_metadata/audit.json", json.dumps(audit, ensure_ascii=False, indent=2),
                         compress_type=zipfile.ZIP_DEFLATED)
            out.writestr(
                "_metadata/README.txt",
                "Only original polished top30 images2/3 candidates are included.\n"
                "Do not create a new split. On the server, from the project root:\n"
                ".venv/bin/python tools/install_skyscript_group_bundle.py --bundle BUNDLE.zip\n"
                "The installer infers the original image root, reuses matching representatives,\n"
                "and generates caption-group manifests using the existing train/val references.\n"
                "After successful installation, retain images and manifests for training.\n",
            )
            for archive, members in by_archive.items():
                with zipfile.ZipFile(archive) as source:
                    for relative, member_name in members:
                        with source.open(member_name) as image, out.open(
                            relative, "w", force_zip64=True
                        ) as target:
                            shutil.copyfileobj(image, target, length=1024 * 1024)
                        copied += 1
                        if copied % 10000 == 0:
                            print(f"bundled_images={copied}/{len(matches)}", flush=True)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    result = {**audit, "bundle": str(destination), "bundle_bytes": destination.stat().st_size}
    print("bundle_completed=" + json.dumps({key: result[key] for key in (
        "bundle", "bundle_bytes", "candidate_images", "caption_groups"
    )}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--archives", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bundle_candidates(csv_path=args.csv, archives=args.archives, output=args.output,
                      dry_run=args.dry_run)


if __name__ == "__main__":
    main()
