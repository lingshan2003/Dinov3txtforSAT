#!/usr/bin/env python3
"""Select SkyScript rows from CSV and extract only their images from ZIP archives.

The source archives must be fully downloaded because a ZIP central directory is
normally stored at the end of the file.  They do not need to be fully extracted:
this tool resolves the selected ``filepath`` values against archive members,
fails before writing if any selection is missing or ambiguous, and then copies
only the requested images.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import heapq
import json
import os
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class SelectedRow:
    priority: int
    filepath: str
    row: dict[str, str]


@dataclass(frozen=True)
class ArchiveMember:
    archive: Path
    member: str
    file_size: int
    crc: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="SkyScript source CSV")
    parser.add_argument(
        "--archives",
        required=True,
        nargs="+",
        type=Path,
        help="One or more fully downloaded SkyScript ZIP archives",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--selection-output",
        required=True,
        type=Path,
        help="Selected rows as a standalone CSV for later manifest preparation",
    )
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--path-field", default="filepath")
    parser.add_argument(
        "--group-field",
        default="title_raw",
        help="Rows with the same normalized value share a sampling group",
    )
    parser.add_argument("--max-per-group", type=int, default=1)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--selection-prefix",
        nargs="+",
        help=(
            "Restrict grouped sampling to these top-level folders before caption deduplication. "
            "For the fast two-archive pilot use: images2 images3."
        ),
    )
    parser.add_argument(
        "--all-csv-rows",
        action="store_true",
        help=(
            "Extract every CSV row under --include-prefix instead of the grouped 50k pilot; "
            "an include prefix is required to keep per-run memory and outputs shard-bounded"
        ),
    )
    parser.add_argument(
        "--include-prefix",
        nargs="+",
        help=(
            "Extract only globally selected paths below these top-level folders, for example "
            "images1. Use this to process one large archive at a time."
        ),
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        help="Optional ZIP_STORED bundle containing selected images and selection CSV",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and index archives without writing images, selection CSV, bundle, or audit",
    )
    parser.add_argument(
        "--max-member-bytes",
        type=int,
        default=100 * 1024 * 1024,
        help="Refuse an unexpectedly large selected archive member (default: 100 MiB)",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def safe_relative_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe or empty SkyScript filepath: {value!r}")
    if Path(path.name).suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported SkyScript image suffix: {value!r}")
    return path.as_posix()


def deterministic_priority(seed: int, filepath: str) -> int:
    digest = hashlib.sha256(f"{seed}\0skyscript:{filepath}".encode()).digest()
    return int.from_bytes(digest[:16], "big")


def _bounded_push(
    heap: list[tuple[int, str, SelectedRow]], *, capacity: int, selected: SelectedRow
) -> None:
    item = (-selected.priority, selected.filepath, selected)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def select_rows(
    csv_path: Path,
    *,
    path_field: str,
    group_field: str,
    max_per_group: int,
    limit: int,
    seed: int,
    all_csv_rows: bool = False,
    include_prefixes: set[str] | None = None,
    selection_prefixes: set[str] | None = None,
) -> tuple[list[SelectedRow], list[str], dict[str, Any]]:
    if max_per_group <= 0 or limit <= 0:
        raise ValueError("max_per_group and limit must be positive")
    source = csv_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if all_csv_rows and not include_prefixes:
        raise ValueError("all_csv_rows requires at least one include prefix")
    group_heaps: dict[str, list[tuple[int, str, SelectedRow]]] = {}
    all_selected: list[SelectedRow] = []
    group_counts: Counter[str] = Counter()
    seen_paths: set[str] = set()
    rows = 0
    rows_in_selection_scope = 0
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_fields = sorted({path_field, group_field} - set(fieldnames))
        if missing_fields:
            raise ValueError(f"CSV is missing required columns: {missing_fields}")
        for line_number, row in enumerate(reader, start=2):
            rows += 1
            if None in row:
                raise ValueError(f"{source}:{line_number}: row has more values than the header")
            filepath = safe_relative_path(row.get(path_field, ""))
            if filepath in seen_paths:
                raise ValueError(f"{source}:{line_number}: duplicate filepath: {filepath}")
            seen_paths.add(filepath)
            filepath_prefix = PurePosixPath(filepath).parts[0]
            in_selection_scope = (
                selection_prefixes is None or filepath_prefix in selection_prefixes
            )
            if not in_selection_scope:
                continue
            rows_in_selection_scope += 1
            group = normalize_text(row.get(group_field, "")).casefold()
            if not group:
                raise ValueError(f"{source}:{line_number}: empty group field {group_field!r}")
            group_counts[group] += 1
            selected = SelectedRow(
                priority=deterministic_priority(seed, filepath),
                filepath=filepath,
                row={name: row.get(name, "") for name in fieldnames},
            )
            if all_csv_rows:
                if filepath_prefix in include_prefixes:
                    all_selected.append(selected)
                continue
            heap = group_heaps.setdefault(group, [])
            _bounded_push(heap, capacity=max_per_group, selected=selected)
    if not rows:
        raise ValueError(f"SkyScript CSV contains no data rows: {source}")
    if all_csv_rows:
        selected_rows = all_selected
        if not selected_rows:
            raise ValueError(
                f"No CSV rows match include prefixes: {sorted(include_prefixes or set())}"
            )
        selection_strategy = "all_csv_rows_matching_include_prefix"
        global_selected_rows = rows
    else:
        candidates = [item[2] for heap in group_heaps.values() for item in heap]
        candidates.sort(key=lambda item: (item.priority, item.filepath))
        selected_rows = candidates[:limit]
        if len(selected_rows) != limit:
            raise ValueError(
                f"Grouped selection produced {len(selected_rows)} rows, not requested "
                f"limit={limit}; raise --max-per-group or lower --limit"
            )
        selection_strategy = "lowest_sha256_priority_per_group_then_globally"
        global_selected_rows = len(selected_rows)
    audit = {
        "source_csv": str(source),
        "source_rows": rows,
        "rows_in_selection_scope": rows_in_selection_scope,
        "unique_filepaths": len(seen_paths),
        "selection_prefixes": (
            None if selection_prefixes is None else sorted(selection_prefixes)
        ),
        "group_field": group_field,
        "unique_normalized_groups": len(group_counts),
        "duplicate_group_records": sum(count - 1 for count in group_counts.values()),
        "max_group_frequency": max(group_counts.values()),
        "max_per_group": max_per_group,
        "limit": limit,
        "seed": seed,
        "all_csv_rows": all_csv_rows,
        "global_selected_rows": global_selected_rows,
        "selected_rows_in_this_shard": len(selected_rows),
        "selection_strategy": selection_strategy,
    }
    return selected_rows, fieldnames, audit


def _member_candidates(
    member: str, archive_stem: str, selected_roots: set[str]
) -> list[str]:
    normalized = member.replace("\\", "/").lstrip("/")
    parts = PurePosixPath(normalized).parts
    candidates = [PurePosixPath(*parts[index:]).as_posix() for index in range(len(parts))]
    if parts and parts[0] != archive_stem:
        candidates.append(PurePosixPath(archive_stem, *parts).as_posix())
    if len(parts) == 1:
        candidates.extend(
            PurePosixPath(root, *parts).as_posix() for root in sorted(selected_roots)
        )
    return list(dict.fromkeys(candidates))


def index_selected_members(
    archives: list[Path], selected_paths: set[str], *, max_member_bytes: int
) -> tuple[dict[str, ArchiveMember], dict[str, Any]]:
    if max_member_bytes <= 0:
        raise ValueError("max_member_bytes must be positive")
    resolved_archives = [path.resolve() for path in archives]
    if len(set(resolved_archives)) != len(resolved_archives):
        raise ValueError("The same ZIP archive was supplied more than once")
    matches: dict[str, ArchiveMember] = {}
    archive_reports: list[dict[str, Any]] = []
    selected_roots = {PurePosixPath(path).parts[0] for path in selected_paths}
    for archive in resolved_archives:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        matched_here = 0
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            for info in members:
                if info.is_dir():
                    continue
                target = next(
                    (
                        candidate
                        for candidate in _member_candidates(
                            info.filename, archive.stem, selected_roots
                        )
                        if candidate in selected_paths
                    ),
                    None,
                )
                if target is None:
                    continue
                if info.file_size > max_member_bytes:
                    raise ValueError(
                        f"Selected member exceeds max size: {archive}:{info.filename} "
                        f"has {info.file_size} bytes"
                    )
                current = ArchiveMember(
                    archive=archive,
                    member=info.filename,
                    file_size=info.file_size,
                    crc=info.CRC,
                )
                if target in matches:
                    previous = matches[target]
                    raise ValueError(
                        f"Selected path {target!r} is ambiguous: "
                        f"{previous.archive}:{previous.member} and {archive}:{info.filename}"
                    )
                matches[target] = current
                matched_here += 1
        archive_reports.append(
            {
                "path": str(archive),
                "members": len(members),
                "selected_members": matched_here,
            }
        )
    missing = sorted(selected_paths - set(matches))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected SkyScript images were not found in the supplied archives; "
            f"first: {missing[:10]}"
        )
    return matches, {"archives": archive_reports}


def _crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value = binascii.crc32(chunk, value)
    return value & 0xFFFFFFFF


def _copy_member(source: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def extract_members(
    matches: dict[str, ArchiveMember], output_root: Path
) -> dict[str, int]:
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    grouped: dict[Path, list[tuple[str, ArchiveMember]]] = {}
    for target, member in matches.items():
        grouped.setdefault(member.archive, []).append((target, member))
    extracted = reused = 0
    for archive in sorted(grouped, key=str):
        with zipfile.ZipFile(archive) as handle:
            for target, member in sorted(grouped[archive]):
                destination = (root / Path(*PurePosixPath(target).parts)).resolve()
                destination.relative_to(root)
                if destination.exists():
                    if (
                        destination.is_file()
                        and destination.stat().st_size == member.file_size
                        and _crc32_file(destination) == member.crc
                    ):
                        reused += 1
                        continue
                    raise FileExistsError(
                        f"Refusing to overwrite non-matching extracted file: {destination}"
                    )
                with handle.open(member.member) as source:
                    _copy_member(source, destination)
                if destination.stat().st_size != member.file_size:
                    raise OSError(f"Extracted size mismatch: {destination}")
                if _crc32_file(destination) != member.crc:
                    raise OSError(f"Extracted CRC mismatch: {destination}")
                extracted += 1
    return {"extracted": extracted, "reused_verified": reused}


def write_selection_csv(
    path: Path, rows: list[SelectedRow], fieldnames: list[str]
) -> str:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected.row for selected in rows)
        if destination.exists():
            if destination.is_file() and destination.read_bytes() == temporary.read_bytes():
                temporary.unlink()
                return "reused_identical"
            raise FileExistsError(
                f"Refusing to overwrite different selection CSV: {destination}"
            )
        os.replace(temporary, destination)
        return "written"
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_bundle(
    path: Path,
    *,
    output_root: Path,
    selected_paths: list[str],
    selection_csv: Path,
) -> None:
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as out:
            out.write(selection_csv, arcname="_metadata/selection.csv")
            for relative in selected_paths:
                source = output_root / Path(*PurePosixPath(relative).parts)
                out.write(source, arcname=relative)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def parse_prefixes(values: list[str] | None, option: str) -> set[str] | None:
    if not values:
        return None
    prefixes: set[str] = set()
    for value in values:
        candidate = PurePosixPath(value.strip())
        if len(candidate.parts) != 1 or candidate.parts[0] in {"", ".", ".."}:
            raise ValueError(f"Invalid top-level {option}: {value!r}")
        prefixes.add(candidate.parts[0])
    return prefixes


def main() -> None:
    args = parse_args()
    prefixes = parse_prefixes(args.include_prefix, "--include-prefix")
    selection_prefixes = parse_prefixes(args.selection_prefix, "--selection-prefix")
    if args.all_csv_rows and selection_prefixes is not None:
        raise ValueError("--selection-prefix is only valid for grouped sampling")
    if (
        prefixes is not None
        and selection_prefixes is not None
        and not prefixes <= selection_prefixes
    ):
        raise ValueError("--include-prefix must be a subset of --selection-prefix")
    global_selection, fieldnames, selection_audit = select_rows(
        args.csv,
        path_field=args.path_field,
        group_field=args.group_field,
        max_per_group=args.max_per_group,
        limit=args.limit,
        seed=args.seed,
        all_csv_rows=args.all_csv_rows,
        include_prefixes=prefixes,
        selection_prefixes=selection_prefixes,
    )
    selected = global_selection
    if prefixes and not args.all_csv_rows:
        selected = [
            row
            for row in global_selection
            if PurePosixPath(row.filepath).parts[0] in prefixes
        ]
        if not selected:
            raise ValueError(f"No globally selected rows match --include-prefix {sorted(prefixes)}")
    selected_paths = [row.filepath for row in selected]
    matches, archive_audit = index_selected_members(
        args.archives, set(selected_paths), max_member_bytes=args.max_member_bytes
    )
    selected_bytes = sum(member.file_size for member in matches.values())
    plan = {
        "format_version": 1,
        "selection": selection_audit,
        **archive_audit,
        "selected_images": len(matches),
        "global_selected_images": selection_audit["global_selected_rows"],
        "include_prefix": None if prefixes is None else sorted(prefixes),
        "selected_uncompressed_bytes": selected_bytes,
        "output_root": str(args.output_root.resolve()),
        "selection_output": str(args.selection_output.resolve()),
        "bundle_output": (
            None if args.bundle_output is None else str(args.bundle_output.resolve())
        ),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    protected_inputs = {args.csv.resolve(), *(path.resolve() for path in args.archives)}
    requested_outputs = {args.audit_output.resolve()}
    if args.bundle_output is not None:
        requested_outputs.add(args.bundle_output.resolve())
    all_output_paths = requested_outputs | {args.selection_output.resolve()}
    if protected_inputs & all_output_paths:
        raise ValueError("Output paths must not overwrite the source CSV or ZIP archives")
    if args.selection_output.resolve() in requested_outputs:
        raise ValueError("Selection, audit, and bundle outputs must be different paths")
    if len(requested_outputs) != 1 + (args.bundle_output is not None):
        raise ValueError("Audit and bundle outputs must be different paths")
    existing_outputs = sorted(path for path in requested_outputs if path.exists())
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing_outputs)
        )

    args.output_root.resolve().parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output_root.resolve().parent).free
    required_bytes = selected_bytes
    if args.bundle_output is not None:
        required_bytes += selected_bytes
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free space: need at least {required_bytes} bytes, have {free_bytes}"
        )
    selection_status = write_selection_csv(
        args.selection_output, global_selection, fieldnames
    )
    extraction = extract_members(matches, args.output_root)
    if args.bundle_output is not None:
        write_bundle(
            args.bundle_output,
            output_root=args.output_root.resolve(),
            selected_paths=selected_paths,
            selection_csv=args.selection_output.resolve(),
        )
    report = {
        **plan,
        **extraction,
        "selection_csv_status": selection_status,
        "source_csv_sha256": sha256_file(args.csv.resolve()),
        "selection_csv_sha256": sha256_file(args.selection_output.resolve()),
        "bundle_sha256": (
            None if args.bundle_output is None else sha256_file(args.bundle_output.resolve())
        ),
        "status": "complete",
    }
    write_json_atomic(args.audit_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
