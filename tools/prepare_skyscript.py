#!/usr/bin/env python3
"""Convert a SkyScript CSV into the project's canonical JSONL contract.

The polished SkyScript CSVs commonly contain ``filepath``, ``title_raw`` and
``title``.  ``title`` must not be passed through the ChatEarthNet "first
sentence" cleaner: doing so would turn many records into the identical caption
``An aerial image.``.  This tool can retain the official title or explicitly
derive the concise clause following the final ``It shows:`` marker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
IT_SHOWS = re.compile(r"(?:^|\s)It\s+shows\s*:\s*(.+?)\s*$", re.IGNORECASE)
COUNTRY_ZOOM = re.compile(r"_(?P<country>[A-Z]{2})_(?P<zoom>\d+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Official SkyScript CSV")
    parser.add_argument(
        "--images-root",
        required=True,
        type=Path,
        help="Directory below which CSV filepath values are resolved exactly",
    )
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument(
        "--caption-field",
        default="title",
        help="CSV column containing the selected text (default: title)",
    )
    parser.add_argument(
        "--caption-mode",
        choices=("auto", "full", "summary", "contextual-summary"),
        default="auto",
        help=(
            "auto keeps ordinary titles but normalizes an available final 'It shows:' summary; "
            "full always keeps the selected field; summary keeps only text after the marker; "
            "contextual-summary restores a single aerial-image prefix and requires the marker"
        ),
    )
    parser.add_argument("--limit", type=int, help="Deterministic hash sample size")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--max-per-caption",
        type=int,
        help="Optional cap per normalized caption before applying --limit",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Omit unresolved images; default is to fail without publishing a manifest",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def caption_from_row(row: dict[str, str], field: str, mode: str) -> str:
    if field not in row:
        raise KeyError(f"CSV has no caption field {field!r}; columns={sorted(row)}")
    full = normalize_text(row[field] or "")
    if not full:
        raise ValueError(f"Empty caption in CSV field {field!r}")
    if mode == "full":
        return full
    matches = list(IT_SHOWS.finditer(full))
    if not matches:
        if mode == "auto":
            return full
        raise ValueError(
            f"caption-mode={mode!r} requires an 'It shows:' marker, got {full!r}"
        )
    summary = normalize_text(matches[-1].group(1))
    if not summary:
        raise ValueError(f"Empty text after 'It shows:' in {full!r}")
    if mode == "summary":
        return summary
    return f"An aerial image. It shows: {summary}"


def safe_relative_path(value: str) -> PurePosixPath:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe or empty SkyScript filepath: {value!r}")
    if Path(path.name).suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image suffix in SkyScript filepath: {value!r}")
    return path


def deterministic_priority(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest()
    return int.from_bytes(digest[:16], "big")


def _bounded_push(
    heap: list[tuple[int, str, dict[str, str]]],
    *,
    capacity: int,
    priority: int,
    record: dict[str, str],
) -> None:
    # Python's heap is a min-heap. Negating the priority keeps the largest
    # (worst) retained priority at index zero.
    item = (-priority, record["id"], record)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _country_zoom(reference: PurePosixPath) -> tuple[str, str]:
    match = COUNTRY_ZOOM.search(reference.stem)
    if not match:
        return "unknown", "unknown"
    return match.group("country"), match.group("zoom") or "unknown"


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _write_record(handle: TextIO, record: dict[str, str]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_manifest(
    *,
    csv_path: Path,
    images_root: Path,
    split: str,
    output: Path,
    caption_field: str,
    caption_mode: str,
    limit: int | None,
    seed: int,
    max_per_caption: int | None,
    allow_missing: bool,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if max_per_caption is not None and max_per_caption <= 0:
        raise ValueError("max_per_caption must be positive")
    source = csv_path.resolve()
    root = images_root.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not root.is_dir():
        raise NotADirectoryError(root)
    destination = output.resolve()
    if destination == source:
        raise ValueError("output must differ from the source CSV")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    total_rows = 0
    valid_rows = 0
    missing: list[str] = []
    seen_ids: set[str] = set()
    caption_counts: Counter[str] = Counter()
    country_counts: Counter[str] = Counter()
    zoom_counts: Counter[str] = Counter()
    word_lengths: list[int] = []
    char_lengths: list[int] = []
    global_heap: list[tuple[int, str, dict[str, str]]] = []
    caption_heaps: dict[str, list[tuple[int, str, dict[str, str]]]] = {}

    try:
        with source.open(encoding="utf-8-sig", newline="") as csv_handle:
            reader = csv.DictReader(csv_handle)
            columns = reader.fieldnames or []
            required = {"filepath", caption_field}
            if missing_columns := sorted(required - set(columns)):
                raise ValueError(f"CSV is missing required columns: {missing_columns}")
            stream_all = limit is None and max_per_caption is None
            output_handle = temporary.open("w", encoding="utf-8") if stream_all else None
            try:
                for line_number, row in enumerate(reader, start=2):
                    total_rows += 1
                    try:
                        reference = safe_relative_path(row.get("filepath", ""))
                        caption = caption_from_row(row, caption_field, caption_mode)
                    except (KeyError, ValueError) as exc:
                        raise ValueError(f"{source}:{line_number}: {exc}") from exc
                    sample_id = f"skyscript:{reference.as_posix()}"
                    if sample_id in seen_ids:
                        raise ValueError(f"{source}:{line_number}: duplicate sample id {sample_id}")
                    seen_ids.add(sample_id)
                    image = (root / Path(*reference.parts)).resolve()
                    try:
                        image.relative_to(root)
                    except ValueError as exc:  # Defensive check in addition to rejecting '..'.
                        raise ValueError(f"Image path escapes images root: {reference}") from exc
                    if not image.is_file():
                        if len(missing) < 100:
                            missing.append(reference.as_posix())
                        continue

                    valid_rows += 1
                    normalized_caption = caption.casefold()
                    caption_counts[normalized_caption] += 1
                    word_lengths.append(len(caption.split()))
                    char_lengths.append(len(caption))
                    country, zoom = _country_zoom(reference)
                    country_counts[country] += 1
                    zoom_counts[zoom] += 1
                    record = {
                        "id": sample_id,
                        "image": str(image),
                        "caption": caption,
                        "split": split,
                        "source": "SkyScript",
                    }
                    if output_handle is not None:
                        _write_record(output_handle, record)
                        continue
                    priority = deterministic_priority(seed, sample_id)
                    if max_per_caption is not None:
                        heap = caption_heaps.setdefault(normalized_caption, [])
                        _bounded_push(
                            heap, capacity=max_per_caption, priority=priority, record=record
                        )
                    else:
                        if limit is None:  # Kept explicit for type narrowing.
                            raise AssertionError("non-streaming mode requires a selection bound")
                        _bounded_push(global_heap, capacity=limit, priority=priority, record=record)
            finally:
                if output_handle is not None:
                    output_handle.close()

        if total_rows == 0:
            raise ValueError(f"SkyScript CSV contains no data rows: {source}")
        missing_count = total_rows - valid_rows
        if missing_count and not allow_missing:
            raise FileNotFoundError(
                f"{missing_count} of {total_rows} SkyScript images are missing below {root}; "
                f"first references: {missing[:5]}"
            )
        if not valid_rows:
            raise ValueError("No SkyScript records resolve to downloaded images")

        if output_handle is None:
            if max_per_caption is not None:
                candidates = [item for heap in caption_heaps.values() for item in heap]
                candidates.sort(key=lambda item: (-item[0], item[1]))
                if limit is not None:
                    candidates = candidates[:limit]
                selected_records = [item[2] for item in candidates]
            else:
                selected_records = [
                    item[2] for item in sorted(global_heap, key=lambda item: (-item[0], item[1]))
                ]
            if not selected_records:
                raise ValueError("Selection constraints produced an empty manifest")
            if limit is not None and len(selected_records) != limit:
                raise ValueError(
                    f"Selection produced {len(selected_records)} records, not requested "
                    f"limit={limit}; "
                    "download more images or relax --max-per-caption"
                )
            with temporary.open("w", encoding="utf-8") as handle:
                for record in selected_records:
                    _write_record(handle, record)
            selected_count = len(selected_records)
        else:
            selected_count = valid_rows

        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    duplicate_records = sum(count - 1 for count in caption_counts.values())
    audit = {
        "format_version": 1,
        "source_csv": str(source),
        "source_csv_sha256": sha256_file(source),
        "images_root": str(root),
        "output_manifest": str(destination),
        "output_manifest_sha256": sha256_file(destination),
        "split": split,
        "caption_field": caption_field,
        "caption_mode": caption_mode,
        "selection": {
            "strategy": (
                "all_in_csv_order"
                if limit is None and max_per_caption is None
                else "lowest_sha256_priority"
            ),
            "seed": seed,
            "limit": limit,
            "max_per_normalized_caption": max_per_caption,
        },
        "csv_rows": total_rows,
        "resolved_rows": valid_rows,
        "selected_rows": selected_count,
        "missing_count": total_rows - valid_rows,
        "missing_references_preview": missing,
        "caption_statistics_before_sampling": {
            "unique_normalized": len(caption_counts),
            "duplicate_records": duplicate_records,
            "duplicate_fraction": duplicate_records / valid_rows,
            "word_length": {
                "p50": _percentile(word_lengths, 0.50),
                "p95": _percentile(word_lengths, 0.95),
                "max": max(word_lengths),
            },
            "character_length": {
                "p50": _percentile(char_lengths, 0.50),
                "p95": _percentile(char_lengths, 0.95),
                "max": max(char_lengths),
            },
            "top_normalized_captions": caption_counts.most_common(20),
        },
        "country_counts_before_sampling": dict(country_counts.most_common()),
        "zoom_counts_before_sampling": dict(zoom_counts.most_common()),
        "notes": [
            "No ChatEarthNet first-sentence cleaning was applied.",
            "Length statistics are whitespace/character proxies, not dino.txt BPE lengths.",
            "Missing images fail closed unless --allow-missing is explicitly supplied.",
        ],
    }
    return audit


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    audit_output = args.audit_output.resolve()
    if output == audit_output:
        raise ValueError("--output and --audit-output must differ")
    audit = prepare_manifest(
        csv_path=args.csv,
        images_root=args.images_root,
        split=args.split,
        output=output,
        caption_field=args.caption_field,
        caption_mode=args.caption_mode,
        limit=args.limit,
        seed=args.seed,
        max_per_caption=args.max_per_caption,
        allow_missing=args.allow_missing,
    )
    write_json_atomic(audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
