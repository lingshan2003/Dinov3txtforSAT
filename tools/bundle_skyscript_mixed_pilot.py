#!/usr/bin/env python3
"""Build a fixed 1:1 SkyScript short/detail-caption pilot upload bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    from extract_skyscript_subset import index_selected_members, safe_relative_path
    from split_skyscript_selection import priority
else:
    from tools.extract_skyscript_subset import index_selected_members, safe_relative_path
    from tools.split_skyscript_selection import priority

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHORT_CSV = (
    ROOT / "SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv"
)
DEFAULT_DETAIL_CSV = ROOT / "SkyScript_train_top30pct_filtered_by_CLIP_openai.csv"
DEFAULT_ARCHIVES = [ROOT / "images2.zip", ROOT / "images3.zip"]
DEFAULT_OUTPUT = ROOT / "outputs/skyscript_mixed_pilot_16k_1to1_v1.zip"
CSV_FIELDS = ("filepath", "short_caption", "detail_caption", "split", "caption_kind")
STOP_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "with",
    "and",
    "or",
    "by",
    "in",
    "on",
    "for",
    "to",
    "from",
    "near",
    "around",
    "is",
    "are",
    "at",
    "as",
    "its",
}
OSM_FILENAME_ID = re.compile(r"^[anwr]\d+$")


def normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def _words(value: str) -> list[str]:
    return normalize(value).split()


def _meaningfully_differs(short: str, detail: str) -> bool:
    short_words, detail_words = set(_words(short)), set(_words(detail))
    additions = {word.strip(".,;:!?()[]{}\"'") for word in detail_words - short_words}
    additions -= STOP_WORDS
    return normalize(short) != normalize(detail) and len(additions) >= 3


def _rank(seed: int, value: str) -> tuple[int, str]:
    digest = hashlib.blake2b(f"{seed}\0{value}".encode(), digest_size=16).digest()
    return int.from_bytes(digest, "big"), value


def _object_key(relative: str) -> str:
    token = Path(relative).stem.split("_")[0]
    if not OSM_FILENAME_ID.fullmatch(token):
        raise ValueError(f"Cannot derive an OSM object ID from image path: {relative}")
    return token


def _read_csv(
    path: Path,
    required: set[str],
    *,
    image_prefixes: set[str] | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if missing := required - set(fields):
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"{path}:{line_number}: extra CSV fields")
            if image_prefixes is not None:
                relative = safe_relative_path(row.get("filepath", ""))
                if relative.split("/")[0] not in image_prefixes:
                    continue
            rows.append({key: value or "" for key, value in row.items() if key is not None})
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows, fields


def reconstruct_group_split(
    short_rows: list[dict[str, str]], *, val_count: int, seed: int
) -> dict[str, str]:
    groups = {normalize(row["title_raw"]) for row in short_rows}
    groups.discard("")
    if not 0 < val_count < len(groups):
        raise ValueError(f"split val_count={val_count} is invalid for {len(groups)} groups")
    val_groups = {group for _, group in sorted((priority(seed, g), g) for g in groups)[:val_count]}
    return {group: ("val" if group in val_groups else "train") for group in groups}


def select_samples(
    short_rows: list[dict[str, str]],
    detail_rows: list[dict[str, str]],
    *,
    train_count: int,
    val_count: int,
    split_val_count: int,
    split_seed: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if train_count <= 0 or val_count <= 0 or train_count % 2 or val_count % 2:
        raise ValueError("train_count and val_count must be positive even numbers")
    group_split = reconstruct_group_split(short_rows, val_count=split_val_count, seed=split_seed)
    short_by_path: dict[str, dict[str, str]] = {}
    for line, row in enumerate(short_rows, 2):
        relative = safe_relative_path(row.get("filepath", ""))
        if relative.split("/")[0] not in {"images2", "images3"}:
            continue
        if relative in short_by_path:
            raise ValueError(f"Duplicate polished filepath: {relative}")
        caption = row.get("title_raw", "").strip()
        group = normalize(caption)
        if not group:
            raise ValueError(f"polished CSV row {line}: empty short caption")
        short_by_path[relative] = {
            "filepath": relative,
            "short_caption": caption,
            "short_group": group,
        }

    detail_by_path: dict[str, str] = {}
    for _line, row in enumerate(detail_rows, 2):
        relative = safe_relative_path(row.get("filepath", ""))
        if relative.split("/")[0] not in {"images2", "images3"}:
            continue
        if relative in detail_by_path:
            raise ValueError(f"Duplicate OpenAI filepath: {relative}")
        detail_by_path[relative] = row.get("title_multi_objects", "").strip()

    candidates: dict[str, dict[str, dict[str, str]]] = {"train": {}, "val": {}}
    object_memberships: dict[str, set[str]] = {}
    for relative, short in short_by_path.items():
        split = group_split.get(short["short_group"])
        if split:
            object_memberships.setdefault(_object_key(relative), set()).add(split)
    cross_split_objects = {obj for obj, splits in object_memberships.items() if len(splits) > 1}
    stats = Counter()
    for relative, short in short_by_path.items():
        detail = detail_by_path.get(relative, "")
        if not detail:
            stats["missing_detail"] += 1
            continue
        sw, dw = len(short["short_caption"].split()), len(detail.split())
        if sw > 20:
            stats["short_over_20_words"] += 1
            continue
        if not 12 <= dw <= 50:
            stats["detail_outside_12_50_words"] += 1
            continue
        if not _meaningfully_differs(short["short_caption"], detail):
            stats["detail_not_meaningfully_different"] += 1
            continue
        group = short["short_group"]
        if group not in group_split:
            raise ValueError(f"Selected short group absent from reconstructed split: {group}")
        split = group_split[group]
        object_id = _object_key(relative)
        if object_id in cross_split_objects:
            stats["object_in_both_original_splits"] += 1
            continue
        candidate = {**short, "detail_caption": detail, "split": split}
        previous = candidates[split].get(group)
        if previous is None or _rank(seed, relative) < _rank(seed, previous["filepath"]):
            candidates[split][group] = candidate

    # Greedy, stable ranking. Both caption forms reserve uniqueness globally, including
    # unchosen alternate forms, so exact text cannot become a cross-split false negative.
    targets = {"train": train_count, "val": val_count}
    accepted: list[dict[str, str]] = []
    used_captions: set[str] = set()
    used_objects: set[str] = set()
    used_groups: set[str] = set()
    rejected = Counter()
    for split in ("train", "val"):
        required_per_kind = targets[split] // 2
        counts = Counter()
        ranked = sorted(candidates[split].values(), key=lambda row: _rank(seed, row["filepath"]))
        for row in ranked:
            kind = "short" if counts["short"] < required_per_kind else "detail"
            if counts[kind] >= required_per_kind:
                continue
            # SkyScript image filenames carry the OSM object number after the leading 'a'.
            object_id = _object_key(row["filepath"])
            if object_id in used_objects:
                rejected["duplicate_osm_object"] += 1
                continue
            if row["short_group"] in used_groups:
                rejected["duplicate_short_group"] += 1
                continue
            forms = {normalize(row["short_caption"]), normalize(row["detail_caption"])}
            if len(forms) != 2 or forms & used_captions:
                rejected["caption_collision"] += 1
                continue
            row = {**row, "caption_kind": kind}
            accepted.append(row)
            counts[kind] += 1
            used_captions.update(forms)
            used_objects.add(object_id)
            used_groups.add(row["short_group"])
        if counts["short"] != required_per_kind or counts["detail"] != required_per_kind:
            raise ValueError(
                f"Insufficient eligible {split} quota: selected short={counts['short']}/"
                f"{required_per_kind}, detail={counts['detail']}/{required_per_kind}; "
                f"eligible_groups={len(candidates[split])}, rejections={dict(rejected)}"
            )
    audit = {
        "format_version": 1,
        "selection": "one_image_per_original_normalized_short_caption_group",
        "source_short_field": "title_raw",
        "source_detail_field": "title_multi_objects",
        "short_word_range": [1, 20],
        "detail_word_range": [12, 50],
        "meaningful_difference": "at least three non-stopword detail terms absent from short",
        "original_split": {
            "method": "tools.split_skyscript_selection.priority",
            "seed": split_seed,
            "val_groups": split_val_count,
            "all_groups": len(group_split),
        },
        "sample_order": "lowest blake2b(seed\\0filepath) priority",
        "sample_seed": seed,
        "requested": {"train": train_count, "val": val_count},
        "selected": {
            split: dict(Counter(row["caption_kind"] for row in accepted if row["split"] == split))
            for split in ("train", "val")
        },
        "candidate_groups": {split: len(candidates[split]) for split in candidates},
        "rejected": dict(rejected),
        "caption_uniqueness": (
            "all normalized short and detail forms unique globally across selected rows"
        ),
        "osm_object_cross_split_in_full_pool": len(cross_split_objects),
        "selected_unique_osm_objects": len(used_objects),
        "selected_group_provenance": [
            {
                "group": row["short_group"],
                "split": row["split"],
                "filepath": row["filepath"],
                "caption_kind": row["caption_kind"],
            }
            for row in accepted
        ],
        "images": len(accepted),
        "notes": [
            "short-group split reconstructed from the complete polished CSV before selection",
            "OSM object key is the leading filename token (for example a222016617)",
            "image bytes are streamed unchanged and ZIP CRC is checked by zipfile",
        ],
    }
    accepted.sort(key=lambda row: (row["split"], row["caption_kind"], row["short_group"]))
    return accepted, {**audit, "eligibility_rejections": dict(stats)}


def bundle_mixed_pilot(
    *,
    csv_path: Path = DEFAULT_SHORT_CSV,
    detail_csv_path: Path = DEFAULT_DETAIL_CSV,
    archives: list[Path] | None = None,
    output: Path = DEFAULT_OUTPUT,
    train_count: int = 16000,
    val_count: int = 1600,
    split_val_count: int = 4055,
    split_seed: int = 23,
    seed: int = 11,
    dry_run: bool = False,
) -> dict[str, Any]:
    archives = archives or DEFAULT_ARCHIVES
    destination = output.resolve()
    inputs = {csv_path.resolve(), detail_csv_path.resolve(), *(p.resolve() for p in archives)}
    if destination in inputs:
        raise ValueError("Bundle output must differ from source CSVs and archives")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite bundle: {destination}")
    short_rows, _ = _read_csv(
        csv_path, {"filepath", "title_raw"}, image_prefixes={"images2", "images3"}
    )
    detail_rows, _ = _read_csv(
        detail_csv_path, {"filepath", "title_multi_objects"}, image_prefixes={"images2", "images3"}
    )
    samples, audit = select_samples(
        short_rows,
        detail_rows,
        train_count=train_count,
        val_count=val_count,
        split_val_count=split_val_count,
        split_seed=split_seed,
        seed=seed,
    )
    matches, archive_audit = index_selected_members(
        archives, {row["filepath"] for row in samples}, max_member_bytes=100 * 1024 * 1024
    )
    audit = {
        **audit,
        **archive_audit,
        "image_bytes": sum(m.file_size for m in matches.values()),
        "short_csv": str(csv_path.resolve()),
        "detail_csv": str(detail_csv_path.resolve()),
        "image_encoding": "original bytes; no resize or re-encode",
        "zip_image_compression": "stored",
    }
    plan_summary = {
        key: audit[key]
        for key in (
            "original_split",
            "requested",
            "selected",
            "candidate_groups",
            "rejected",
            "osm_object_cross_split_in_full_pool",
            "selected_unique_osm_objects",
            "images",
            "image_bytes",
            "eligibility_rejections",
            "archives",
        )
    }
    print("bundle_plan=" + json.dumps(plan_summary, ensure_ascii=False), flush=True)
    if dry_run:
        return audit
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"Previous partial bundle exists: {temporary}; inspect it first")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows({key: row[key] for key in CSV_FIELDS} for row in samples)
    by_archive: dict[Path, list[tuple[str, str]]] = {}
    for path, member in matches.items():
        by_archive.setdefault(member.archive, []).append((path, member.member))
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as out:
            out.writestr(
                "_metadata/samples.csv", buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED
            )
            out.writestr(
                "_metadata/audit.json",
                json.dumps(audit, ensure_ascii=False, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            out.writestr(
                "_metadata/README.txt",
                "Fixed 1:1 SkyScript mixed-caption pilot.\n"
                "Install on the server with tools/install_skyscript_mixed_pilot.py.\n",
            )
            copied = 0
            for archive, members in by_archive.items():
                with zipfile.ZipFile(archive) as source:
                    for relative, member_name in members:
                        with (
                            source.open(member_name) as image,
                            out.open(relative, "w", force_zip64=True) as target,
                        ):
                            shutil.copyfileobj(image, target, length=1024 * 1024)
                        copied += 1
                        if copied % 5000 == 0:
                            print(f"bundled_images={copied}/{len(matches)}", flush=True)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {**audit, "bundle": str(destination), "bundle_bytes": destination.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=DEFAULT_SHORT_CSV)
    parser.add_argument(
        "--detail-csv", dest="detail_csv_path", type=Path, default=DEFAULT_DETAIL_CSV
    )
    parser.add_argument("--archives", nargs="+", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-count", type=int, default=16000)
    parser.add_argument("--val-count", type=int, default=1600)
    parser.add_argument("--split-val-count", type=int, default=4055)
    parser.add_argument("--split-seed", type=int, default=23)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = bundle_mixed_pilot(**vars(args))
    summary = {
        key: report[key]
        for key in (
            "bundle",
            "bundle_bytes",
            "images",
            "requested",
            "selected",
            "image_bytes",
            "candidate_groups",
            "osm_object_cross_split_in_full_pool",
        )
        if key in report
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
