#!/usr/bin/env python3
"""Install and tokenize a fixed mixed-caption SkyScript pilot bundle."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    from bundle_skyscript_mixed_pilot import CSV_FIELDS, _object_key, normalize
    from extract_skyscript_subset import (
        _crc32_file,
        extract_members,
        index_selected_members,
    )
    from install_skyscript_group_bundle import DEFAULT_TRAIN, DEFAULT_VAL, _references, _safe_name
    from prepare_global77_manifest import fit_complete_words, load_tokenizer, token_count
else:
    from tools.bundle_skyscript_mixed_pilot import CSV_FIELDS, _object_key, normalize
    from tools.extract_skyscript_subset import (
        _crc32_file,
        extract_members,
        index_selected_members,
    )
    from tools.install_skyscript_group_bundle import (
        DEFAULT_TRAIN,
        DEFAULT_VAL,
        _references,
        _safe_name,
    )
    from tools.prepare_global77_manifest import fit_complete_words, load_tokenizer, token_count

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("assets/data/manifests/skyscript_mixed_pilot_v1")
DEFAULT_DINOV3 = ROOT / "external/dinov3"
DEFAULT_BPE = ROOT / "assets/checkpoints/bpe_simple_vocab_16e6.txt.gz"
METADATA_NAMES = {"_metadata/samples.csv", "_metadata/audit.json", "_metadata/README.txt"}


def _read_bundle(
    bundle: Path, max_member_bytes: int
) -> tuple[list[dict[str, str]], dict[str, Any], set[str]]:
    if max_member_bytes <= 0:
        raise ValueError("max_member_bytes must be positive")
    with zipfile.ZipFile(bundle) as archive:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            name = _safe_name(info.filename)
            if name in members:
                raise ValueError(f"Duplicate bundle member: {name}")
            mode = stat.S_IFMT(info.external_attr >> 16)
            if info.is_dir() or mode not in {0, stat.S_IFREG}:
                raise ValueError(f"Bundle member must be a regular file: {name}")
            if info.file_size > max_member_bytes:
                raise ValueError(f"Bundle member exceeds max size: {name}")
            if name not in METADATA_NAMES:
                _safe_name(name, image=True)
            members[name] = info
        missing = METADATA_NAMES - members.keys()
        if missing:
            raise ValueError(f"Missing bundle metadata: {sorted(missing)}")
        audit = json.loads(archive.read("_metadata/audit.json"))
        if not isinstance(audit, dict) or audit.get("format_version") != 1:
            raise ValueError("Unsupported or invalid mixed-pilot audit metadata")
        archive.read("_metadata/README.txt").decode("utf-8")
        raw_csv = archive.read("_metadata/samples.csv").decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(raw_csv, newline=""))
    if reader.fieldnames != list(CSV_FIELDS):
        raise ValueError(f"samples.csv columns must be exactly {list(CSV_FIELDS)}")
    samples: list[dict[str, str]] = []
    paths: set[str] = set()
    for line_number, row in enumerate(reader, 2):
        if None in row:
            raise ValueError(f"samples.csv:{line_number}: extra CSV fields")
        relative = _safe_name(row.get("filepath") or "", image=True)
        if relative in paths:
            raise ValueError(f"Duplicate selected image path: {relative}")
        if row.get("split") not in {"train", "val"} or row.get("caption_kind") not in {
            "short",
            "detail",
        }:
            raise ValueError(f"samples.csv:{line_number}: invalid split or caption_kind")
        for field in ("short_caption", "detail_caption"):
            if not (row.get(field) or "").strip():
                raise ValueError(f"samples.csv:{line_number}: empty {field}")
        paths.add(relative)
        samples.append({key: row[key] for key in CSV_FIELDS})
    if not samples:
        raise ValueError("samples.csv is empty")
    image_members = set(members) - METADATA_NAMES
    if image_members != paths:
        raise ValueError("Bundle image members and samples.csv paths do not match exactly")
    return samples, audit, paths


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def install_mixed_pilot(
    *,
    bundle: Path,
    train_reference: Path = DEFAULT_TRAIN,
    val_reference: Path = DEFAULT_VAL,
    output_dir: Path = DEFAULT_OUTPUT,
    dinov3_repo: Path = DEFAULT_DINOV3,
    bpe_vocab: Path = DEFAULT_BPE,
    tokenizer: Any | None = None,
    context_length: int = 77,
    max_member_bytes: int = 100 * 1024 * 1024,
) -> dict[str, Any]:
    destination = output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output directory already exists: {destination}")
    if context_length < 3:
        raise ValueError("context_length must be at least 3")
    root, original_groups, _ = _references(train_reference, val_reference)
    samples, producer_audit, selected_paths = _read_bundle(bundle, max_member_bytes)
    seen_images: set[str] = set()
    seen_groups: set[str] = set()
    selected_objects: set[str] = set()
    untrimmed_captions: set[str] = set()
    for row in samples:
        path = row["filepath"]
        group = normalize(row["short_caption"])
        if group in seen_groups:
            raise ValueError(
                f"More than one image selected for original short-caption group: {group}"
            )
        seen_groups.add(group)
        if original_groups.get(group) != row["split"]:
            raise ValueError(
                "Selected group is absent from or assigned to the wrong original "
                f"{row['split']} manifest: {group}"
            )
        if path in seen_images:
            raise ValueError(f"Duplicate selected image: {path}")
        seen_images.add(path)
        obj = _object_key(path)
        if obj in selected_objects:
            raise ValueError(f"More than one selected image for OSM object: {obj}")
        selected_objects.add(obj)
        for form in (row["short_caption"], row["detail_caption"]):
            normalized = normalize(form)
            if normalized in untrimmed_captions:
                raise ValueError(
                    f"Untrimmed selected captions are not globally unique: {normalized}"
                )
            untrimmed_captions.add(normalized)
    expected = producer_audit.get("selected", {})
    observed = {
        split: dict(Counter(row["caption_kind"] for row in samples if row["split"] == split))
        for split in ("train", "val")
    }
    if expected and observed != expected:
        raise ValueError(
            f"Producer selection counts disagree with samples.csv: {observed} vs {expected}"
        )
    requested = producer_audit.get("requested")
    if not isinstance(requested, dict) or set(requested) != {"train", "val"}:
        raise ValueError("Producer audit is missing requested train/val quotas")
    for split in ("train", "val"):
        quota = requested[split]
        if not isinstance(quota, int) or quota <= 0 or quota % 2:
            raise ValueError(f"Producer audit has invalid even {split} quota: {quota!r}")
        if observed[split] != {"short": quota // 2, "detail": quota // 2}:
            raise ValueError(f"{split} selection must satisfy the fixed 1:1 caption ratio")
    if len(samples) != requested["train"] + requested["val"]:
        raise ValueError("samples.csv record count does not match requested quotas")
    if tokenizer is None:
        tokenizer = load_tokenizer(dinov3_repo, bpe_vocab)

    rows_by_split_kind: dict[tuple[str, str], list[dict[str, Any]]] = {}
    fitted_uniqueness: set[str] = set()
    trim_counts = Counter()
    trim_examples: list[dict[str, Any]] = []
    for row in samples:
        relative = row["filepath"]
        image = (root / Path(*PurePosixPath(relative).parts)).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"Selected image escapes original images root: {relative}")
        image_id = f"skyscript:{relative}"
        paired: dict[str, dict[str, Any]] = {}
        for kind, source_caption in (
            ("short", row["short_caption"]),
            ("detail", row["detail_caption"]),
        ):
            fitted, trimmed = fit_complete_words(tokenizer, source_caption, context_length)
            normalized_fitted = normalize(fitted)
            if normalized_fitted in fitted_uniqueness:
                raise ValueError(
                    f"77-token fitting creates a duplicate caption across selected rows: {fitted!r}"
                )
            fitted_uniqueness.add(normalized_fitted)
            trim_counts[kind] += int(trimmed)
            if trimmed and len(trim_examples) < 50:
                trim_examples.append(
                    {
                        "filepath": relative,
                        "caption_kind": kind,
                        "original_caption": source_caption,
                        "fitted_caption": fitted,
                        "original_tokens": token_count(tokenizer, source_caption),
                        "fitted_tokens": token_count(tokenizer, fitted),
                    }
                )
            paired[kind] = {
                "id": image_id,
                "image": str(image),
                "image_id": image_id,
                "caption": fitted,
                "original_caption": source_caption,
                "source": "SkyScript",
                "split": row["split"],
                "group_id": normalized_fitted,
                "csv_filepath": relative,
                "caption_kind": kind,
                "selected_caption_group": normalize(row["short_caption"]),
                "short_caption": row["short_caption"],
                "detail_caption": row["detail_caption"],
            }
            rows_by_split_kind.setdefault((row["split"], kind), []).append(paired[kind])
    by_key = {
        (row["split"], row["caption_kind"], row["csv_filepath"]): row
        for rows in rows_by_split_kind.values()
        for row in rows
    }
    mixed = {split: [] for split in ("train", "val")}
    for row in samples:
        kind = row["caption_kind"]
        relative = row["filepath"]
        candidate = by_key[(row["split"], kind, relative)]
        mixed[row["split"]].append(candidate)
    if any(not mixed[s] for s in mixed):
        raise ValueError("Mixed train and validation manifests must both be nonempty")

    matches, _ = index_selected_members([bundle], selected_paths, max_member_bytes=max_member_bytes)
    for relative, member in matches.items():
        image = (root / Path(*PurePosixPath(relative).parts)).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"Selected path escapes images root: {relative}")
        if image.exists() and not (
            image.is_file()
            and image.stat().st_size == member.file_size
            and _crc32_file(image) == member.crc
        ):
            raise FileExistsError(f"Refusing to overwrite non-matching extracted file: {image}")

    output_audit = {
        "status": "complete",
        "format_version": 1,
        "schema": "skyscript_mixed_pilot_v1",
        "bundle": str(bundle.resolve()),
        "output_dir": str(destination),
        "images_root": str(root),
        "original_split_verification": (
            "passed: every selected short group matches supplied original manifest split"
        ),
        "selected_groups": len(seen_groups),
        "selected_images": len(seen_images),
        "selected_kinds": observed,
        "mixed_ratio": {
            split: dict(Counter(row["caption_kind"] for row in mixed[split])) for split in mixed
        },
        "selected_group_provenance": producer_audit.get("selected_group_provenance", []),
        "caption_group_overlap": 0,
        "image_overlap": 0,
        "osm_object_cross_split": 0,
        "untrimmed_global_caption_uniqueness": "passed",
        "fitted_global_caption_uniqueness": "passed",
        "token_context": context_length,
        "complete_word_backoff": {
            "short": trim_counts["short"],
            "detail": trim_counts["detail"],
            "examples": trim_examples,
        },
        "producer_audit": producer_audit,
        "manifests": {
            "train_mixed.jsonl": len(mixed["train"]),
            "val_mixed.jsonl": len(mixed["val"]),
            "train_short.jsonl": len(rows_by_split_kind[("train", "short")]),
            "val_short.jsonl": len(rows_by_split_kind[("val", "short")]),
            "val_detail.jsonl": len(rows_by_split_kind[("val", "detail")]),
        },
        "splits": {
            "train": len(mixed["train"]),
            "val": len(mixed["val"]),
            "val_short": len(rows_by_split_kind[("val", "short")]),
            "val_detail": len(rows_by_split_kind[("val", "detail")]),
        },
    }
    extraction = extract_members(matches, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.part-", dir=destination.parent))
    try:
        _write_jsonl(temporary / "train_mixed.jsonl", mixed["train"])
        _write_jsonl(temporary / "val_mixed.jsonl", mixed["val"])
        _write_jsonl(temporary / "train_short.jsonl", rows_by_split_kind[("train", "short")])
        _write_jsonl(temporary / "val_short.jsonl", rows_by_split_kind[("val", "short")])
        _write_jsonl(temporary / "val_detail.jsonl", rows_by_split_kind[("val", "detail")])
        (temporary / "audit.json").write_text(
            json.dumps({**output_audit, **extraction}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (temporary / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(samples)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Output directory appeared during installation: {destination}")
        os.rename(temporary, destination)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**output_audit, **extraction}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--train-reference", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-reference", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3)
    parser.add_argument("--bpe-vocab", type=Path, default=DEFAULT_BPE)
    parser.add_argument("--context-length", type=int, default=77)
    parser.add_argument("--max-member-bytes", type=int, default=100 * 1024 * 1024)
    args = parser.parse_args()
    print(json.dumps(install_mixed_pilot(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
