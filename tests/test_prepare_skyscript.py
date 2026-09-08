import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.prepare_skyscript import caption_from_row, prepare_manifest


def _write_fixture(root: Path, rows: list[dict[str, str]]) -> Path:
    for row in rows:
        path = root / row["filepath"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(20, 40, 60)).save(path)
    csv_path = root / "top30.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "title_raw", "title"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_contextual_summary_does_not_collapse_to_first_sentence() -> None:
    row = {
        "title": "An aerial image. It shows: Hospital providing healthcare services.",
    }
    assert caption_from_row(row, "title", "contextual-summary") == row["title"]
    assert caption_from_row(row, "title", "summary") == (
        "Hospital providing healthcare services."
    )


def test_auto_mode_supports_both_skyscript_csv_variants() -> None:
    polished = "An aerial image. It shows: Warehouse."
    filtered = "a satellite image of building of warehouse"
    assert caption_from_row({"title": polished}, "title", "auto") == polished
    assert caption_from_row({"title": filtered}, "title", "auto") == filtered


def test_prepare_manifest_is_canonical_deterministic_and_audited(tmp_path) -> None:
    rows = [
        {
            "filepath": "images2/a222016617_US_20.jpg",
            "title_raw": "Commercial building used for cars.",
            "title": "An aerial image. It shows: Commercial building used for cars.",
        },
        {
            "filepath": "images3/a516241327_US_18.jpg",
            "title_raw": "Hospital providing healthcare services.",
            "title": "An aerial image. It shows: Hospital providing healthcare services.",
        },
        {
            "filepath": "images6/a975696811_CH_21.jpg",
            "title_raw": "Warehouse.",
            "title": "An aerial image. It shows: Warehouse.",
        },
    ]
    csv_path = _write_fixture(tmp_path, rows)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    kwargs = {
        "csv_path": csv_path,
        "images_root": tmp_path,
        "split": "train",
        "caption_field": "title",
        "caption_mode": "contextual-summary",
        "limit": 2,
        "seed": 11,
        "max_per_caption": None,
        "allow_missing": False,
    }

    audit = prepare_manifest(output=first, **kwargs)
    prepare_manifest(output=second, **kwargs)
    records = [json.loads(line) for line in first.read_text().splitlines()]

    assert first.read_bytes() == second.read_bytes()
    assert len(records) == 2
    assert all(set(record) == {"id", "image", "caption", "split", "source"} for record in records)
    assert all(record["id"].startswith("skyscript:images") for record in records)
    assert all(record["caption"].startswith("An aerial image. It shows:") for record in records)
    assert audit["selected_rows"] == 2
    assert audit["country_counts_before_sampling"] == {"US": 2, "CH": 1}
    assert audit["zoom_counts_before_sampling"] == {"20": 1, "18": 1, "21": 1}


def test_missing_images_fail_without_publishing_partial_manifest(tmp_path) -> None:
    row = {
        "filepath": "images2/missing_US_20.jpg",
        "title_raw": "A missing image.",
        "title": "An aerial image. It shows: A missing image.",
    }
    csv_path = tmp_path / "top30.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    output = tmp_path / "manifest.jsonl"

    with pytest.raises(FileNotFoundError, match="images are missing"):
        prepare_manifest(
            csv_path=csv_path,
            images_root=tmp_path,
            split="train",
            output=output,
            caption_field="title",
            caption_mode="contextual-summary",
            limit=None,
            seed=11,
            max_per_caption=None,
            allow_missing=False,
        )

    assert not output.exists()
    assert not output.with_suffix(".jsonl.part").exists()


def test_caption_cap_limits_exact_repetitions(tmp_path) -> None:
    rows = [
        {
            "filepath": f"images1/item{index}_US_20.jpg",
            "title_raw": "Warehouse.",
            "title": "An aerial image. It shows: Warehouse.",
        }
        for index in range(4)
    ]
    csv_path = _write_fixture(tmp_path, rows)
    output = tmp_path / "manifest.jsonl"

    audit = prepare_manifest(
        csv_path=csv_path,
        images_root=tmp_path,
        split="train",
        output=output,
        caption_field="title",
        caption_mode="contextual-summary",
        limit=2,
        seed=11,
        max_per_caption=2,
        allow_missing=False,
    )

    assert len(output.read_text().splitlines()) == 2
    assert audit["caption_statistics_before_sampling"]["duplicate_records"] == 3
