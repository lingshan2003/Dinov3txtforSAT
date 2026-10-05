import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.prepare_skyscript_groups import prepare_grouped_manifests


def _setup(tmp_path: Path) -> dict:
    rows = [
        {"filepath": "images2/train.jpg", "title_raw": "Airport."},
        {"filepath": "images3/train_extra.jpg", "title_raw": "  AIRPORT.  "},
        {"filepath": "images2/val.jpg", "title_raw": "River by factory."},
        {"filepath": "images3/val_extra.jpg", "title_raw": "River  by factory."},
        {"filepath": "images2/unknown.jpg", "title_raw": "Unknown caption."},
        {"filepath": "images1/excluded.jpg", "title_raw": "Airport."},
    ]
    for row in rows[:4]:
        image = tmp_path / row["filepath"]
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(b"test-image")
    source = tmp_path / "source.csv"
    _write_csv(source, rows)
    for split, caption in (("train", "Airport."), ("val", "River by factory.")):
        record = {
            "id": f"original-{split}", "image": str(tmp_path / f"images2/{split}.jpg"),
            "caption": caption, "split": split, "source": "SkyScript", "custom": "keep",
        }
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(record) + "\n")
    return dict(
        csv_path=source, images_root=tmp_path,
        train_reference=tmp_path / "train.jsonl", val_reference=tmp_path / "val.jsonl",
        output_dir=tmp_path / "restored",
    )


def _write_csv(source: Path, rows: list[dict]) -> None:
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "title_raw"])
        writer.writeheader()
        writer.writerows(rows)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_restore_all_siblings_in_original_split_and_preserve_representatives(tmp_path):
    kwargs = _setup(tmp_path)
    audit = prepare_grouped_manifests(**kwargs)
    train = _rows(kwargs["output_dir"] / "train_grouped.jsonl")
    val = _rows(kwargs["output_dir"] / "val_grouped.jsonl")
    assert len(train) == len(val) == 2
    assert train[0]["id"] == "original-train"
    assert train[0]["custom"] == "keep"
    assert train[0]["reference_representative"] is True
    assert train[1]["id"] == "skyscript:images3/train_extra.jpg"
    assert all(record["caption"] == "Airport." for record in train)
    assert all(record["group_id"] == "airport." for record in train)
    assert all(record["split"] == "val" for record in val)
    assert {record["image"] for record in train}.isdisjoint(record["image"] for record in val)
    assert audit["row_counts"] == {
        "csv_rows": 6, "eligible_rows": 4, "excluded_prefix_rows": 1,
        "excluded_unknown_group_rows": 1, "missing_eligible_rows": 0,
    }
    assert audit["splits"]["train"]["group_size_histogram"] == {2: 1}
    assert not any("sha256" in key for key in audit)


def test_cli_runs_directly(tmp_path):
    kwargs = _setup(tmp_path)
    result = subprocess.run(
        [sys.executable, "tools/prepare_skyscript_groups.py", "--csv", str(kwargs["csv_path"]),
         "--images-root", str(tmp_path), "--train-reference", str(kwargs["train_reference"]),
         "--val-reference", str(kwargs["val_reference"]),
         "--output-dir", str(kwargs["output_dir"])],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["reference_representatives_retained"] == 2


def test_missing_eligible_images_fail_closed_unless_explicitly_allowed(tmp_path):
    kwargs = _setup(tmp_path)
    (tmp_path / "images3/train_extra.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="eligible images are missing"):
        prepare_grouped_manifests(**kwargs)
    assert not kwargs["output_dir"].exists()
    audit = prepare_grouped_manifests(**kwargs, allow_missing=True)
    assert audit["splits"]["train"]["singleton_groups"] == 1
    assert audit["row_counts"]["missing_eligible_rows"] == 1


@pytest.mark.parametrize("allow_missing", [False, True])
def test_reference_image_cannot_be_omitted(tmp_path, allow_missing):
    kwargs = _setup(tmp_path)
    (tmp_path / "images2/train.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="Reference representative"):
        prepare_grouped_manifests(**kwargs, allow_missing=allow_missing)
    assert not kwargs["output_dir"].exists()


def test_reference_present_on_disk_but_absent_from_csv_fails(tmp_path):
    kwargs = _setup(tmp_path)
    rows = list(csv.DictReader(kwargs["csv_path"].open()))
    _write_csv(kwargs["csv_path"], rows[1:])
    with pytest.raises(ValueError, match="representatives are absent"):
        prepare_grouped_manifests(**kwargs, allow_missing=True)


@pytest.mark.parametrize("field,value,match", [
    ("caption", " AIRPORT. ", "caption groups overlap"),
    ("image", "images2/train.jpg", "images or IDs overlap"),
    ("id", "original-train", "images or IDs overlap"),
])
def test_reference_leakage_fails_before_publication(tmp_path, field, value, match):
    kwargs = _setup(tmp_path)
    record = _rows(kwargs["val_reference"])[0]
    record[field] = str(tmp_path / value) if field == "image" else value
    kwargs["val_reference"].write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match=match):
        prepare_grouped_manifests(**kwargs)
    assert not kwargs["output_dir"].exists()


@pytest.mark.parametrize("filepath", ["../outside.jpg", "/outside.jpg", "images2/../x.jpg"])
def test_unsafe_csv_paths_rejected_even_for_unknown_groups(tmp_path, filepath):
    kwargs = _setup(tmp_path)
    with kwargs["csv_path"].open("a") as handle:
        handle.write(f"{filepath},Unknown group\n")
    with pytest.raises(ValueError, match="Unsafe"):
        prepare_grouped_manifests(**kwargs)
    assert not kwargs["output_dir"].exists()


def test_symlink_escape_and_aliased_files_rejected(tmp_path):
    kwargs = _setup(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.jpg"
    outside.write_bytes(b"outside")
    (tmp_path / "images2/link.jpg").symlink_to(outside)
    with kwargs["csv_path"].open("a") as handle:
        handle.write("images2/link.jpg,Airport.\n")
    with pytest.raises(ValueError, match="escapes images root"):
        prepare_grouped_manifests(**kwargs)
    outside.unlink()


@pytest.mark.parametrize("caption", ["Airport.", "River by factory.", "Unknown."])
def test_duplicate_or_conflicting_csv_image_rejected(tmp_path, caption):
    kwargs = _setup(tmp_path)
    with kwargs["csv_path"].open("a") as handle:
        handle.write(f"images2/train.jpg,{caption}\n")
    with pytest.raises(ValueError, match="Duplicate or aliased CSV image"):
        prepare_grouped_manifests(**kwargs)


def test_changed_reference_caption_membership_fails(tmp_path):
    kwargs = _setup(tmp_path)
    rows = list(csv.DictReader(kwargs["csv_path"].open()))
    rows[0]["title_raw"] = "River by factory."
    _write_csv(kwargs["csv_path"], rows)
    with pytest.raises(ValueError, match="conflicting CSV caption"):
        prepare_grouped_manifests(**kwargs)


def test_no_overwrite(tmp_path):
    kwargs = _setup(tmp_path)
    kwargs["output_dir"].mkdir()
    marker = kwargs["output_dir"] / "keep"
    marker.write_text("untouched")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_grouped_manifests(**kwargs)
    assert marker.read_text() == "untouched"


def test_generated_id_cannot_collide_with_reference_id(tmp_path):
    kwargs = _setup(tmp_path)
    record = _rows(kwargs["val_reference"])[0]
    record["id"] = "skyscript:images3/train_extra.jpg"
    kwargs["val_reference"].write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="conflicting restored ID"):
        prepare_grouped_manifests(**kwargs)


def test_failed_directory_publication_cleans_temporary_files(tmp_path, monkeypatch):
    kwargs = _setup(tmp_path)

    def fail_publication(*args):
        raise OSError("publication failed")

    monkeypatch.setattr("tools.prepare_skyscript_groups.os.rename", fail_publication)
    with pytest.raises(OSError, match="publication failed"):
        prepare_grouped_manifests(**kwargs)
    assert not kwargs["output_dir"].exists()
    assert not list(tmp_path.glob(".restored.part-*"))


def test_prefix_override_is_explicit_and_audited(tmp_path):
    kwargs = _setup(tmp_path)
    path = tmp_path / "images4"
    (tmp_path / "images2").rename(path)
    for split in ("train", "val"):
        record = _rows(kwargs[f"{split}_reference"])[0]
        record["image"] = str(path / f"{split}.jpg")
        kwargs[f"{split}_reference"].write_text(json.dumps(record) + "\n")
    rows = list(csv.DictReader(kwargs["csv_path"].open()))
    for row in rows:
        row["filepath"] = row["filepath"].replace("images2/", "images4/")
    _write_csv(kwargs["csv_path"], rows)
    with pytest.raises(ValueError, match="unselected prefix"):
        prepare_grouped_manifests(**kwargs)
    audit = prepare_grouped_manifests(**kwargs, allow_prefixes=("images4", "images3"))
    assert audit["allowed_prefixes"] == ["images3", "images4"]


@pytest.mark.parametrize("source", ["OtherDataset", "", None])
def test_reference_source_must_be_skyscript(tmp_path, source):
    kwargs = _setup(tmp_path)
    record = _rows(kwargs["train_reference"])[0]
    record["source"] = source
    kwargs["train_reference"].write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="expected source='SkyScript'"):
        prepare_grouped_manifests(**kwargs)
    assert not kwargs["output_dir"].exists()
