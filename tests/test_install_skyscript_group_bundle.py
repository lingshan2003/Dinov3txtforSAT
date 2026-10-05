import csv
import io
import json
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

from tools.install_skyscript_group_bundle import install_group_bundle


def _fixture(tmp_path: Path) -> tuple[dict, list[dict], dict[str, bytes]]:
    root = tmp_path / "existing-images"
    rows = [
        {"filepath": "images2/train.jpg", "title_raw": "Airport."},
        {"filepath": "images3/train_extra.jpg", "title_raw": " AIRPORT. "},
        {"filepath": "images2/val.jpg", "title_raw": "Factory by river."},
        {"filepath": "images3/val_extra.jpg", "title_raw": "Factory  by river."},
        {"filepath": "images2/unknown.jpg", "title_raw": "Unknown."},
    ]
    images = {row["filepath"]: row["filepath"].encode() for row in rows}
    references = {}
    for split, caption in (("train", "Airport."), ("val", "Factory by river.")):
        image = root / f"images2/{split}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(images[f"images2/{split}.jpg"])
        reference = tmp_path / f"{split}.jsonl"
        reference.write_text(json.dumps({
            "id": f"original-{split}", "image": str(image), "caption": caption,
            "source": "SkyScript", "split": split,
        }) + "\n")
        references[f"{split}_reference"] = reference
    bundle = tmp_path / "candidates.zip"
    _write_bundle(bundle, rows, images)
    return dict(bundle=bundle, output_dir=tmp_path / "grouped", **references), rows, images


def _write_bundle(bundle: Path, rows: list[dict], images: dict[str, bytes], extra=None):
    csv_text = io.StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=["filepath", "title_raw"])
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("_metadata/candidates.csv", csv_text.getvalue())
        archive.writestr("_metadata/audit.json", "{}")
        archive.writestr("_metadata/README.txt", "Already polished top30 candidates.")
        for path, data in images.items():
            archive.writestr(path, data)
        for path, data in extra or []:
            archive.writestr(path, data)


def _assert_no_new_files(kwargs: dict):
    root = kwargs["train_reference"].parent / "existing-images"
    assert not (root / "images3").exists()
    assert not (root / "_metadata").exists()
    assert not kwargs["output_dir"].exists()


def test_install_keeps_original_root_extracts_only_heldout_groups_and_reuses(tmp_path):
    kwargs, _, images = _fixture(tmp_path)
    report = install_group_bundle(**kwargs)
    root = Path(report["images_root"])
    assert report["eligible_images"] == 4
    assert report["extracted"] == report["reused_verified"] == 2
    assert not (root / "images2/unknown.jpg").exists()
    assert (root / "images3/train_extra.jpg").read_bytes() == images["images3/train_extra.jpg"]
    assert (root / "_metadata/group_candidates_v1.csv").is_file()
    train = [json.loads(line) for line in
             (kwargs["output_dir"] / "train_grouped.jsonl").read_text().splitlines()]
    assert train[0]["image"] == str(root / "images2/train.jpg")
    assert train[0]["id"] == "original-train"
    assert train[1]["group_id"] == "airport."
    assert report["splits"]["val"]["records"] == 2
    assert kwargs["bundle"].exists()


def test_restart_reuses_previously_extracted_images_and_identical_csv(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    first = install_group_bundle(**kwargs)
    kwargs["output_dir"] = tmp_path / "another-output"
    second = install_group_bundle(**kwargs)
    assert second["extracted"] == 0
    assert second["reused_verified"] == first["eligible_images"]


def test_existing_output_fails_closed(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    kwargs["output_dir"].mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        install_group_bundle(**kwargs)


@pytest.mark.parametrize("field,value", [
    ("caption", " AIRPORT. "), ("id", "original-train"), ("source", "Other"),
    ("image", "images2/train.jpg"),
])
def test_reference_leakage_or_invalid_source_fails_before_extraction(tmp_path, field, value):
    kwargs, _, _ = _fixture(tmp_path)
    record = json.loads(kwargs["val_reference"].read_text())
    record[field] = (
        str(tmp_path / "existing-images" / value) if field == "image" else value
    )
    kwargs["val_reference"].write_text(json.dumps(record))
    with pytest.raises(ValueError):
        install_group_bundle(**kwargs)
    _assert_no_new_files(kwargs)


def test_inconsistent_original_image_roots_fail(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    record = json.loads(kwargs["val_reference"].read_text())
    record["image"] = str(tmp_path / "different-root/images2/val.jpg")
    kwargs["val_reference"].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="share one"):
        install_group_bundle(**kwargs)
    _assert_no_new_files(kwargs)


@pytest.mark.parametrize("remove_from", ["csv", "zip", "both"])
def test_missing_reference_representative_fails_before_extraction(tmp_path, remove_from):
    kwargs, rows, images = _fixture(tmp_path)
    if remove_from in {"csv", "both"}:
        rows = rows[1:]
    if remove_from in {"zip", "both"}:
        del images["images2/train.jpg"]
    _write_bundle(kwargs["bundle"], rows, images)
    with pytest.raises(ValueError, match="do not match|Missing reference"):
        install_group_bundle(**kwargs)
    _assert_no_new_files(kwargs)


def test_reference_caption_conflict_fails_before_extraction(tmp_path):
    kwargs, rows, images = _fixture(tmp_path)
    rows[0]["title_raw"] = "Factory by river."
    _write_bundle(kwargs["bundle"], rows, images)
    with pytest.raises(ValueError, match="conflicting CSV caption"):
        install_group_bundle(**kwargs)
    _assert_no_new_files(kwargs)


@pytest.mark.parametrize("path", ["../bad.jpg", "/images2/bad.jpg", "images2/../bad.jpg"])
def test_unsafe_bundle_members_rejected_before_extraction(tmp_path, path):
    kwargs, rows, images = _fixture(tmp_path)
    _write_bundle(kwargs["bundle"], rows, images, extra=[(path, b"bad")])
    with pytest.raises(ValueError, match="Unsafe"):
        install_group_bundle(**kwargs)
    _assert_no_new_files(kwargs)


def test_duplicate_members_rejected(tmp_path):
    kwargs, rows, images = _fixture(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _write_bundle(kwargs["bundle"], rows, images, extra=[("images2/train.jpg", b"bad")])
    with pytest.raises(ValueError, match="Duplicate bundle member"):
        install_group_bundle(**kwargs)
    _assert_no_new_files(kwargs)


def test_nonmatching_existing_image_is_not_overwritten(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    image = tmp_path / "existing-images/images2/val.jpg"
    image.write_bytes(b"different")
    with pytest.raises(FileExistsError, match="non-matching"):
        install_group_bundle(**kwargs)
    assert image.read_bytes() == b"different"
    _assert_no_new_files(kwargs)


def test_nonmatching_metadata_csv_fails_before_extraction(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    metadata = tmp_path / "existing-images/_metadata/group_candidates_v1.csv"
    metadata.parent.mkdir()
    metadata.write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="different candidate CSV"):
        install_group_bundle(**kwargs)
    assert metadata.read_text() == "do not overwrite"
    assert not (tmp_path / "existing-images/images3").exists()


def test_generated_id_collision_fails_before_extraction(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    record = json.loads(kwargs["val_reference"].read_text())
    record["id"] = "skyscript:images3/train_extra.jpg"
    kwargs["val_reference"].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="ID conflicts"):
        install_group_bundle(**kwargs)
    _assert_no_new_files(kwargs)


def test_reference_missing_on_disk_can_be_restored_from_valid_bundle(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    (tmp_path / "existing-images/images2/train.jpg").unlink()
    report = install_group_bundle(**kwargs)
    assert report["extracted"] == 3
    assert report["reused_verified"] == 1


def test_direct_cli(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    arguments = [sys.executable, "tools/install_skyscript_group_bundle.py"]
    for field, value in kwargs.items():
        arguments.extend([f"--{field.replace('_', '-')}", str(value)])
    result = subprocess.run(arguments, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == "complete"


def test_same_size_nonmatching_existing_image_is_rejected_by_crc(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    image = tmp_path / "existing-images/images2/val.jpg"
    image.write_bytes(b"x" * image.stat().st_size)
    with pytest.raises(FileExistsError, match="non-matching"):
        install_group_bundle(**kwargs)
    _assert_no_new_files(kwargs)


def test_interrupted_manifest_preparation_can_resume_installation(tmp_path, monkeypatch):
    kwargs, _, _ = _fixture(tmp_path)
    import tools.install_skyscript_group_bundle as installer

    original = installer.prepare_grouped_manifests

    def interrupted(*args, **kwargs):
        raise OSError("interrupted before manifest publication")

    monkeypatch.setattr(installer, "prepare_grouped_manifests", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        install_group_bundle(**kwargs)
    assert not kwargs["output_dir"].exists()
    monkeypatch.setattr(installer, "prepare_grouped_manifests", original)
    report = install_group_bundle(**kwargs)
    assert report["extracted"] == 0
    assert report["reused_verified"] == 4


def test_eligible_image_symlink_escape_fails_before_extraction(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    root = tmp_path / "existing-images"
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "images3").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes root"):
        install_group_bundle(**kwargs)
    assert not list(outside.iterdir())
    assert not kwargs["output_dir"].exists()


def test_metadata_symlink_escape_fails_before_extraction(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    root = tmp_path / "existing-images"
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "_metadata").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="metadata path escapes"):
        install_group_bundle(**kwargs)
    assert not (root / "images3").exists()
    assert not list(outside.iterdir())


@pytest.mark.parametrize("metadata", [
    "_metadata/candidates.csv", "_metadata/audit.json", "_metadata/README.txt",
])
def test_missing_required_metadata_fails_before_extraction(tmp_path, metadata):
    kwargs, _, _ = _fixture(tmp_path)
    with zipfile.ZipFile(kwargs["bundle"]) as archive:
        contents = {info.filename: archive.read(info) for info in archive.infolist()
                    if info.filename != metadata}
    with zipfile.ZipFile(kwargs["bundle"], "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match="Missing bundle metadata"):
        install_group_bundle(**kwargs)
    _assert_no_new_files(kwargs)
