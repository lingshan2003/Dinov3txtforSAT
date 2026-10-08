import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tools.bundle_skyscript_mixed_pilot import (
    bundle_mixed_pilot,
    reconstruct_group_split,
    select_samples,
)
from tools.install_skyscript_mixed_pilot import install_mixed_pilot


class WordTokenizer:
    def encode(self, text):
        return text.split()


def _rows(n=8):
    short = []
    detail = []
    for index in range(n):
        path = f"images{2 + (index % 2)}/a{100000 + index}_US_{10 + index}.jpg"
        caption = f"Building zone{index} sector"
        long = (
            f"Satellite view zone{index} reveals a warehouse beside a service road "
            f"and nearby storage tanks across the industrial district today clearly."
        )
        short.append({"filepath": path, "title_raw": caption, "title": caption})
        detail.append({"filepath": path, "title_multi_objects": long})
    return short, detail


def _select(short, detail, *, train=4, val=2, original_val=2, seed=11):
    return select_samples(
        short,
        detail,
        train_count=train,
        val_count=val,
        split_val_count=original_val,
        split_seed=23,
        seed=seed,
    )


def _write_csv(path: Path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bundle_inputs(tmp_path, n=8):
    short, detail = _rows(n)
    short_csv, detail_csv = tmp_path / "short.csv", tmp_path / "detail.csv"
    _write_csv(short_csv, ["filepath", "title_raw", "title"], short)
    _write_csv(detail_csv, ["filepath", "title_multi_objects"], detail)
    archives = [tmp_path / "images2.zip", tmp_path / "images3.zip"]
    by_prefix = {2: [], 3: []}
    for index, row in enumerate(short):
        prefix = int(row["filepath"].split("/")[0][-1])
        by_prefix[prefix].append((row["filepath"], f"original-image-{index}".encode()))
    for prefix, archive in zip((2, 3), archives, strict=True):
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for relative, content in by_prefix[prefix]:
                handle.writestr(relative, content)
    return short_csv, detail_csv, archives, short, detail


def _references(tmp_path, rows, val_count=2):
    root = tmp_path / "original-images"
    split = reconstruct_group_split(rows, val_count=val_count, seed=23)
    paths = {row["title_raw"].casefold(): row["filepath"] for row in rows}
    refs = {}
    for kind in ("train", "val"):
        records = []
        for index, (group, membership) in enumerate(split.items()):
            if membership != kind:
                continue
            relative = paths[group]
            records.append(
                {
                    "id": f"original-{kind}-{index}",
                    "image": str(root / relative),
                    "caption": next(
                        row["title_raw"] for row in rows if row["title_raw"].casefold() == group
                    ),
                    "source": "SkyScript",
                    "split": kind,
                }
            )
        path = tmp_path / f"{kind}.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        refs[f"{kind}_reference"] = path
    return refs, root


def test_reconstructs_original_group_split_and_selects_exact_fixed_ratio():
    short, detail = _rows()
    first, audit = _select(short, detail)
    second, _ = _select(short, detail)
    assert [(r["filepath"], r["caption_kind"]) for r in first] == [
        (r["filepath"], r["caption_kind"]) for r in second
    ]
    assert audit["original_split"]["all_groups"] == len(short)
    assert audit["selected"] == {
        "train": {"short": 2, "detail": 2},
        "val": {"short": 1, "detail": 1},
    }
    assert len({row["short_group"] for row in first}) == len(first)
    assert len({Path(row["filepath"]).stem.split("_")[0] for row in first}) == len(first)
    captions = [
        " ".join(text.split()).casefold()
        for row in first
        for text in (row["short_caption"], row["detail_caption"])
    ]
    assert len(set(captions)) == 2 * len(first)


def test_quota_shortage_is_an_error_instead_of_silent_shrink():
    short, detail = _rows(5)
    with pytest.raises(ValueError, match="Insufficient eligible"):
        _select(short, detail, train=4, val=2, original_val=2)


def test_full_pool_osm_objects_spanning_original_splits_are_excluded():
    short, detail = _rows(8)
    membership = reconstruct_group_split(short, val_count=2, seed=23)
    train_group = next(group for group, split in membership.items() if split == "train")
    val_group = next(group for group, split in membership.items() if split == "val")
    train_row = next(row for row in short if row["title_raw"].casefold() == train_group)
    val_row = next(row for row in short if row["title_raw"].casefold() == val_group)
    shared_obj = "a999999"
    train_row["filepath"] = f"images2/{shared_obj}_US_10.jpg"
    val_row["filepath"] = f"images3/{shared_obj}_US_11.jpg"
    detail.extend(
        [
            {
                "filepath": train_row["filepath"],
                "title_multi_objects": next(
                    row["title_multi_objects"] for row in detail if "100000" in row["filepath"]
                ),
            },
            {
                "filepath": val_row["filepath"],
                "title_multi_objects": next(
                    row["title_multi_objects"] for row in detail if "100001" in row["filepath"]
                ),
            },
        ]
    )
    # Replace the old paths so only the shared cross-split object versions remain.
    detail = [
        row
        for row in detail
        if row["filepath"]
        not in {
            f"images{2 + (i % 2)}/a{100000 + i}_US_{10 + i}.jpg"
            for i in range(8)
            if i not in {short.index(train_row), short.index(val_row)}
        }
    ]
    with pytest.raises(ValueError, match="Insufficient eligible") as exc:
        _select(short, detail, train=4, val=2, original_val=2)
    assert "object_in_both_original_splits" in str(exc.value) or "eligible_groups" in str(exc.value)


def test_bundle_streams_only_selected_original_bytes_and_dry_run_is_read_only(tmp_path):
    short_csv, detail_csv, archives, short, _ = _bundle_inputs(tmp_path)
    output = tmp_path / "pilot.zip"
    report = bundle_mixed_pilot(
        csv_path=short_csv,
        detail_csv_path=detail_csv,
        archives=archives,
        output=output,
        train_count=4,
        val_count=2,
        split_val_count=2,
        split_seed=23,
        seed=11,
        dry_run=True,
    )
    assert report["images"] == 6
    assert not output.exists()
    report = bundle_mixed_pilot(
        csv_path=short_csv,
        detail_csv_path=detail_csv,
        archives=archives,
        output=output,
        train_count=4,
        val_count=2,
        split_val_count=2,
        split_seed=23,
        seed=11,
    )
    with zipfile.ZipFile(output) as handle:
        assert handle.testzip() is None
        image_names = {name for name in handle.namelist() if not name.startswith("_metadata/")}
        assert len(image_names) == 6
        assert all(handle.getinfo(name).compress_type == zipfile.ZIP_STORED for name in image_names)
        samples = list(csv.DictReader(handle.read("_metadata/samples.csv").decode().splitlines()))
        assert len(samples) == 6
        assert report["selected"] == {
            "train": {"short": 2, "detail": 2},
            "val": {"short": 1, "detail": 1},
        }


def test_installer_validates_original_split_fits_captions_and_emits_paired_validation(tmp_path):
    short_csv, detail_csv, archives, short_rows, _ = _bundle_inputs(tmp_path)
    bundle = tmp_path / "pilot.zip"
    bundle_mixed_pilot(
        csv_path=short_csv,
        detail_csv_path=detail_csv,
        archives=archives,
        output=bundle,
        train_count=4,
        val_count=2,
        split_val_count=2,
        split_seed=23,
        seed=11,
    )
    refs, root = _references(tmp_path, short_rows)
    output = tmp_path / "installed"
    report = install_mixed_pilot(
        bundle=bundle, **refs, output_dir=output, tokenizer=WordTokenizer(), context_length=8
    )
    assert report["schema"] == "skyscript_mixed_pilot_v1"
    assert report["splits"] == {"train": 4, "val": 2, "val_short": 2, "val_detail": 2}
    assert report["mixed_ratio"] == {
        "train": {"short": 2, "detail": 2},
        "val": {"short": 1, "detail": 1},
    }
    val_short = [json.loads(line) for line in (output / "val_short.jsonl").read_text().splitlines()]
    val_detail = [
        json.loads(line) for line in (output / "val_detail.jsonl").read_text().splitlines()
    ]
    val_mixed = [json.loads(line) for line in (output / "val_mixed.jsonl").read_text().splitlines()]
    assert {row["image_id"] for row in val_short} == {row["image_id"] for row in val_detail}
    assert {row["id"] for row in val_short} == {row["id"] for row in val_detail}
    assert len(val_mixed) == 2
    assert all(
        row["group_id"] == " ".join(row["caption"].split()).casefold()
        for row in val_short + val_detail
    )
    assert report["complete_word_backoff"]["detail"] > 0
    assert all((root / row["csv_filepath"]).is_file() for row in val_short + val_detail)


def test_installer_refuses_reference_split_mismatch_before_extracting(tmp_path):
    short_csv, detail_csv, archives, short_rows, _ = _bundle_inputs(tmp_path)
    bundle = tmp_path / "pilot.zip"
    bundle_mixed_pilot(
        csv_path=short_csv,
        detail_csv_path=detail_csv,
        archives=archives,
        output=bundle,
        train_count=4,
        val_count=2,
        split_val_count=2,
        split_seed=23,
        seed=11,
    )
    refs, root = _references(tmp_path, short_rows)
    record = json.loads(refs["train_reference"].read_text().splitlines()[0])
    record["split"] = "val"
    refs["train_reference"].write_text(
        json.dumps(record)
        + "\n"
        + "\n".join(refs["train_reference"].read_text().splitlines()[1:])
        + "\n"
    )
    with pytest.raises(ValueError, match="invalid reference source or split|assigned to the wrong"):
        install_mixed_pilot(
            bundle=bundle,
            **refs,
            output_dir=tmp_path / "installed",
            tokenizer=WordTokenizer(),
            context_length=8,
        )
    assert not (root / "images2").exists()
    assert not (tmp_path / "installed").exists()


def test_fitted_caption_collision_fails_before_extraction_or_publish(tmp_path):
    short_csv, detail_csv, archives, short_rows, _ = _bundle_inputs(tmp_path)
    bundle = tmp_path / "pilot.zip"
    bundle_mixed_pilot(
        csv_path=short_csv,
        detail_csv_path=detail_csv,
        archives=archives,
        output=bundle,
        train_count=4,
        val_count=2,
        split_val_count=2,
        split_seed=23,
        seed=11,
    )
    refs, root = _references(tmp_path, short_rows)
    with pytest.raises(ValueError, match="fitting creates a duplicate"):
        install_mixed_pilot(
            bundle=bundle,
            **refs,
            output_dir=tmp_path / "installed",
            tokenizer=WordTokenizer(),
            context_length=4,
        )
    assert not (root / "images2").exists()
    assert not (tmp_path / "installed").exists()


@pytest.mark.parametrize("member", ["../escape.jpg", "/absolute.jpg", "images4/bad.jpg"])
def test_installer_rejects_unsafe_bundle_members(tmp_path, member):
    bundle = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "_metadata/samples.csv", "filepath,short_caption,detail_caption,split,caption_kind\n"
        )
        archive.writestr("_metadata/audit.json", json.dumps({"format_version": 1}))
        archive.writestr("_metadata/README.txt", "ok")
        archive.writestr(member, b"bad")
    with pytest.raises(ValueError):
        from tools.install_skyscript_mixed_pilot import _read_bundle

        _read_bundle(bundle, 1000)


def test_both_command_line_interfaces_load(tmp_path):
    for script in (
        "tools/bundle_skyscript_mixed_pilot.py",
        "tools/install_skyscript_mixed_pilot.py",
    ):
        result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
