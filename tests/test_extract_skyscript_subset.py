import csv
import json
import zipfile
from pathlib import Path

import pytest

from tools.extract_skyscript_subset import (
    extract_members,
    index_selected_members,
    select_rows,
    write_bundle,
    write_selection_csv,
)


def _write_csv(path: Path) -> None:
    rows = [
        {
            "filepath": "images1/warehouse-a.jpg",
            "title_raw": "Warehouse.",
            "title": "An aerial image. It shows: Warehouse.",
        },
        {
            "filepath": "images1/warehouse-b.jpg",
            "title_raw": "Warehouse.",
            "title": "An aerial image. It shows: Warehouse.",
        },
        {
            "filepath": "images2/school.jpg",
            "title_raw": "School building.",
            "title": "An aerial image. It shows: School building.",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_archives(root: Path) -> list[Path]:
    first = root / "images1.zip"
    second = root / "images2.zip"
    with zipfile.ZipFile(first, "w") as handle:
        handle.writestr("warehouse-a.jpg", b"warehouse-a")
        handle.writestr("warehouse-b.jpg", b"warehouse-b")
        handle.writestr("not-selected.jpg", b"unused")
    with zipfile.ZipFile(second, "w") as handle:
        handle.writestr("some-wrapper/images2/school.jpg", b"school")
    return [first, second]


def test_select_extract_and_bundle_only_requested_images(tmp_path) -> None:
    source_csv = tmp_path / "source.csv"
    _write_csv(source_csv)
    selected, fields, audit = select_rows(
        source_csv,
        path_field="filepath",
        group_field="title_raw",
        max_per_group=1,
        limit=2,
        seed=11,
    )
    selected_paths = {row.filepath for row in selected}
    assert len(selected_paths & {"images1/warehouse-a.jpg", "images1/warehouse-b.jpg"}) == 1
    assert "images2/school.jpg" in selected_paths
    assert audit["unique_normalized_groups"] == 2

    matches, archive_audit = index_selected_members(
        _write_archives(tmp_path), selected_paths, max_member_bytes=1024
    )
    assert set(matches) == selected_paths
    assert sum(item["selected_members"] for item in archive_audit["archives"]) == 2

    output_root = tmp_path / "extracted"
    result = extract_members(matches, output_root)
    assert result == {"extracted": 2, "reused_verified": 0}
    assert not (output_root / "images1/not-selected.jpg").exists()

    selection_csv = tmp_path / "selection.csv"
    assert write_selection_csv(selection_csv, selected, fields) == "written"
    assert write_selection_csv(selection_csv, selected, fields) == "reused_identical"
    bundle = tmp_path / "subset.zip"
    write_bundle(
        bundle,
        output_root=output_root,
        selected_paths=[row.filepath for row in selected],
        selection_csv=selection_csv,
    )
    with zipfile.ZipFile(bundle) as handle:
        assert set(handle.namelist()) == selected_paths | {"_metadata/selection.csv"}


def test_missing_selected_member_fails_before_extraction(tmp_path) -> None:
    archive = tmp_path / "images1.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("present.jpg", b"present")

    with pytest.raises(FileNotFoundError, match="not found"):
        index_selected_members(
            [archive], {"images1/missing.jpg"}, max_member_bytes=1024
        )
    assert not (tmp_path / "extracted").exists()


def test_existing_matching_file_is_verified_and_reused(tmp_path) -> None:
    archive = tmp_path / "images1.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("same.jpg", b"same-content")
    matches, _ = index_selected_members(
        [archive], {"images1/same.jpg"}, max_member_bytes=1024
    )
    root = tmp_path / "extracted"
    target = root / "images1/same.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"same-content")

    result = extract_members(matches, root)

    assert result == {"extracted": 0, "reused_verified": 1}


def test_selection_csv_is_parseable_after_write(tmp_path) -> None:
    source = tmp_path / "source.csv"
    _write_csv(source)
    selected, fields, _ = select_rows(
        source,
        path_field="filepath",
        group_field="title_raw",
        max_per_group=1,
        limit=2,
        seed=11,
    )
    output = tmp_path / "selected.csv"
    assert write_selection_csv(output, selected, fields) == "written"

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["title_raw"] for row in rows} == {"Warehouse.", "School building."}
    assert json.dumps(rows)


def test_all_csv_rows_mode_is_bounded_to_requested_archive_prefix(tmp_path) -> None:
    source = tmp_path / "source.csv"
    _write_csv(source)

    selected, _, audit = select_rows(
        source,
        path_field="filepath",
        group_field="title_raw",
        max_per_group=1,
        limit=1,
        seed=11,
        all_csv_rows=True,
        include_prefixes={"images1"},
    )

    assert {row.filepath for row in selected} == {
        "images1/warehouse-a.jpg",
        "images1/warehouse-b.jpg",
    }
    assert audit["global_selected_rows"] == 3
    assert audit["selected_rows_in_this_shard"] == 2
    assert audit["selection_strategy"] == "all_csv_rows_matching_include_prefix"


def test_all_csv_rows_requires_a_prefix(tmp_path) -> None:
    source = tmp_path / "source.csv"
    _write_csv(source)

    with pytest.raises(ValueError, match="requires at least one include prefix"):
        select_rows(
            source,
            path_field="filepath",
            group_field="title_raw",
            max_per_group=1,
            limit=1,
            seed=11,
            all_csv_rows=True,
        )


def test_grouped_selection_can_be_scoped_to_two_archive_prefixes(tmp_path) -> None:
    source = tmp_path / "source.csv"
    rows = [
        {
            "filepath": "images2/shared.jpg",
            "title_raw": "Shared caption.",
            "title": "An aerial image. It shows: Shared caption.",
        },
        {
            "filepath": "images3/shared.jpg",
            "title_raw": "Shared caption.",
            "title": "An aerial image. It shows: Shared caption.",
        },
        {
            "filepath": "images2/two-only.jpg",
            "title_raw": "Images two only.",
            "title": "An aerial image. It shows: Images two only.",
        },
        {
            "filepath": "images3/three-only.jpg",
            "title_raw": "Images three only.",
            "title": "An aerial image. It shows: Images three only.",
        },
        {
            "filepath": "images4/excluded.jpg",
            "title_raw": "Excluded caption.",
            "title": "An aerial image. It shows: Excluded caption.",
        },
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected, _, audit = select_rows(
        source,
        path_field="filepath",
        group_field="title_raw",
        max_per_group=1,
        limit=3,
        seed=11,
        selection_prefixes={"images2", "images3"},
    )

    assert len(selected) == 3
    assert {row.row["title_raw"] for row in selected} == {
        "Shared caption.",
        "Images two only.",
        "Images three only.",
    }
    assert all(row.filepath.startswith(("images2/", "images3/")) for row in selected)
    assert audit["source_rows"] == 5
    assert audit["rows_in_selection_scope"] == 4
    assert audit["unique_normalized_groups"] == 3
    assert audit["selection_prefixes"] == ["images2", "images3"]
