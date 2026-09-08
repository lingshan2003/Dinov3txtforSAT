import csv

import pytest

from tools.split_skyscript_selection import read_unique_rows, split_rows


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "title_raw", "title"])
        writer.writeheader()
        writer.writerows(rows)


def test_deterministic_split_is_group_disjoint_and_preserves_input_order(tmp_path) -> None:
    source = tmp_path / "selection.csv"
    rows = [
        {
            "filepath": f"images{2 + index % 2}/{index}.jpg",
            "title_raw": f"Caption {index}.",
            "title": f"An aerial image. It shows: Caption {index}.",
        }
        for index in range(10)
    ]
    _write_rows(source, rows)
    loaded, _ = read_unique_rows(source, group_field="title_raw", path_field="filepath")

    train_a, val_a = split_rows(loaded, group_field="title_raw", val_count=2, seed=23)
    train_b, val_b = split_rows(loaded, group_field="title_raw", val_count=2, seed=23)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 8
    assert len(val_a) == 2
    assert {row["title_raw"] for row in train_a}.isdisjoint(
        {row["title_raw"] for row in val_a}
    )
    assert [rows.index(row) for row in train_a] == sorted(rows.index(row) for row in train_a)
    assert [rows.index(row) for row in val_a] == sorted(rows.index(row) for row in val_a)


def test_duplicate_normalized_caption_is_rejected(tmp_path) -> None:
    source = tmp_path / "selection.csv"
    rows = [
        {
            "filepath": "images2/a.jpg",
            "title_raw": "Warehouse.",
            "title": "An aerial image. It shows: Warehouse.",
        },
        {
            "filepath": "images3/b.jpg",
            "title_raw": "  WAREHOUSE. ",
            "title": "An aerial image. It shows: Warehouse.",
        },
    ]
    _write_rows(source, rows)

    with pytest.raises(ValueError, match="duplicate normalized"):
        read_unique_rows(source, group_field="title_raw", path_field="filepath")
