from collections import Counter

from tools.prepare_eurosat_manifest import EUROSAT_CLASSES, build_records, write_manifest


def test_build_eurosat_manifest_is_sorted_and_auditable(tmp_path) -> None:
    images_root = tmp_path / "eurosat"
    for index, label in enumerate(reversed(EUROSAT_CLASSES)):
        directory = images_root / label
        directory.mkdir(parents=True)
        (directory / f"{index:02d}.jpg").write_bytes(b"not-opened-by-manifest-builder")

    records = build_records(images_root, expected_total=len(EUROSAT_CLASSES))
    audit = write_manifest(
        records, tmp_path / "manifest.jsonl", tmp_path / "audit.json", images_root
    )

    assert [record["id"] for record in records] == sorted(record["id"] for record in records)
    assert audit["records"] == len(EUROSAT_CLASSES)
    assert Counter(record["label"] for record in records) == Counter(
        {label: 1 for label in EUROSAT_CLASSES}
    )
