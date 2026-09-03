import hashlib
import json

from PIL import Image

from tools.filter_manifest_by_image_hash import filter_manifest


def test_filter_manifest_excludes_exact_image_hash_and_keeps_input(tmp_path) -> None:
    black = tmp_path / "black.png"
    red = tmp_path / "red.png"
    Image.new("RGB", (2, 2), "black").save(black)
    Image.new("RGB", (2, 2), "red").save(red)
    manifest = tmp_path / "source.jsonl"
    source = "\n".join(
        json.dumps(record)
        for record in (
            {"id": "black", "image": str(black), "caption": "black", "split": "train"},
            {"id": "red", "image": str(red), "caption": "red", "split": "train"},
        )
    ) + "\n"
    manifest.write_text(source, encoding="utf-8")
    black_hash = hashlib.sha256(black.read_bytes()).hexdigest()

    records, audit = filter_manifest(manifest, {black_hash})

    assert [record["id"] for record in records] == ["red"]
    assert audit["input_records"] == 2
    assert audit["removed_records"] == 1
    assert audit["removed_by_sha256"] == {black_hash: 1}
    assert manifest.read_text(encoding="utf-8") == source
