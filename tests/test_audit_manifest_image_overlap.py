from __future__ import annotations

import json
import shutil

from PIL import Image

from tools.audit_manifest_image_overlap import audit_overlap


def _manifest(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_overlap_audit_detects_file_and_decoded_pixel_identity(tmp_path) -> None:
    first = tmp_path / "first.png"
    file_copy = tmp_path / "copy.png"
    pixel_copy = tmp_path / "pixel-copy.bmp"
    distinct = tmp_path / "distinct.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(first)
    shutil.copyfile(first, file_copy)
    Image.open(first).save(pixel_copy)
    Image.new("RGB", (8, 8), color=(30, 20, 10)).save(distinct)
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _manifest(left, [{"id": "left", "image": str(first)}])
    _manifest(
        right,
        [
            {"id": "file-copy", "image": str(file_copy)},
            {"id": "pixel-copy", "image": str(pixel_copy)},
            {"id": "distinct", "image": str(distinct)},
        ],
    )

    report = audit_overlap(left, right)

    assert report["status"] == "overlap_detected"
    assert report["exact_file_overlap_count"] == 1
    assert report["exact_decoded_pixel_overlap_count"] == 2


def test_overlap_audit_deduplicates_repeated_caption_images(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(first)
    Image.new("RGB", (8, 8), color=(30, 20, 10)).save(second)
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _manifest(left, [{"id": "left", "image": str(first)}])
    _manifest(
        right,
        [
            {"id": "caption-1", "image_id": "image-1", "image": str(second)},
            {"id": "caption-2", "image_id": "image-1", "image": str(second)},
        ],
    )

    report = audit_overlap(left, right)

    assert report["status"] == "clear"
    assert report["right_manifest"]["unique_images"] == 1
