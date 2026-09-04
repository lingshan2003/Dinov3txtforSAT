import json

from tools.prepare_rsicd_retrieval_manifest import build_records, write_manifest


def test_prepare_rsicd_test_manifest_expands_all_captions(tmp_path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"one")
    (images / "two.jpg").write_bytes(b"two")
    annotations = tmp_path / "dataset_rsicd.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "split": "test",
                        "imgid": 1,
                        "filename": "one.jpg",
                        "sentences": [{"raw": "one caption"}, {"tokens": ["another", "one"]}],
                    },
                    {
                        "split": "train",
                        "imgid": 2,
                        "filename": "two.jpg",
                        "sentences": [{"raw": "training caption"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    records, audit = build_records(annotations, images, "test")
    persisted = write_manifest(records, audit, tmp_path / "manifest.jsonl", tmp_path / "audit.json")

    assert [record["caption"] for record in records] == ["one caption", "another one"]
    assert {record["image_id"] for record in records} == {"rsicd:1"}
    assert persisted["images"] == 1
    assert persisted["captions"] == 2
    assert persisted["caption_count_histogram"] == {2: 1}
