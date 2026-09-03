import json

from PIL import Image

from tools.prepare_chatearthnet import caption_text, prepare_manifest


def test_caption_text_accepts_official_singleton_list() -> None:
    assert caption_text(["  A satellite   image. "]) == "A satellite image."


def test_prepare_manifest_accepts_object_root_and_nested_images(tmp_path) -> None:
    images_root = tmp_path / "extracted" / "s2_rgb_images"
    image_path = images_root / "tile_1" / "patch.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image_path)
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "filename": "patch.png",
                        "captions": ["  north of water  "],
                        "split": "train",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records, audit = prepare_manifest(
        annotations=annotations,
        images_root=images_root,
        split=None,
        limit=None,
        seed=11,
        allow_missing=False,
    )

    assert records == [
        {
            "id": "chatearthnet:patch.png",
            "image": str(image_path.resolve()),
            "caption": "north of water",
            "split": "train",
            "source": "ChatEarthNet",
        }
    ]
    assert audit["indexed_images"] == 1
    assert audit["missing_count"] == 0
