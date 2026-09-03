import json

from PIL import Image

from tools.verify_server_assets import build_report


def test_build_report_checks_annotation_image_references(tmp_path) -> None:
    checkpoint = (
        tmp_path / "assets" / "checkpoints" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    data_root = tmp_path / "assets" / "data" / "raw" / "chatearthnet"
    image_path = data_root / "extracted" / "rgb" / "tile.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image_path)
    (data_root / "captions.json").write_text(
        json.dumps([{"image": "tile.jpg", "caption": "a tile"}]), encoding="utf-8"
    )

    report = build_report(tmp_path, verify_images=True)

    checkpoints = {record["name"]: record for record in report["checkpoints"]}
    assert checkpoints[checkpoint.name]["exists"]
    references = report["datasets"]["chatearthnet"]["annotation_candidates"][0][
        "image_reference_inventory"
    ]
    assert references["references"] == 1
    assert references["missing_references"] == 0
    assert references["ambiguous_references"] == 0
