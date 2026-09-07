import json

import pytest
import torch

from dinotxt_rs.evaluation.retrieval import load_rsicd_records, retrieval_metrics


def test_retrieval_metrics_handles_multiple_captions_per_image() -> None:
    image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text_features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    metrics = retrieval_metrics(
        image_features,
        text_features,
        text_to_image=[0, 0, 1],
        device=torch.device("cpu"),
        chunk_size=1,
    )

    assert metrics["image_to_text"]["r1"] == pytest.approx(1.0)
    assert metrics["text_to_image"]["r1"] == pytest.approx(1.0)
    assert metrics["mean_recall"] == pytest.approx(1.0)


def test_load_rsicd_records_accepts_explicit_val_split(tmp_path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "rsicd:1:caption00",
                "image_id": "rsicd:1",
                "image": str(image),
                "caption": "a caption",
                "split": "val",
                "source": "RSICD",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    images, captions = load_rsicd_records(manifest, expected_split="val")

    assert images == [{"id": "rsicd:1", "image": str(image)}]
    assert captions[0]["split"] == "val"
    with pytest.raises(ValueError, match="split='test'"):
        load_rsicd_records(manifest)
