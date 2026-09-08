import json
from types import SimpleNamespace

import pytest
import torch

from dinotxt_rs.evaluation.retrieval import (
    evaluate_paired_retrieval,
    load_paired_records,
    load_rsicd_records,
    retrieval_metrics,
)


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


def _write_paired_manifest(tmp_path, captions: tuple[str, ...] = ("forest", "bridge")):
    records = []
    for index, caption in enumerate(captions):
        image = tmp_path / f"image-{index}.jpg"
        image.write_bytes(b"image")
        records.append(
            {
                "id": f"skyscript:{index}",
                "image": str(image),
                "caption": caption,
                "split": "val",
                "source": "SkyScript",
            }
        )
    manifest = tmp_path / "paired.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return manifest, records


def test_load_paired_records_preserves_real_source_and_order(tmp_path) -> None:
    manifest, expected = _write_paired_manifest(tmp_path)

    records = load_paired_records(
        manifest,
        expected_split="val",
        expected_source="SkyScript",
    )

    assert records == expected


def test_load_paired_records_rejects_duplicate_normalized_caption(tmp_path) -> None:
    manifest, _ = _write_paired_manifest(tmp_path, captions=("A Forest", "  a   forest "))

    with pytest.raises(ValueError, match="duplicate caption"):
        load_paired_records(manifest, expected_split="val", expected_source="SkyScript")


def test_evaluate_paired_retrieval_uses_manifest_row_positives(tmp_path, monkeypatch) -> None:
    manifest, _ = _write_paired_manifest(tmp_path)
    image_features = torch.eye(2)
    text_features = torch.eye(2)
    monkeypatch.setattr(
        "dinotxt_rs.evaluation.retrieval.encode_images",
        lambda *args, **kwargs: (image_features, 100.0, {"images": 2}),
    )
    monkeypatch.setattr(
        "dinotxt_rs.evaluation.retrieval.encode_texts",
        lambda *args, **kwargs: (text_features, 100.0, {"captions": 2}),
    )
    model = SimpleNamespace(device=torch.device("cpu"), metadata={"checkpoint": None})

    report = evaluate_paired_retrieval(
        model,
        manifest,
        batch_size=2,
        num_workers=0,
        retrieval_chunk_size=1,
        split="val",
        source="SkyScript",
    )

    assert report["task"] == "paired_image_text_global_retrieval"
    assert report["positive_definition"] == "manifest_row_one_to_one"
    assert report["sources"] == ["SkyScript"]
    assert report["counts"] == {"images": 2, "captions": 2, "pairs": 2}
    assert report["metrics"]["mean_recall"] == pytest.approx(1.0)
