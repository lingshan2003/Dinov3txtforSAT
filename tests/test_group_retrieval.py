import json
from types import SimpleNamespace

import pytest
import torch

from dinotxt_rs.evaluation.retrieval import (
    RETRIEVAL_TIE_POLICY,
    evaluate_caption_group_retrieval,
    load_caption_group_records,
    load_paired_records,
    retrieval_metrics,
    retrieval_metrics_from_positive_sets,
)

CPU = torch.device("cpu")


def _manifest(tmp_path, captions=("Airport", "  airport ", "Forest")):
    records = []
    for index, caption in enumerate(captions):
        image = tmp_path / f"image-{index}.jpg"
        image.write_bytes(b"image")
        records.append(
            {
                "id": str(index),
                "image": str(image),
                "caption": caption,
                "group_id": " ".join(caption.split()).casefold(),
                "split": "val",
                "source": "SkyScript",
            }
        )
    return _write(tmp_path, records), records


def _write(tmp_path, records):
    manifest = tmp_path / "groups.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return manifest


def test_group_loader_deduplicates_caption_candidates_preserving_manifest_order(tmp_path):
    manifest, expected = _manifest(tmp_path)
    images, texts = load_caption_group_records(manifest, expected_split="val")
    assert images == expected
    assert texts == [
        {"group_id": "airport", "caption": "Airport"},
        {"group_id": "forest", "caption": "Forest"},
    ]
    # The old protocol continues to reject the same data.
    with pytest.raises(ValueError, match="duplicate caption"):
        load_paired_records(manifest, expected_split="val", expected_source="SkyScript")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("group_id", "Airport", "normalized caption"),
        ("group_id", "forest", "normalized caption"),
        ("caption", "coastal airport", "normalized caption"),
        ("id", "0", "duplicate sample"),
        ("split", "train", "expected source"),
        ("source", "RSICD", "expected source"),
        ("source", " ", "required caption-group"),
        ("group_id", "", "required caption-group"),
        ("caption", "\t", "required caption-group"),
    ],
)
def test_group_loader_rejects_invalid_relation_or_metadata(tmp_path, field, value, error):
    _, records = _manifest(tmp_path)
    records[1][field] = value
    with pytest.raises(ValueError, match=error):
        load_caption_group_records(_write(tmp_path, records), expected_split="val")


def test_group_loader_rejects_duplicate_resolved_image_and_missing_fields(tmp_path):
    _, records = _manifest(tmp_path)
    records[1]["image"] = records[0]["image"]
    with pytest.raises(ValueError, match="duplicate image"):
        load_caption_group_records(_write(tmp_path, records), expected_split="val")
    del records[1]["source"]
    with pytest.raises(ValueError, match="required caption-group"):
        load_caption_group_records(_write(tmp_path, records), expected_split="val")


def test_group_loader_rejects_empty_manifest_or_missing_image(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_caption_group_records(_write(tmp_path, []), expected_split="val")
    _, records = _manifest(tmp_path)
    records[0]["image"] = str(tmp_path / "missing.jpg")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_caption_group_records(_write(tmp_path, records), expected_split="val")


@pytest.mark.parametrize("chunk_size", [1, 2, 256])
def test_text_queries_accept_any_of_several_positive_images(chunk_size):
    images = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    texts = torch.eye(2)
    result = retrieval_metrics_from_positive_sets(
        images,
        texts,
        [[0], [0], [1]],
        [[0, 1], [2]],
        device=CPU,
        chunk_size=chunk_size,
        image_group_ids=["airport", "airport", "forest"],
    )
    assert result["mean_recall"] == 1
    assert result["image_to_text_group_balanced"] == {"r1": 1.0, "r5": 1.0, "r10": 1.0}


def test_explicit_sets_support_multi_positives_in_both_directions_and_exact_ties():
    # All scores tie: only index order decides whether the first candidate is positive.
    result = retrieval_metrics_from_positive_sets(
        torch.ones(2, 2),
        torch.ones(3, 2),
        [[0, 1], [1, 2]],
        [[0], [0, 1], [1]],
        device=CPU,
        chunk_size=1,
    )
    assert result["image_to_text"]["r1"] == pytest.approx(0.5)
    assert result["image_to_text"]["mean_rank"] == pytest.approx(1.5)
    assert result["text_to_image"]["r1"] == pytest.approx(2 / 3)
    assert result["text_to_image"]["mean_rank"] == pytest.approx(4 / 3)


def test_singleton_positive_sets_match_legacy_metrics_exactly():
    generator = torch.Generator().manual_seed(13)
    images = torch.randn(12, 5, generator=generator)
    texts = torch.randn(12, 5, generator=generator)
    positives = [[index] for index in range(12)]
    new = retrieval_metrics_from_positive_sets(images, texts, positives, positives, device=CPU)
    old = retrieval_metrics(images, texts, list(range(12)), device=CPU)
    assert new == old


def test_group_balanced_metrics_give_each_caption_group_equal_weight():
    # Five unsuccessful group-A images and one successful group-B image.
    result = retrieval_metrics_from_positive_sets(
        torch.tensor([[0.0, 1.0]] * 6),
        torch.eye(2),
        [[0]] * 5 + [[1]],
        [list(range(5)), [5]],
        device=CPU,
        image_group_ids=["A"] * 5 + ["B"],
    )
    assert result["image_to_text"]["r1"] == pytest.approx(1 / 6)
    assert result["image_to_text_group_balanced"]["r1"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "image_sets,text_sets,error",
    [
        ([[0]], [[0], [1]], "Every image"),
        ([[], [1]], [[0], [1]], "no positive"),
        ([[2], [1]], [[0], [1]], "out of bounds"),
        ([[-1], [1]], [[0], [1]], "out of bounds"),
        ([[True], [1]], [[0], [1]], "out of bounds"),
        ([[0, 0], [1]], [[0], [1]], "duplicate"),
        ([[0], [1]], [[1], [0]], "reciprocal"),
    ],
)
def test_explicit_positive_sets_validate_indices_and_reciprocity(image_sets, text_sets, error):
    with pytest.raises(ValueError, match=error):
        retrieval_metrics_from_positive_sets(
            torch.eye(2),
            torch.eye(2),
            image_sets,
            text_sets,
            device=CPU,
        )


def test_group_evaluation_encodes_text_once_per_group_and_declares_protocol(tmp_path, monkeypatch):
    manifest, _ = _manifest(tmp_path)
    texts_encoded = []
    monkeypatch.setattr(
        "dinotxt_rs.evaluation.retrieval.encode_images",
        lambda *args, **kwargs: (torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 100.0, {}),
    )

    def encode_texts(model, captions, **kwargs):
        texts_encoded.extend(captions)
        return torch.eye(2), 100.0, {}

    monkeypatch.setattr("dinotxt_rs.evaluation.retrieval.encode_texts", encode_texts)
    report = evaluate_caption_group_retrieval(
        SimpleNamespace(device=CPU, metadata={}),
        manifest,
        batch_size=2,
        num_workers=0,
        retrieval_chunk_size=1,
        split="val",
    )
    assert texts_encoded == ["Airport", "Forest"]
    assert report["task"] == "caption_group_image_text_global_retrieval"
    assert report["positive_definition"] == "normalized_complete_caption_group"
    assert report["recall_definition"] == "fraction_of_queries_with_any_positive_in_top_k"
    assert report["tie_policy"] == RETRIEVAL_TIE_POLICY
    assert report["counts"] == {"images": 3, "captions": 2, "groups": 2, "positive_pairs": 3}
    assert report["metrics"]["mean_recall"] == 1.0


@pytest.mark.parametrize("choice", [None, "caption-group"])
def test_cli_preserves_default_and_accepts_new_protocol(monkeypatch, choice):
    from dinotxt_rs.cli.evaluate_skyscript import parse_args

    argv = ["evaluate", "--config", "model.toml", "--manifest", "val.jsonl", "--output", "r.json"]
    if choice:
        argv += ["--positive-definition", choice]
    monkeypatch.setattr("sys.argv", argv)
    assert parse_args().positive_definition == (choice or "one-to-one")
