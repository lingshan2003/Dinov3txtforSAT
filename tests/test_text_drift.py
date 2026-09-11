import json
from pathlib import Path

import pytest
import torch

from dinotxt_rs.evaluation.text_drift import (
    compare_text_embeddings,
    load_caption_records,
)


def test_compare_text_embeddings_reports_identity_and_rotation() -> None:
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    identity = compare_text_embeddings(reference, reference.clone())
    rotated = compare_text_embeddings(
        reference,
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    )

    assert identity["cosine_distance"]["max"] == pytest.approx(0.0)
    assert identity["l2_distance"]["max"] == pytest.approx(0.0)
    assert rotated["cosine_similarity"]["mean"] == pytest.approx(0.5)
    assert rotated["cosine_distance"]["mean"] == pytest.approx(0.5)
    assert rotated["fraction_cosine_distance_above"]["1e-2"] == pytest.approx(0.5)


def test_load_caption_records_preserves_order_and_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "captions.jsonl"
    records = [
        {"id": "b", "caption": "second", "image": "unused-b.png"},
        {"id": "a", "caption": "first", "image": "unused-a.png"},
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    assert load_caption_records(manifest) == [
        {"id": "b", "caption": "second"},
        {"id": "a", "caption": "first"},
    ]

    manifest.write_text(
        json.dumps(records[0]) + "\n" + json.dumps(records[0]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        load_caption_records(manifest)
