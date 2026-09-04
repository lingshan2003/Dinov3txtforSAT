import pytest
import torch

from dinotxt_rs.evaluation.retrieval import retrieval_metrics


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
