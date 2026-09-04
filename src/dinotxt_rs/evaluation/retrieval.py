from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from dinotxt_rs.evaluation.common import (
    EvaluationModel,
    encode_images,
    encode_texts,
    evaluation_runtime,
    manifest_metadata,
    validate_finite_metric,
)


def load_rsicd_records(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_image: dict[str, dict[str, str]] = {}
    captions: list[dict[str, str]] = []
    seen_caption_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        required = ("id", "image_id", "image", "caption", "split", "source")
        if any(not isinstance(value.get(field), str) or not value[field] for field in required):
            raise ValueError(f"{path}:{line_number}: missing a required RSICD field")
        if value["id"] in seen_caption_ids:
            raise ValueError(f"{path}:{line_number}: duplicate caption id {value['id']!r}")
        if value["split"] != "test" or value["source"] != "RSICD":
            raise ValueError(f"{path}:{line_number}: retrieval must use RSICD test only")
        image_path = Path(value["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"{path}:{line_number}: image does not exist: {image_path}")
        existing = by_image.get(value["image_id"])
        image_record = {"id": value["image_id"], "image": value["image"]}
        if existing is not None and existing != image_record:
            raise ValueError(
                f"{path}:{line_number}: inconsistent image path for {value['image_id']}"
            )
        by_image[value["image_id"]] = image_record
        captions.append({field: value[field] for field in required})
        seen_caption_ids.add(value["id"])
    if not captions:
        raise ValueError(f"RSICD retrieval manifest is empty: {path}")
    images = [by_image[key] for key in sorted(by_image)]
    return images, sorted(captions, key=lambda record: record["id"])


def _metric_summary(ranks: torch.Tensor) -> dict[str, float]:
    if ranks.ndim != 1 or not len(ranks) or (ranks < 1).any():
        raise ValueError("Retrieval ranks must be a nonempty one-dimensional positive tensor")
    result = {
        "r1": float((ranks <= 1).float().mean()),
        "r5": float((ranks <= 5).float().mean()),
        "r10": float((ranks <= 10).float().mean()),
        "median_rank": float(ranks.float().median()),
        "mean_rank": float(ranks.float().mean()),
    }
    for label, value in result.items():
        validate_finite_metric(value, f"retrieval {label}")
    return result


def _ranks_from_positive_sets(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    positive_candidate_indices: list[list[int]],
    *,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if len(query_features) != len(positive_candidate_indices):
        raise ValueError("Every retrieval query needs a positive candidate set")
    if chunk_size <= 0:
        raise ValueError("Retrieval chunk_size must be positive")
    candidates = candidate_features.to(device)
    ranks: list[torch.Tensor] = []
    for offset in range(0, len(query_features), chunk_size):
        features = query_features[offset : offset + chunk_size].to(device)
        scores = features @ candidates.T
        batch_ranks: list[torch.Tensor] = []
        for local_index, positives in enumerate(
            positive_candidate_indices[offset : offset + len(features)]
        ):
            if not positives:
                raise ValueError("Retrieval query has no positive candidate")
            positive_scores = scores[local_index, torch.tensor(positives, device=device)]
            best_positive = positive_scores.max()
            batch_ranks.append((scores[local_index] > best_positive).sum() + 1)
        ranks.append(torch.stack(batch_ranks).cpu())
    return torch.cat(ranks).to(torch.long)


def retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    text_to_image: list[int],
    *,
    device: torch.device,
    chunk_size: int = 256,
) -> dict[str, Any]:
    if len(text_features) != len(text_to_image):
        raise ValueError("Every text feature needs one positive image index")
    image_to_text: dict[int, list[int]] = defaultdict(list)
    for text_index, image_index in enumerate(text_to_image):
        if not 0 <= image_index < len(image_features):
            raise ValueError("Text positive image index is out of bounds")
        image_to_text[image_index].append(text_index)
    if set(image_to_text) != set(range(len(image_features))):
        raise ValueError("Every image needs at least one positive text")
    i2t_ranks = _ranks_from_positive_sets(
        image_features,
        text_features,
        [image_to_text[index] for index in range(len(image_features))],
        chunk_size=chunk_size,
        device=device,
    )
    t2i_ranks = _ranks_from_positive_sets(
        text_features,
        image_features,
        [[image_index] for image_index in text_to_image],
        chunk_size=chunk_size,
        device=device,
    )
    i2t = _metric_summary(i2t_ranks)
    t2i = _metric_summary(t2i_ranks)
    return {
        "image_to_text": i2t,
        "text_to_image": t2i,
        "mean_recall": (i2t["r1"] + i2t["r5"] + i2t["r10"] + t2i["r1"] + t2i["r5"] + t2i["r10"])
        / 6,
    }


def evaluate_rsicd_retrieval(
    evaluation_model: EvaluationModel,
    manifest: Path,
    *,
    batch_size: int,
    num_workers: int,
    retrieval_chunk_size: int,
) -> dict[str, Any]:
    image_records, caption_records = load_rsicd_records(manifest)
    image_features, image_scale, image_stats = encode_images(
        evaluation_model,
        image_records,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    text_features, text_scale, text_stats = encode_texts(
        evaluation_model,
        [record["caption"] for record in caption_records],
        batch_size=batch_size,
    )
    image_index = {record["id"]: index for index, record in enumerate(image_records)}
    metrics = retrieval_metrics(
        image_features,
        text_features,
        [image_index[record["image_id"]] for record in caption_records],
        device=evaluation_model.device,
        chunk_size=retrieval_chunk_size,
    )
    return {
        "format_version": 1,
        "task": "rsicd_image_text_retrieval",
        "manifest": manifest_metadata(manifest),
        "model": evaluation_model.metadata,
        "split": "test",
        "metrics": metrics,
        "counts": {"images": len(image_records), "captions": len(caption_records)},
        "encoding": {
            "image": image_stats,
            "text": text_stats,
            "image_logit_scale": image_scale,
            "text_logit_scale": text_scale,
            "retrieval_chunk_size": retrieval_chunk_size,
        },
        "runtime": evaluation_runtime(evaluation_model.device),
    }
