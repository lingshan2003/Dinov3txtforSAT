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

RETRIEVAL_TIE_POLICY = "score_descending_then_candidate_index_ascending"


def load_rsicd_records(
    path: Path, *, expected_split: str = "test"
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    expected_split = expected_split.lower()
    if expected_split not in {"train", "val", "test"}:
        raise ValueError("RSICD expected_split must be train, val, or test")
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
        if value["split"].lower() != expected_split or value["source"] != "RSICD":
            raise ValueError(f"{path}:{line_number}: expected RSICD split={expected_split!r} only")
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


def load_paired_records(
    path: Path,
    *,
    expected_split: str,
    expected_source: str | None = None,
) -> list[dict[str, str]]:
    """Load a canonical manifest whose rows define one-to-one image/text positives."""
    expected_split = expected_split.lower()
    if not expected_split:
        raise ValueError("Paired retrieval expected_split must be nonempty")
    if expected_source is not None and not expected_source:
        raise ValueError("Paired retrieval expected_source must be nonempty when provided")

    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_images: set[Path] = set()
    seen_captions: set[str] = set()
    required = ("id", "image", "caption", "split", "source")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        if any(not isinstance(value.get(field), str) or not value[field] for field in required):
            raise ValueError(f"{path}:{line_number}: missing a required paired field")
        if value["split"].lower() != expected_split:
            raise ValueError(f"{path}:{line_number}: expected split={expected_split!r} only")
        if expected_source is not None and value["source"] != expected_source:
            raise ValueError(f"{path}:{line_number}: expected source={expected_source!r} only")
        if value["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate sample id {value['id']!r}")

        image_path = Path(value["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"{path}:{line_number}: image does not exist: {image_path}")
        resolved_image = image_path.resolve()
        if resolved_image in seen_images:
            raise ValueError(f"{path}:{line_number}: duplicate image in one-to-one manifest")

        normalized_caption = " ".join(value["caption"].split()).casefold()
        if normalized_caption in seen_captions:
            raise ValueError(f"{path}:{line_number}: duplicate caption in one-to-one manifest")

        records.append({field: value[field] for field in required})
        seen_ids.add(value["id"])
        seen_images.add(resolved_image)
        seen_captions.add(normalized_caption)
    if not records:
        raise ValueError(f"Paired retrieval manifest is empty: {path}")
    return records


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


def load_caption_group_records(
    path: Path,
    *,
    expected_split: str,
    expected_source: str = "SkyScript",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load unique images and deduplicate texts by an exact normalized caption group.

    Both candidate pools retain first-occurrence manifest order for deterministic
    ties. One image belongs to one caption group in this protocol. Captions use
    the first annotated spelling in each group; no semantic groups are inferred.
    """
    expected_split = expected_split.lower()
    if expected_split not in {"train", "val", "test"} or not expected_source.strip():
        raise ValueError("Caption-group retrieval needs a valid split and nonempty source")
    required = ("id", "image", "caption", "split", "source", "group_id")
    records: list[dict[str, str]] = []
    texts: dict[str, dict[str, str]] = {}
    seen_ids: set[str] = set()
    seen_images: set[Path] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict) or any(
            not isinstance(value.get(field), str) or not value[field].strip() for field in required
        ):
            raise ValueError(f"{path}:{line_number}: missing a required caption-group field")
        if value["split"].lower() != expected_split or value["source"] != expected_source:
            raise ValueError(
                f"{path}:{line_number}: expected source={expected_source!r}, "
                f"split={expected_split!r} only"
            )
        normalized_caption = " ".join(value["caption"].split()).casefold()
        if value["group_id"] != normalized_caption:
            raise ValueError(f"{path}:{line_number}: group_id must equal normalized caption")
        if value["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate sample id {value['id']!r}")
        image = Path(value["image"])
        if not image.is_file():
            raise FileNotFoundError(f"{path}:{line_number}: image does not exist: {image}")
        if image.resolve() in seen_images:
            raise ValueError(f"{path}:{line_number}: duplicate image in caption-group manifest")
        records.append({field: value[field] for field in required})
        texts.setdefault(
            value["group_id"],
            {"group_id": value["group_id"], "caption": value["caption"]},
        )
        seen_ids.add(value["id"])
        seen_images.add(image.resolve())
    if not records:
        raise ValueError(f"Caption-group retrieval manifest is empty: {path}")
    return records, list(texts.values())


def _ranks_from_positive_sets(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    positive_candidate_indices: list[list[int]],
    *,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the best positive rank with deterministic, exact-score tie breaking.

    Candidates are ordered by descending score, then ascending candidate index.
    For multiple positives, choose the highest score and the lowest index among
    positives sharing that score. No perturbation is added to the scores.
    """
    if len(query_features) != len(positive_candidate_indices):
        raise ValueError("Every retrieval query needs a positive candidate set")
    if chunk_size <= 0:
        raise ValueError("Retrieval chunk_size must be positive")
    candidates = candidate_features.to(device)
    candidate_indices = torch.arange(len(candidates), device=device)
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
            positive_indices = torch.tensor(positives, device=device)
            query_scores = scores[local_index]
            positive_scores = query_scores[positive_indices]
            best_positive_score = positive_scores.max()
            best_positive_index = positive_indices[positive_scores == best_positive_score].min()
            preceding_candidates = (query_scores > best_positive_score) | (
                (query_scores == best_positive_score) & (candidate_indices < best_positive_index)
            )
            batch_ranks.append(preceding_candidates.sum() + 1)
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


def retrieval_metrics_from_positive_sets(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_to_text_positives: list[list[int]],
    text_to_image_positives: list[list[int]],
    *,
    device: torch.device,
    chunk_size: int = 256,
    image_group_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Report query hit Recall@K for a reciprocal explicit bipartite relation.

    Recall hits when ANY annotated positive occurs in top K, not the proportion
    of all positives recovered. Optional image groups give macro-average image
    hit rates so that groups with many images do not dominate the result.
    """
    for label, features in (("image", image_features), ("text", text_features)):
        if features.ndim != 2 or not len(features) or not torch.isfinite(features).all():
            raise ValueError(f"Retrieval {label} features must be nonempty finite matrices")
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError("Retrieval feature dimensions must match")
    relations: list[set[tuple[int, int]]] = []
    for label, positives, queries, candidates in (
        ("image", image_to_text_positives, len(image_features), len(text_features)),
        ("text", text_to_image_positives, len(text_features), len(image_features)),
    ):
        if len(positives) != queries:
            raise ValueError(f"Every {label} query needs a positive candidate set")
        edges: set[tuple[int, int]] = set()
        for query_index, indices in enumerate(positives):
            if not indices:
                raise ValueError(f"Retrieval {label} query has no positive candidate")
            if any(type(index) is not int or not 0 <= index < candidates for index in indices):
                raise ValueError(f"Retrieval {label} positive index is out of bounds")
            if len(set(indices)) != len(indices):
                raise ValueError(f"Retrieval {label} positive set contains duplicate indices")
            edges.update(
                (query_index, index) if label == "image" else (index, query_index)
                for index in indices
            )
        relations.append(edges)
    if relations[0] != relations[1]:
        raise ValueError("Image/text positive relations must be reciprocal")
    i2t_ranks = _ranks_from_positive_sets(
        image_features,
        text_features,
        image_to_text_positives,
        chunk_size=chunk_size,
        device=device,
    )
    t2i_ranks = _ranks_from_positive_sets(
        text_features,
        image_features,
        text_to_image_positives,
        chunk_size=chunk_size,
        device=device,
    )
    i2t, t2i = _metric_summary(i2t_ranks), _metric_summary(t2i_ranks)
    result = {
        "image_to_text": i2t,
        "text_to_image": t2i,
        "mean_recall": sum(values[key] for values in (i2t, t2i) for key in ("r1", "r5", "r10")) / 6,
    }
    if image_group_ids is not None:
        if len(image_group_ids) != len(image_features) or any(
            not isinstance(group, str) or not group.strip() for group in image_group_ids
        ):
            raise ValueError("Every image needs a nonempty group id for group-balanced metrics")
        groups: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(image_group_ids):
            groups[group].append(index)
        balanced = {
            f"r{k}": sum(
                float((i2t_ranks[indices] <= k).float().mean()) for indices in groups.values()
            )
            / len(groups)
            for k in (1, 5, 10)
        }
        result["image_to_text_group_balanced"] = balanced
        result["group_balanced_mean_recall"] = (
            sum(balanced.values()) + sum(t2i[key] for key in ("r1", "r5", "r10"))
        ) / 6
    return result


def evaluate_rsicd_retrieval(
    evaluation_model: EvaluationModel,
    manifest: Path,
    *,
    batch_size: int,
    num_workers: int,
    retrieval_chunk_size: int,
    split: str = "test",
) -> dict[str, Any]:
    split = split.lower()
    image_records, caption_records = load_rsicd_records(manifest, expected_split=split)
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
        "tie_policy": RETRIEVAL_TIE_POLICY,
        "manifest": manifest_metadata(manifest),
        "model": evaluation_model.metadata,
        "split": split,
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


def evaluate_paired_retrieval(
    evaluation_model: EvaluationModel,
    manifest: Path,
    *,
    batch_size: int,
    num_workers: int,
    retrieval_chunk_size: int,
    split: str,
    source: str | None = None,
) -> dict[str, Any]:
    """Evaluate global retrieval when each manifest row is exactly one positive pair."""
    records = load_paired_records(
        manifest,
        expected_split=split,
        expected_source=source,
    )
    image_records = [{"id": record["id"], "image": record["image"]} for record in records]
    image_features, image_scale, image_stats = encode_images(
        evaluation_model,
        image_records,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    text_features, text_scale, text_stats = encode_texts(
        evaluation_model,
        [record["caption"] for record in records],
        batch_size=batch_size,
    )
    metrics = retrieval_metrics(
        image_features,
        text_features,
        list(range(len(records))),
        device=evaluation_model.device,
        chunk_size=retrieval_chunk_size,
    )
    sources = sorted({record["source"] for record in records})
    return {
        "format_version": 1,
        "task": "paired_image_text_global_retrieval",
        "tie_policy": RETRIEVAL_TIE_POLICY,
        "manifest": manifest_metadata(manifest),
        "model": evaluation_model.metadata,
        "split": split.lower(),
        "sources": sources,
        "positive_definition": "manifest_row_one_to_one",
        "metrics": metrics,
        "counts": {"images": len(records), "captions": len(records), "pairs": len(records)},
        "encoding": {
            "image": image_stats,
            "text": text_stats,
            "image_logit_scale": image_scale,
            "text_logit_scale": text_scale,
            "retrieval_chunk_size": retrieval_chunk_size,
        },
        "runtime": evaluation_runtime(evaluation_model.device),
    }


def evaluate_caption_group_retrieval(
    evaluation_model: EvaluationModel,
    manifest: Path,
    *,
    batch_size: int,
    num_workers: int,
    retrieval_chunk_size: int,
    split: str,
    source: str = "SkyScript",
) -> dict[str, Any]:
    """Evaluate deduplicated caption queries against every held-out group image."""
    images, captions = load_caption_group_records(
        manifest, expected_split=split, expected_source=source
    )
    image_features, image_scale, image_stats = encode_images(
        evaluation_model,
        images,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    text_features, text_scale, text_stats = encode_texts(
        evaluation_model,
        [record["caption"] for record in captions],
        batch_size=batch_size,
    )
    caption_index = {record["group_id"]: index for index, record in enumerate(captions)}
    image_to_text = [[caption_index[record["group_id"]]] for record in images]
    text_to_image: list[list[int]] = [[] for _ in captions]
    for index, (text_index,) in enumerate(image_to_text):
        text_to_image[text_index].append(index)
    metrics = retrieval_metrics_from_positive_sets(
        image_features,
        text_features,
        image_to_text,
        text_to_image,
        device=evaluation_model.device,
        chunk_size=retrieval_chunk_size,
        image_group_ids=[record["group_id"] for record in images],
    )
    return {
        "format_version": 1,
        "task": "caption_group_image_text_global_retrieval",
        "tie_policy": RETRIEVAL_TIE_POLICY,
        "manifest": manifest_metadata(manifest),
        "model": evaluation_model.metadata,
        "split": split.lower(),
        "sources": [source],
        "positive_definition": "normalized_complete_caption_group",
        "recall_definition": "fraction_of_queries_with_any_positive_in_top_k",
        "caption_candidates": "first_manifest_caption_per_group",
        "metrics": metrics,
        "counts": {
            "images": len(images),
            "captions": len(captions),
            "groups": len(captions),
            "positive_pairs": len(images),
        },
        "encoding": {
            "image": image_stats,
            "text": text_stats,
            "image_logit_scale": image_scale,
            "text_logit_scale": text_scale,
            "retrieval_chunk_size": retrieval_chunk_size,
        },
        "runtime": evaluation_runtime(evaluation_model.device),
    }
