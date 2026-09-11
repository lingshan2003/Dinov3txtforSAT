from __future__ import annotations

import gc
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from dinotxt_rs.config import Config
from dinotxt_rs.evaluation.common import (
    encode_texts,
    evaluation_runtime,
    load_evaluation_model,
    load_official_reference_model,
    manifest_metadata,
    validate_finite_metric,
)


def load_caption_records(path: Path) -> list[dict[str, str]]:
    """Load text identities without requiring the corresponding images to be decoded."""
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        sample_id = value.get("id")
        caption = value.get("caption")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_number}: missing nonempty id")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"{path}:{line_number}: missing nonempty caption")
        if sample_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {sample_id!r}")
        records.append({"id": sample_id, "caption": caption})
        seen_ids.add(sample_id)
    if not records:
        raise ValueError(f"Caption manifest is empty: {path}")
    return records


def _distribution(values: torch.Tensor) -> dict[str, float]:
    if values.ndim != 1 or not len(values):
        raise ValueError("Text drift distribution must be a nonempty vector")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Text drift distribution contains non-finite values")
    result = {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "p05": float(torch.quantile(values, 0.05)),
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }
    for name, value in result.items():
        validate_finite_metric(value, f"text drift {name}")
    return result


def compare_text_embeddings(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("Text embeddings must be rank-two matrices")
    if reference.shape != candidate.shape or not reference.shape[0]:
        raise ValueError(
            f"Text embedding shapes must match and be nonempty: {reference.shape}, "
            f"{candidate.shape}"
        )
    reference = torch.nn.functional.normalize(reference.float(), dim=-1)
    candidate = torch.nn.functional.normalize(candidate.float(), dim=-1)
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise FloatingPointError("Text embeddings contain non-finite values")
    cosine_similarity = (reference * candidate).sum(dim=-1).clamp(-1.0, 1.0)
    cosine_distance = 1.0 - cosine_similarity
    l2_distance = (reference - candidate).norm(dim=-1)
    return {
        "captions": reference.shape[0],
        "embedding_dim": reference.shape[1],
        "cosine_similarity": _distribution(cosine_similarity),
        "cosine_distance": _distribution(cosine_distance),
        "l2_distance": _distribution(l2_distance),
        "fraction_cosine_distance_above": {
            "1e-4": float((cosine_distance > 1e-4).float().mean()),
            "1e-3": float((cosine_distance > 1e-3).float().mean()),
            "1e-2": float((cosine_distance > 1e-2).float().mean()),
        },
    }


def _release_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def evaluate_text_embedding_drift(
    config: Config,
    *,
    manifests: Mapping[str, Path],
    checkpoints: Mapping[int, Path],
    training_output: Path,
    batch_size: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("Text drift batch_size must be positive")
    if not manifests:
        raise ValueError("Text drift evaluation requires at least one manifest")
    if not checkpoints:
        raise ValueError("Text drift evaluation requires at least one checkpoint")
    if any(step < 0 for step in checkpoints):
        raise ValueError("Text drift checkpoint steps must be nonnegative")

    records = {name: load_caption_records(path) for name, path in manifests.items()}
    reference_model = load_official_reference_model(config)
    reference_metadata = reference_model.metadata
    reference_features: dict[str, torch.Tensor] = {}
    reference_encoding: dict[str, Any] = {}
    for name, dataset_records in records.items():
        features, _, stats = encode_texts(
            reference_model,
            [record["caption"] for record in dataset_records],
            batch_size=batch_size,
        )
        reference_features[name] = features
        reference_encoding[name] = stats
    runtime = evaluation_runtime(reference_model.device)
    reference_device = reference_model.device
    del reference_model
    _release_device_cache(reference_device)

    checkpoint_reports: dict[str, Any] = {}
    for step, checkpoint in sorted(checkpoints.items()):
        candidate_model = load_evaluation_model(
            config,
            checkpoint=checkpoint,
            training_output=training_output,
        )
        checkpoint_metadata = candidate_model.metadata.get("checkpoint")
        if not isinstance(checkpoint_metadata, dict) or checkpoint_metadata.get("step") != step:
            observed = (
                checkpoint_metadata.get("step")
                if isinstance(checkpoint_metadata, dict)
                else None
            )
            candidate_device = candidate_model.device
            del candidate_model
            _release_device_cache(candidate_device)
            raise ValueError(
                f"Text drift checkpoint label step={step} does not match payload step={observed}"
            )
        dataset_reports: dict[str, Any] = {}
        all_reference: list[torch.Tensor] = []
        all_candidate: list[torch.Tensor] = []
        for name, dataset_records in records.items():
            candidate_features, _, stats = encode_texts(
                candidate_model,
                [record["caption"] for record in dataset_records],
                batch_size=batch_size,
            )
            dataset_reports[name] = {
                **compare_text_embeddings(reference_features[name], candidate_features),
                "encoding": stats,
            }
            all_reference.append(reference_features[name])
            all_candidate.append(candidate_features)
        checkpoint_reports[str(step)] = {
            "checkpoint": checkpoint_metadata,
            "datasets": dataset_reports,
            "aggregate": compare_text_embeddings(
                torch.cat(all_reference), torch.cat(all_candidate)
            ),
        }
        candidate_device = candidate_model.device
        del candidate_model
        _release_device_cache(candidate_device)

    for step, report in checkpoint_reports.items():
        mean_distance = report["aggregate"]["cosine_distance"]["mean"]
        if not math.isfinite(mean_distance):
            raise FloatingPointError(f"Non-finite aggregate drift at step {step}")
    return {
        "format_version": 1,
        "task": "text_embedding_drift_against_frozen_official_reference",
        "reference_definition": (
            "Normalized text embeddings from the frozen official dino.txt text model before "
            "any project fine-tuning, evaluated on identical ordered caption strings."
        ),
        "model": {
            "backbone_domain": config.model.backbone_domain,
            "training_output": str(training_output.resolve()),
            "reference": reference_metadata,
        },
        "manifests": {
            name: {
                **manifest_metadata(path),
                "captions": len(records[name]),
                "ordered_ids_match_by_construction": True,
                "reference_encoding": reference_encoding[name],
            }
            for name, path in manifests.items()
        },
        "checkpoints": checkpoint_reports,
        "runtime": runtime,
    }
