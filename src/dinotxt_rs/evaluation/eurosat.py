from __future__ import annotations

import json
from collections import Counter
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

EUROSAT_CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

EUROSAT_CLASS_TEXT = {
    "AnnualCrop": "annual crop fields",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "highway",
    "Industrial": "industrial buildings",
    "Pasture": "pasture",
    "PermanentCrop": "permanent crop fields",
    "Residential": "residential buildings",
    "River": "river",
    "SeaLake": "sea or lake",
}

PROMPT_TEMPLATES = (
    "a satellite image of {class_name}",
    "a Sentinel-2 satellite image of {class_name}",
)


def load_eurosat_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        required = ("id", "image", "label", "split", "source")
        if any(not isinstance(value.get(field), str) or not value[field] for field in required):
            raise ValueError(f"{path}:{line_number}: missing a required EuroSAT field")
        if value["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {value['id']!r}")
        if value["label"] not in EUROSAT_CLASSES:
            raise ValueError(f"{path}:{line_number}: unknown EuroSAT label {value['label']!r}")
        if value["split"] != "test" or value["source"] != "EuroSAT":
            raise ValueError(f"{path}:{line_number}: unexpected EuroSAT split or source")
        image = Path(value["image"])
        if not image.is_file():
            raise FileNotFoundError(f"{path}:{line_number}: image does not exist: {image}")
        seen_ids.add(value["id"])
        records.append({field: value[field] for field in required})
    if not records:
        raise ValueError(f"EuroSAT manifest is empty: {path}")
    counts = Counter(record["label"] for record in records)
    if set(counts) != set(EUROSAT_CLASSES):
        raise ValueError("EuroSAT manifest does not contain every expected class")
    return records


def evaluate_eurosat(
    evaluation_model: EvaluationModel,
    manifest: Path,
    *,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    records = load_eurosat_records(manifest)
    prompts = [
        template.format(class_name=EUROSAT_CLASS_TEXT[label])
        for label in EUROSAT_CLASSES
        for template in PROMPT_TEMPLATES
    ]
    text_features, text_scale, text_stats = encode_texts(
        evaluation_model, prompts, batch_size=batch_size
    )
    prototypes = text_features.reshape(len(EUROSAT_CLASSES), len(PROMPT_TEMPLATES), -1).mean(dim=1)
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1).to(evaluation_model.device)
    image_features, image_scale, image_stats = encode_images(
        evaluation_model,
        records,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    labels = torch.tensor(
        [EUROSAT_CLASSES.index(record["label"]) for record in records], dtype=torch.long
    )
    predictions: list[torch.Tensor] = []
    for offset in range(0, len(records), batch_size):
        image_batch = image_features[offset : offset + batch_size].to(evaluation_model.device)
        logits = image_scale * image_batch @ prototypes.T
        predictions.append(logits.argmax(dim=1).cpu())
    predicted = torch.cat(predictions)
    correct = predicted.eq(labels)
    per_class_accuracy = {
        label: float(correct[labels == index].float().mean())
        for index, label in enumerate(EUROSAT_CLASSES)
    }
    accuracy = float(correct.float().mean())
    mean_per_class_accuracy = sum(per_class_accuracy.values()) / len(per_class_accuracy)
    validate_finite_metric(accuracy, "EuroSAT top-1 accuracy")
    validate_finite_metric(mean_per_class_accuracy, "EuroSAT mean per-class accuracy")
    return {
        "format_version": 1,
        "task": "eurosat_zero_shot_classification",
        "manifest": manifest_metadata(manifest),
        "model": evaluation_model.metadata,
        "classes": list(EUROSAT_CLASSES),
        "class_text": EUROSAT_CLASS_TEXT,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "prompts": prompts,
        "metrics": {
            "top1_accuracy": accuracy,
            "mean_per_class_accuracy": mean_per_class_accuracy,
            "per_class_accuracy": per_class_accuracy,
        },
        "counts": {
            "images": len(records),
            "class_counts": dict(sorted(Counter(record["label"] for record in records).items())),
        },
        "encoding": {
            "image": image_stats,
            "text": text_stats,
            "image_logit_scale": image_scale,
            "text_logit_scale": text_scale,
        },
        "runtime": evaluation_runtime(evaluation_model.device),
    }
