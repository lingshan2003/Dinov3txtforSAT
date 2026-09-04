from __future__ import annotations

import json
import math
import os
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dinotxt_rs.config import Config
from dinotxt_rs.data import make_transform
from dinotxt_rs.models import configure_trainable_parameters, load_official_dinotxt
from dinotxt_rs.training.provenance import run_identity, sha256_file, sha256_text


def _autocast(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _checkpoint_identity_mismatches(
    expected_identity: dict[str, Any], observed_identity: Any
) -> list[str]:
    if not isinstance(observed_identity, dict):
        return ["run_identity"]
    mismatches = [
        name
        for name in ("format_version", "config_sha256", "project_commit", "dinov3_commit")
        if observed_identity.get(name) != expected_identity.get(name)
    ]
    if observed_identity.get("files") != expected_identity.get("files"):
        mismatches.append("files")
    return mismatches


def _load_evaluation_checkpoint(
    *,
    model: Any,
    config: Config,
    checkpoint: Path,
    training_output: Path,
    current_input_hashes: dict[str, str],
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    training_output = training_output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
    saved_config_path = training_output / "config.toml"
    provenance_path = training_output / "provenance.json"
    if not saved_config_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            "Checkpoint evaluation requires config.toml and provenance.json in training output: "
            f"{training_output}"
        )
    config_text = config.source.read_text(encoding="utf-8")
    if saved_config_path.read_text(encoding="utf-8") != config_text:
        raise ValueError("Evaluation config does not exactly match the checkpoint training config")
    provenance = _read_json_object(provenance_path)
    expected_identity = run_identity(config_text, provenance)
    expected_files = provenance.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError("Training provenance does not contain input file hashes")
    for name, actual_hash in current_input_hashes.items():
        expected_record = expected_files.get(name)
        if not isinstance(expected_record, dict) or not isinstance(
            expected_record.get("sha256"), str
        ):
            raise ValueError(f"Training provenance has no SHA-256 for {name}")
        if actual_hash != expected_record["sha256"]:
            raise ValueError(f"Current evaluation input differs from checkpoint provenance: {name}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise ValueError("Evaluation requires a format_version=2 training checkpoint")
    mismatches = _checkpoint_identity_mismatches(expected_identity, payload.get("run_identity"))
    if mismatches:
        raise ValueError(
            "Refusing evaluation because checkpoint identity differs: " + ", ".join(mismatches)
        )
    if payload.get("config_toml") != config_text:
        raise ValueError("Checkpoint config snapshot does not exactly match evaluation config")
    trainable_state = payload.get("trainable_model")
    if not isinstance(trainable_state, dict):
        raise ValueError("Checkpoint trainable model state is invalid")
    expected_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(trainable_state) != expected_names:
        raise ValueError("Checkpoint trainable parameter names do not match the configured model")
    result = model.load_state_dict(trainable_state, strict=False)
    if result.unexpected_keys:
        raise ValueError("Checkpoint contains unexpected trainable parameter names")
    return {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "step": payload.get("step"),
        "run_identity": payload.get("run_identity"),
        "training_output": str(training_output),
    }


@dataclass(frozen=True)
class EvaluationModel:
    model: Any
    tokenizer: Any
    config: Config
    device: torch.device
    metadata: dict[str, Any]


def _config_input_hashes(config: Config) -> dict[str, str]:
    paths = {
        "backbone_weights": config.model.backbone_weights,
        "dinotxt_weights": config.model.dinotxt_weights,
        "bpe_vocab": config.model.bpe_vocab,
        "train_manifest": config.data.train_manifest,
    }
    if config.data.val_manifest is not None:
        paths["val_manifest"] = config.data.val_manifest
    if config.data.fixed_monitor_manifest is not None:
        paths["fixed_monitor_manifest"] = config.data.fixed_monitor_manifest
    return {name: sha256_file(path) for name, path in paths.items()}


def load_evaluation_model(
    config: Config,
    *,
    checkpoint: Path | None = None,
    training_output: Path | None = None,
) -> EvaluationModel:
    if (checkpoint is None) != (training_output is None):
        raise ValueError("checkpoint and training_output must be supplied together")
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Evaluation requested CUDA but torch.cuda.is_available() is false")
    input_hashes = _config_input_hashes(config)
    model, tokenizer = load_official_dinotxt(
        config.model.dinov3_repo,
        config.model.backbone_weights,
        config.model.dinotxt_weights,
        config.model.bpe_vocab,
    )
    counts = configure_trainable_parameters(
        model,
        text_last_k=config.model.text_last_k,
        train_vision_head=config.model.train_vision_head,
        train_text_projection=config.model.train_text_projection,
        train_logit_scale=config.model.train_logit_scale,
    )
    checkpoint_metadata = (
        None
        if checkpoint is None
        else _load_evaluation_checkpoint(
            model=model,
            config=config,
            checkpoint=checkpoint,
            training_output=training_output,
            current_input_hashes=input_hashes,
        )
    )
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return EvaluationModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        metadata={
            "model_config": str(config.source.resolve()),
            "model_config_sha256": sha256_text(config.source.read_text(encoding="utf-8")),
            "backbone_domain": config.model.backbone_domain,
            "backbone_weights": {
                "path": str(config.model.backbone_weights.resolve()),
                "sha256": input_hashes["backbone_weights"],
            },
            "dinotxt_weights": {
                "path": str(config.model.dinotxt_weights.resolve()),
                "sha256": input_hashes["dinotxt_weights"],
            },
            "bpe_vocab": {
                "path": str(config.model.bpe_vocab.resolve()),
                "sha256": input_hashes["bpe_vocab"],
            },
            "trainable_parameters": counts,
            "checkpoint": checkpoint_metadata,
        },
    )


class _ImageRecords(Dataset[dict[str, Any]]):
    def __init__(self, records: Sequence[dict[str, str]], transform: Any) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = Path(record["image"])
        try:
            with Image.open(path) as image:
                pixels = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read evaluation image {record['id']}: {path}") from exc
        return {"pixels": pixels, "id": record["id"]}


def _collate_images(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixels": torch.stack([sample["pixels"] for sample in batch]),
        "ids": [sample["id"] for sample in batch],
    }


def encode_images(
    evaluation_model: EvaluationModel,
    records: Sequence[dict[str, str]],
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("Evaluation batch_size must be positive and num_workers nonnegative")
    transform = make_transform(
        evaluation_model.config.model.image_size,
        evaluation_model.config.model.backbone_domain,
        train=False,
    )
    loader = DataLoader(
        _ImageRecords(records, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=evaluation_model.device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=_collate_images,
    )
    features: list[torch.Tensor] = []
    logit_scale: float | None = None
    start_time = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixels"].to(evaluation_model.device, non_blocking=True)
            captions = ["a remote sensing image"] * len(batch["ids"])
            tokens = evaluation_model.tokenizer.tokenize(captions).to(evaluation_model.device)
            with _autocast(evaluation_model.device, evaluation_model.config.train.precision):
                image_features, _, scale, _, _ = evaluation_model.model(pixels, tokens)
            if not torch.isfinite(image_features).all() or not torch.isfinite(scale).all():
                raise FloatingPointError("Non-finite image feature during evaluation")
            features.append(F.normalize(image_features.float(), dim=-1).cpu())
            logit_scale = float(scale)
    if not features or logit_scale is None:
        raise RuntimeError("Image evaluation completed without features")
    return (
        torch.cat(features),
        logit_scale,
        {
            "images": len(records),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "elapsed_seconds": time.monotonic() - start_time,
        },
    )


def encode_texts(
    evaluation_model: EvaluationModel,
    captions: Sequence[str],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("Evaluation text batch_size must be positive")
    if not captions:
        raise ValueError("Evaluation requires at least one text caption")
    features: list[torch.Tensor] = []
    logit_scale: float | None = None
    start_time = time.monotonic()
    image_size = evaluation_model.config.model.image_size
    with torch.inference_mode():
        for offset in range(0, len(captions), batch_size):
            caption_batch = list(captions[offset : offset + batch_size])
            tokens = evaluation_model.tokenizer.tokenize(caption_batch).to(evaluation_model.device)
            # The official public model exposes paired forward only. Its vision and text towers are
            # independent, so deterministic zero pixels safely provide the unused image argument.
            pixels = torch.zeros(
                (len(caption_batch), 3, image_size, image_size), device=evaluation_model.device
            )
            with _autocast(evaluation_model.device, evaluation_model.config.train.precision):
                _, text_features, scale, _, _ = evaluation_model.model(pixels, tokens)
            if not torch.isfinite(text_features).all() or not torch.isfinite(scale).all():
                raise FloatingPointError("Non-finite text feature during evaluation")
            features.append(F.normalize(text_features.float(), dim=-1).cpu())
            logit_scale = float(scale)
    if not features or logit_scale is None:
        raise RuntimeError("Text evaluation completed without features")
    return (
        torch.cat(features),
        logit_scale,
        {
            "captions": len(captions),
            "batch_size": batch_size,
            "elapsed_seconds": time.monotonic() - start_time,
        },
    )


def evaluation_runtime(device: torch.device) -> dict[str, Any]:
    peak: int | None = None
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_cuda_allocated_bytes": peak,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def manifest_metadata(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def validate_finite_metric(value: float, label: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite: {value!r}")
