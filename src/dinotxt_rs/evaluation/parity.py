from __future__ import annotations

import gc
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from dinotxt_rs.config import Config
from dinotxt_rs.data import ImageTextDataset, collate_image_text, make_transform
from dinotxt_rs.evaluation.common import evaluation_runtime, load_evaluation_model
from dinotxt_rs.losses import symmetric_contrastive_loss
from dinotxt_rs.training.provenance import git_commit, sha256_file


def default_tolerances(precision: str) -> tuple[float, float]:
    if precision == "fp32":
        return 1e-6, 1e-5
    return 1e-3, 1e-3


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable_parameter_fingerprint(model: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    names: list[str] = []
    tensors = 0
    elements = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        tensors += 1
        elements += parameter.numel()
        record = json.dumps(
            {
                "name": name,
                "dtype": str(parameter.dtype),
                "shape": list(parameter.shape),
                "sha256": tensor_sha256(parameter),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
    if not names:
        raise ValueError("Parity model has no trainable parameters")
    return {
        "sha256": digest.hexdigest(),
        "tensors": tensors,
        "elements": elements,
        "names_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
    }


def compare_tensors(
    reference: torch.Tensor,
    observed: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if reference.shape != observed.shape:
        return {
            "passed": False,
            "reference_shape": list(reference.shape),
            "observed_shape": list(observed.shape),
            "max_absolute_error": None,
            "max_relative_error": None,
        }
    reference_float = reference.detach().cpu().float()
    observed_float = observed.detach().cpu().float()
    finite = bool(torch.isfinite(reference_float).all() and torch.isfinite(observed_float).all())
    if not finite:
        return {
            "passed": False,
            "reference_shape": list(reference.shape),
            "observed_shape": list(observed.shape),
            "finite": False,
            "max_absolute_error": None,
            "max_relative_error": None,
        }
    difference = (reference_float - observed_float).abs()
    denominator = reference_float.abs().clamp_min(max(atol, torch.finfo(torch.float32).eps))
    return {
        "passed": bool(torch.allclose(reference_float, observed_float, atol=atol, rtol=rtol)),
        "reference_shape": list(reference.shape),
        "observed_shape": list(observed.shape),
        "finite": True,
        "max_absolute_error": float(difference.max()) if difference.numel() else 0.0,
        "max_relative_error": (
            float((difference / denominator).max()) if difference.numel() else 0.0
        ),
    }


def _autocast(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _capture_outputs(
    evaluation_model: Any,
    *,
    pixels: torch.Tensor,
    captions: list[str],
) -> dict[str, torch.Tensor]:
    device_pixels = pixels.to(evaluation_model.device)
    tokens = evaluation_model.tokenizer.tokenize(captions).to(evaluation_model.device)
    with torch.inference_mode(), _autocast(
        evaluation_model.device, evaluation_model.config.train.precision
    ):
        outputs = evaluation_model.model(device_pixels, tokens)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
            raise ValueError("Official dino.txt forward must return exactly five values")
        image_features, text_features, logit_scale, patch_tokens, backbone_patch_tokens = outputs
        named = {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": logit_scale.reshape(1),
            "patch_tokens": patch_tokens,
            "backbone_patch_tokens": backbone_patch_tokens,
        }
        for name, value in named.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Parity output {name} is not a tensor")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Parity output {name} contains non-finite values")
        similarity_logits = logit_scale.float() * image_features.float() @ text_features.float().T
        loss = symmetric_contrastive_loss(
            image_features, text_features, logit_scale, queue=None
        ).loss
        named["similarity_logits"] = similarity_logits
        named["symmetric_contrastive_loss"] = loss.reshape(1)
    return {
        "tokens": tokens.detach().cpu(),
        **{name: value.detach().cpu() for name, value in named.items()},
    }


def _load_input(
    config: Config, manifest: Path, batch_size: int
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("Parity batch_size must be positive")
    dataset = ImageTextDataset(
        manifest,
        make_transform(config.model.image_size, config.model.backbone_domain, train=False),
    )
    if len(dataset) < batch_size:
        raise ValueError(
            f"Parity manifest has {len(dataset)} samples, fewer than batch_size={batch_size}"
        )
    samples = [dataset[index] for index in range(batch_size)]
    batch = collate_image_text(samples)
    records = dataset.records[:batch_size]
    image_records = [
        {
            "id": str(record["id"]),
            "path": str(Path(record["image"]).resolve()),
            "sha256": sha256_file(Path(record["image"])),
            "caption_sha256": hashlib.sha256(str(record["caption"]).encode("utf-8")).hexdigest(),
        }
        for record in records
    ]
    metadata = {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "batch_size": batch_size,
        "ids": batch["ids"],
        "pixels_sha256": tensor_sha256(batch["pixels"]),
        "images": image_records,
    }
    metadata["identity_sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return batch["pixels"], batch["captions"], metadata


def _release_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_step0_parity(
    config: Config,
    *,
    checkpoint: Path,
    training_output: Path,
    input_manifest: Path,
    batch_size: int = 16,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict[str, Any]:
    if atol is None or rtol is None:
        default_atol, default_rtol = default_tolerances(config.train.precision)
        atol = default_atol if atol is None else atol
        rtol = default_rtol if rtol is None else rtol
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("Parity tolerances must be finite and nonnegative")

    pixels, captions, input_metadata = _load_input(config, input_manifest, batch_size)

    official = load_evaluation_model(config)
    official_metadata = official.metadata
    official_fingerprint = trainable_parameter_fingerprint(official.model)
    official_outputs = _capture_outputs(official, pixels=pixels, captions=captions)
    runtime = evaluation_runtime(official.device)
    official_device = official.device
    del official
    _release_device_cache(official_device)

    restored = load_evaluation_model(
        config,
        checkpoint=checkpoint,
        training_output=training_output,
    )
    checkpoint_metadata = restored.metadata.get("checkpoint")
    if not isinstance(checkpoint_metadata, dict) or checkpoint_metadata.get("step") != 0:
        observed_step = (
            checkpoint_metadata.get("step") if isinstance(checkpoint_metadata, dict) else None
        )
        restored_device = restored.device
        del restored
        _release_device_cache(restored_device)
        raise ValueError(f"Parity requires a step-0 checkpoint, got step={observed_step!r}")
    restored_fingerprint = trainable_parameter_fingerprint(restored.model)
    restored_outputs = _capture_outputs(restored, pixels=pixels, captions=captions)
    restored_metadata = restored.metadata
    restored_device = restored.device
    del restored
    _release_device_cache(restored_device)

    token_match = bool(torch.equal(official_outputs["tokens"], restored_outputs["tokens"]))
    comparisons = {
        name: compare_tensors(
            official_outputs[name], restored_outputs[name], atol=atol, rtol=rtol
        )
        for name in (
            "image_features",
            "text_features",
            "logit_scale",
            "patch_tokens",
            "backbone_patch_tokens",
            "similarity_logits",
            "symmetric_contrastive_loss",
        )
    }
    parameter_match = official_fingerprint == restored_fingerprint
    passed = token_match and parameter_match and all(
        comparison["passed"] for comparison in comparisons.values()
    )
    return {
        "format_version": 1,
        "status": "pass" if passed else "fail",
        "project_commit": git_commit(config.source.parent),
        "backbone_domain": config.model.backbone_domain,
        "precision": config.train.precision,
        "tolerances": {"atol": atol, "rtol": rtol},
        "input": input_metadata,
        "official_model": official_metadata,
        "step0_model": restored_metadata,
        "checkpoint_step": checkpoint_metadata["step"],
        "trainable_parameters": {
            "passed": parameter_match,
            "official": official_fingerprint,
            "step0": restored_fingerprint,
        },
        "tokens": {
            "passed": token_match,
            "shape": list(official_outputs["tokens"].shape),
            "official_sha256": tensor_sha256(official_outputs["tokens"]),
            "step0_sha256": tensor_sha256(restored_outputs["tokens"]),
        },
        "comparisons": comparisons,
        "runtime": runtime,
    }
