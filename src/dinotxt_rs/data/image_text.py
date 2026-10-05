from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

WEB_MEAN = (0.485, 0.456, 0.406)
WEB_STD = (0.229, 0.224, 0.225)
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)


def make_transform(image_size: int, backbone_domain: str, train: bool = True) -> v2.Compose:
    mean, std = (SAT_MEAN, SAT_STD) if backbone_domain == "sat" else (WEB_MEAN, WEB_STD)
    spatial: list[Callable[..., Any]]
    if train:
        # Captions can encode north/south/east/west relationships, so geometry-changing
        # flips are not valid unless captions are transformed with the image.
        spatial = [
            v2.RandomResizedCrop((image_size, image_size), scale=(0.7, 1.0), antialias=True),
        ]
    else:
        spatial = [v2.Resize((image_size, image_size), antialias=True)]
    return v2.Compose(
        [v2.ToImage(), *spatial, v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    )


class ImageTextDataset(Dataset):
    """Dataset for the canonical JSONL manifest defined in docs/DEVELOPMENT_ARCHITECTURE.md."""

    def __init__(self, manifest: str | Path, transform: Callable[[Image.Image], Any]) -> None:
        self.manifest = Path(manifest)
        self.transform = transform
        self.records: list[dict[str, Any]] = []
        with self.manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                for field in ("id", "image", "caption", "split", "source"):
                    if field not in record:
                        raise ValueError(f"{self.manifest}:{line_number}: missing {field!r}")
                self.records.append(record)
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = Path(record["image"])
        try:
            with Image.open(image_path) as image:
                pixels = self.transform(image.convert("RGB"))
        except Exception as exc:
            message = f"Failed to read image for sample {record['id']}: {image_path}"
            raise RuntimeError(message) from exc
        sample = {"pixels": pixels, "caption": record["caption"], "id": record["id"]}
        if "group_id" in record:
            sample["group_id"] = record["group_id"]
            sample["index"] = index
        return sample


def collate_image_text(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    result = {
        "pixels": torch.stack([sample["pixels"] for sample in batch]),
        "captions": [sample["caption"] for sample in batch],
        "ids": [sample["id"] for sample in batch],
    }
    if any("group_id" in sample for sample in batch):
        if not all("group_id" in sample for sample in batch):
            raise ValueError("Mixed grouped and ungrouped samples in one batch")
        result["group_ids"] = [sample["group_id"] for sample in batch]
        result["indices"] = [sample["index"] for sample in batch]
    return result


def validate_caption_groups(records: list[dict[str, Any]], *, split: str) -> list[str]:
    """Validate the explicit, exact-caption relation before group sampling."""
    groups: list[str] = []
    seen_ids: set[str] = set()
    seen_images: set[Path] = set()
    for record in records:
        fields = ("id", "image", "caption", "split", "source", "group_id")
        if any(not isinstance(record.get(field), str) or not record[field] for field in fields):
            raise ValueError("Caption-group manifest requires nonempty string fields")
        normalized = " ".join(record["caption"].split()).casefold()
        if not normalized or record["group_id"] != normalized:
            raise ValueError("group_id must equal the normalized complete caption")
        if record["split"] != split:
            raise ValueError(f"Caption-group manifest must have split={split!r}")
        image = Path(record["image"]).resolve()
        if record["id"] in seen_ids or image in seen_images:
            raise ValueError("Duplicate sample id or image in caption-group manifest")
        if not image.is_file():
            raise FileNotFoundError(image)
        seen_ids.add(record["id"])
        seen_images.add(image)
        groups.append(record["group_id"])
    return groups
