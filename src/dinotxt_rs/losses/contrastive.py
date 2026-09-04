from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    loss: torch.Tensor
    image_to_text: torch.Tensor
    text_to_image: torch.Tensor


class EmbeddingQueue:
    """FIFO of detached negatives. It is deliberately not a source of positive pairs."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.images: torch.Tensor | None = None
        self.texts: torch.Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.images is None else self.images.shape[0]

    @torch.no_grad()
    def enqueue(self, images: torch.Tensor, texts: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        new_images = images.detach()
        new_texts = texts.detach()
        if self.images is not None:
            new_images = torch.cat([self.images, new_images], dim=0)
            new_texts = torch.cat([self.texts, new_texts], dim=0)
        self.images = new_images[-self.capacity :]
        self.texts = new_texts[-self.capacity :]

    def state_dict(self) -> dict[str, torch.Tensor | int | None]:
        return {
            "capacity": self.capacity,
            "images": None if self.images is None else self.images.detach().cpu(),
            "texts": None if self.texts is None else self.texts.detach().cpu(),
        }

    def load_state_dict(
        self, state: dict[str, torch.Tensor | int | None], device: torch.device
    ) -> None:
        if state.get("capacity") != self.capacity:
            raise ValueError(
                "Checkpoint queue capacity="
                f"{state.get('capacity')!r} does not match {self.capacity}"
            )
        images = state.get("images")
        texts = state.get("texts")
        if images is None and texts is None:
            self.images = None
            self.texts = None
            return
        if not isinstance(images, torch.Tensor) or not isinstance(texts, torch.Tensor):
            raise ValueError("Checkpoint queue embeddings are invalid")
        if images.ndim != 2 or texts.ndim != 2 or images.shape != texts.shape:
            raise ValueError("Checkpoint queue embedding shapes are invalid")
        if not 0 < images.shape[0] <= self.capacity:
            raise ValueError("Checkpoint queue length is invalid")
        self.images = images.to(device=device)
        self.texts = texts.to(device=device)


def symmetric_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    queue: EmbeddingQueue | None = None,
) -> LossOutput:
    if image_features.shape != text_features.shape:
        message = (
            "Paired embeddings must have equal shapes, got "
            f"{image_features.shape} and {text_features.shape}"
        )
        raise ValueError(
            message
        )
    all_text = text_features
    all_images = image_features
    if queue is not None and len(queue):
        all_text = torch.cat([text_features, queue.texts.to(text_features.device)], dim=0)
        all_images = torch.cat([image_features, queue.images.to(image_features.device)], dim=0)
    targets = torch.arange(image_features.shape[0], device=image_features.device)
    image_logits = logit_scale * image_features @ all_text.T
    text_logits = logit_scale * text_features @ all_images.T
    image_loss = F.cross_entropy(image_logits.float(), targets)
    text_loss = F.cross_entropy(text_logits.float(), targets)
    return LossOutput((image_loss + text_loss) / 2, image_loss, text_loss)
