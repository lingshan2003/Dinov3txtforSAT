from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler[list[int]]):
    """A batch sampler whose consumed position is explicit checkpoint state.

    Advancing is intentionally done by the training loop after a physical batch is
    consumed.  This keeps the checkpoint position correct even when DataLoader
    workers prefetch future batches.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.epoch = 0
        self.batch_offset = 0
        self._order: torch.Tensor | None = None

    @property
    def batches_per_epoch(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _current_order(self) -> torch.Tensor:
        if self._order is None:
            if self.shuffle:
                self._order = torch.randperm(self.dataset_size, generator=self.generator)
            else:
                self._order = torch.arange(self.dataset_size)
        return self._order

    def __iter__(self) -> Iterator[list[int]]:
        order = self._current_order()
        for batch_index in range(self.batch_offset, self.batches_per_epoch):
            start = batch_index * self.batch_size
            stop = min(start + self.batch_size, self.dataset_size)
            indices = order[start:stop].tolist()
            if len(indices) == self.batch_size or not self.drop_last:
                yield indices

    def advance(self) -> None:
        if self.batch_offset >= self.batches_per_epoch:
            raise RuntimeError("Cannot advance a completed epoch")
        self.batch_offset += 1
        if self.batch_offset == self.batches_per_epoch:
            self.epoch += 1
            self.batch_offset = 0
            self._order = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "batch_offset": self.batch_offset,
            "generator_state": self.generator.get_state(),
            "order": self._order,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name, expected in {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }.items():
            if state.get(name) != expected:
                raise ValueError(
                    f"Checkpoint sampler {name}={state.get(name)!r} does not match {expected!r}"
                )
        epoch = state.get("epoch")
        batch_offset = state.get("batch_offset")
        order = state.get("order")
        generator_state = state.get("generator_state")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Checkpoint sampler epoch is invalid")
        if not isinstance(batch_offset, int) or not 0 <= batch_offset < self.batches_per_epoch:
            raise ValueError("Checkpoint sampler batch_offset is invalid")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("Checkpoint sampler generator_state is invalid")
        if order is not None:
            if not isinstance(order, torch.Tensor) or order.numel() != self.dataset_size:
                raise ValueError("Checkpoint sampler order is invalid")
            order = order.to(dtype=torch.int64, device="cpu")
        elif batch_offset:
            raise ValueError("Checkpoint sampler has a nonzero offset without an order")
        self.epoch = epoch
        self.batch_offset = batch_offset
        self._order = order
        self.generator.set_state(generator_state)
