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


class ResumableCaptionGroupBatchSampler(Sampler[list[int]]):
    """Visit caption groups once per epoch and rotate their available images.

    A group contributes up to ``images_per_group`` distinct rows, limited by the
    space left in the current physical batch. Groups never cross batch boundaries;
    singleton groups remain eligible. With ``drop_last``, the incomplete final
    batch is discarded and its image cursors do not advance. A group epoch is
    therefore different from a complete pass over every image.

    Epoch plans are prepared eagerly. Only ``advance`` changes the consumed
    position, so worker prefetch cannot move the checkpoint ahead of training.
    Rotation cursors describe the end of the prepared epoch, while start cursors
    and the full plan allow validation and exact restoration of its remainder.
    """

    def __init__(
        self,
        *,
        group_ids: list[str],
        batch_size: int,
        images_per_group: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        if not group_ids or any(not isinstance(group, str) or not group for group in group_ids):
            raise ValueError("group_ids must contain a nonempty string for every dataset row")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if type(images_per_group) is not int or not 1 <= images_per_group <= batch_size:
            raise ValueError("images_per_group must be between 1 and batch_size")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.group_ids = list(group_ids)
        self.dataset_size = len(group_ids)
        self.batch_size = batch_size
        self.images_per_group = images_per_group
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        group_rows: dict[str, list[int]] = {}
        for row_index, group in enumerate(group_ids):
            group_rows.setdefault(group, []).append(row_index)
        self._caption_groups = list(group_rows)
        self._group_rows = list(group_rows.values())
        # Separate generators keep image permutations fixed across group epochs.
        self._image_orders: list[list[int]] = []
        for group_index, rows in enumerate(self._group_rows):
            generator = torch.Generator().manual_seed(
                (seed + 104729 * (group_index + 1)) % (2**63 - 1)
            )
            permutation = torch.randperm(len(rows), generator=generator).tolist()
            self._image_orders.append([rows[index] for index in permutation])
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.batch_offset = 0
        self._rotation_cursors = [0] * len(self._group_rows)
        self._prepare_epoch()

    @property
    def batches_per_epoch(self) -> int:
        return len(self._batch_plan)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _make_plan(
        self, group_order: list[int], start_cursors: list[int]
    ) -> tuple[list[list[int]], list[int]]:
        cursors = list(start_cursors)
        plan: list[list[int]] = []
        batch: list[int] = []
        pending_cursors: dict[int, int] = {}
        for group_index in group_order:
            image_order = self._image_orders[group_index]
            count = min(len(image_order), self.images_per_group, self.batch_size - len(batch))
            cursor = cursors[group_index]
            batch.extend(
                image_order[(cursor + offset) % len(image_order)] for offset in range(count)
            )
            pending_cursors[group_index] = (cursor + count) % len(image_order)
            if len(batch) == self.batch_size:
                plan.append(batch)
                for index, next_cursor in pending_cursors.items():
                    cursors[index] = next_cursor
                batch = []
                pending_cursors = {}
        if batch and not self.drop_last:
            plan.append(batch)
            for index, next_cursor in pending_cursors.items():
                cursors[index] = next_cursor
        return plan, cursors

    def _prepare_epoch(self) -> None:
        group_count = len(self._group_rows)
        self._group_order = (
            torch.randperm(group_count, generator=self.generator).tolist()
            if self.shuffle
            else list(range(group_count))
        )
        self._plan_start_cursors = list(self._rotation_cursors)
        self._batch_plan, self._rotation_cursors = self._make_plan(
            self._group_order, self._plan_start_cursors
        )

    def __iter__(self) -> Iterator[list[int]]:
        # Keep this iterator tied to its original epoch even if advance prepares
        # the next plan while DataLoader still holds prefetched batches.
        plan = self._batch_plan
        for batch_index in range(self.batch_offset, len(plan)):
            yield list(plan[batch_index])

    def advance(self) -> None:
        if self.batch_offset >= self.batches_per_epoch:
            raise RuntimeError("Cannot advance a completed or empty epoch")
        self.batch_offset += 1
        if self.batch_offset == self.batches_per_epoch:
            self.epoch += 1
            self.batch_offset = 0
            self._prepare_epoch()

    def state_dict(self) -> dict[str, Any]:
        return {
            "sampler_type": "caption_group_v1",
            "caption_groups": list(self._caption_groups),
            "group_rows": [list(rows) for rows in self._group_rows],
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "images_per_group": self.images_per_group,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "batch_offset": self.batch_offset,
            "generator_state": self.generator.get_state().clone(),
            "group_order": list(self._group_order),
            "plan_start_cursors": list(self._plan_start_cursors),
            "rotation_cursors": list(self._rotation_cursors),
            "batch_plan": [list(batch) for batch in self._batch_plan],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name, expected in {
            "sampler_type": "caption_group_v1",
            "caption_groups": self._caption_groups,
            "group_rows": self._group_rows,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "images_per_group": self.images_per_group,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
        }.items():
            if state.get(name) != expected:
                raise ValueError(f"Checkpoint caption-group sampler {name} does not match")
        if any(
            not isinstance(rows, list) or any(type(index) is not int for index in rows)
            for rows in state["group_rows"]
        ):
            raise ValueError("Checkpoint caption-group sampler group_rows is invalid")
        epoch = state.get("epoch")
        offset = state.get("batch_offset")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Checkpoint caption-group sampler epoch is invalid")
        group_count = len(self._group_rows)
        order = state.get("group_order")
        if (
            not isinstance(order, list)
            or any(type(index) is not int for index in order)
            or sorted(order) != list(range(group_count))
            or (not self.shuffle and order != list(range(group_count)))
        ):
            raise ValueError("Checkpoint caption-group sampler group_order is invalid")
        start_cursors = state.get("plan_start_cursors")
        cursors = state.get("rotation_cursors")
        for values in (start_cursors, cursors):
            if (
                not isinstance(values, list)
                or len(values) != group_count
                or any(
                    type(cursor) is not int or not 0 <= cursor < len(rows)
                    for cursor, rows in zip(values, self._group_rows, strict=True)
                )
            ):
                raise ValueError("Checkpoint caption-group sampler rotation cursors are invalid")
        expected_plan, expected_cursors = self._make_plan(order, start_cursors)
        stored_plan = state.get("batch_plan")
        if (
            not isinstance(stored_plan, list)
            or any(
                not isinstance(batch, list) or any(type(index) is not int for index in batch)
                for batch in stored_plan
            )
            or stored_plan != expected_plan
            or cursors != expected_cursors
        ):
            raise ValueError("Checkpoint caption-group sampler batch_plan is invalid")
        if type(offset) is not int or not 0 <= offset < max(1, len(expected_plan)):
            raise ValueError("Checkpoint caption-group sampler batch_offset is invalid")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("Checkpoint caption-group sampler generator_state is invalid")
        restored_generator = torch.Generator()
        try:
            restored_generator.set_state(generator_state.cpu())
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                "Checkpoint caption-group sampler generator_state is invalid"
            ) from error
        self.epoch = epoch
        self.batch_offset = offset
        self.generator = restored_generator
        self._group_order = list(order)
        self._plan_start_cursors = list(start_cursors)
        self._rotation_cursors = list(cursors)
        self._batch_plan = [list(batch) for batch in expected_plan]
