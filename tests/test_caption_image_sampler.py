from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from dinotxt_rs.training.sampler import ResumableCaptionImageBatchSampler


def make_sampler(
    groups: list[str], *, batch_size: int = 4, images_per_group: int = 2,
    shuffle: bool = True, seed: int = 23,
) -> ResumableCaptionImageBatchSampler:
    return ResumableCaptionImageBatchSampler(
        group_ids=groups,
        batch_size=batch_size,
        images_per_group=images_per_group,
        shuffle=shuffle,
        seed=seed,
    )


def consume_epoch(sampler: ResumableCaptionImageBatchSampler) -> list[list[int]]:
    batches = list(sampler)
    for _ in batches:
        sampler.advance()
    return batches


@pytest.mark.parametrize(
    "groups",
    [
        ["a", "a", "b", "b", "c", "c", "d", "d", "e"],
        ["one-large-group"] * 101,
        [f"singleton-{index}" for index in range(37)],
    ],
)
def test_each_epoch_visits_every_image_once_and_retains_partial_tail(groups: list[str]) -> None:
    sampler = make_sampler(groups, batch_size=4, shuffle=True)
    batches = consume_epoch(sampler)
    flattened = [index for batch in batches for index in batch]
    assert sampler.drop_last is False
    assert len(batches) == (len(groups) + 3) // 4
    assert sorted(flattened) == list(range(len(groups)))
    assert len(flattened) == len(set(flattened))
    assert len(batches[-1]) == len(groups) % 4 or len(batches[-1]) == 4
    assert sampler.epoch == 1
    assert sampler.batch_offset == 0


def test_chunks_are_flattened_before_physical_batch_boundaries() -> None:
    groups = ["a"] * 7 + ["b"] * 5
    sampler = make_sampler(groups, batch_size=3, shuffle=False)
    batches = list(sampler)
    flattened = [row for batch in batches for row in batch]
    assert flattened == list(range(len(groups)))
    # The first same-caption pair straddles the first physical batch boundary.
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    assert all(len(batch) <= 3 for batch in batches)


def test_prefetch_does_not_advance_and_checkpoint_resumes_exactly_across_epochs() -> None:
    groups = [f"group-{index % 11}" for index in range(47)]
    uninterrupted = make_sampler(groups)
    iterator = iter(uninterrupted)
    first = next(iterator)
    uninterrupted.advance()
    consumed_state = uninterrupted.state_dict()
    list(iterator)  # Simulate DataLoader workers prefetching the remaining batches.
    after_prefetch = uninterrupted.state_dict()
    assert after_prefetch["batch_offset"] == 1
    assert torch.equal(after_prefetch["order"], consumed_state["order"])
    assert torch.equal(after_prefetch["generator_state"], consumed_state["generator_state"])

    restored = make_sampler(groups)
    restored.load_state_dict(after_prefetch)
    assert list(uninterrupted) == list(restored)
    assert first not in list(restored)  # The restored iterator starts after the consumed batch.
    for _ in range(4):
        assert consume_epoch(uninterrupted) == consume_epoch(restored)
        assert uninterrupted.epoch == restored.epoch


@pytest.mark.parametrize(
    "changed",
    [
        ["a", "b", "a", "b", "c", "c"],
        ["a", "a", "b", "b", "c", "d"],
    ],
)
def test_changed_group_mapping_is_rejected(changed: list[str]) -> None:
    state = make_sampler(["a", "a", "b", "b", "c", "c"]).state_dict()
    with pytest.raises(ValueError, match="group"):
        make_sampler(changed).load_state_dict(state)


def test_changed_images_per_group_is_rejected() -> None:
    groups = ["a", "a", "a", "b", "b", "b"]
    state = make_sampler(groups, images_per_group=2).state_dict()
    with pytest.raises(ValueError, match="images_per_group"):
        make_sampler(groups, images_per_group=1).load_state_dict(state)


def test_corrupt_state_is_rejected_without_mutating_sampler() -> None:
    sampler = make_sampler(["a", "a", "b", "b", "c", "c"])
    before = sampler.state_dict()
    invalid = deepcopy(before)
    invalid["batch_offset"] = sampler.batches_per_epoch
    with pytest.raises(ValueError):
        sampler.load_state_dict(invalid)
    after = sampler.state_dict()
    assert after["epoch"] == before["epoch"]
    assert after["batch_offset"] == before["batch_offset"]
    if before["order"] is None:
        assert after["order"] is None
    else:
        assert torch.equal(after["order"], before["order"])
    assert torch.equal(after["generator_state"], before["generator_state"])
