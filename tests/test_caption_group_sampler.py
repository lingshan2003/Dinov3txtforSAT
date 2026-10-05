from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest
import torch

from dinotxt_rs.training.sampler import ResumableCaptionGroupBatchSampler


def make_sampler(
    groups: list[str],
    *,
    batch_size: int = 16,
    images_per_group: int = 2,
    shuffle: bool = True,
    seed: int = 23,
    drop_last: bool = True,
) -> ResumableCaptionGroupBatchSampler:
    return ResumableCaptionGroupBatchSampler(
        group_ids=groups,
        batch_size=batch_size,
        images_per_group=images_per_group,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def consume_epoch(sampler: ResumableCaptionGroupBatchSampler) -> list[list[int]]:
    batches = list(sampler)
    for _ in batches:
        sampler.advance()
    return batches


def test_prefetch_does_not_advance_and_exact_resume_crosses_group_epochs() -> None:
    groups = [f"group-{index}" for index in range(33) for _ in range(1 + index % 4)]
    uninterrupted = make_sampler(groups)
    iterator = iter(uninterrupted)
    next(iterator)
    uninterrupted.advance()
    state_before_prefetch = uninterrupted.state_dict()
    list(iterator)  # Workers may fetch every remaining batch before any is consumed.
    state_after_prefetch = uninterrupted.state_dict()
    assert state_before_prefetch["batch_plan"] == state_after_prefetch["batch_plan"]
    assert state_after_prefetch["batch_offset"] == 1
    assert torch.equal(
        state_before_prefetch["generator_state"], state_after_prefetch["generator_state"]
    )
    restored = make_sampler(groups)
    restored.load_state_dict(state_after_prefetch)
    for _ in range(5):
        assert list(uninterrupted) == list(restored)
        assert uninterrupted.epoch == restored.epoch
        assert consume_epoch(uninterrupted) == consume_epoch(restored)


@pytest.mark.parametrize("images_per_group", [1, 2])
def test_images_rotate_without_repeating_a_row_in_a_batch(images_per_group: int) -> None:
    groups = ["airport"] * 5
    sampler = make_sampler(
        groups, batch_size=images_per_group, images_per_group=images_per_group, shuffle=False
    )
    seen = []
    for _ in range(5):
        batch = consume_epoch(sampler)[0]
        assert len(batch) == images_per_group
        assert len(set(batch)) == images_per_group
        seen.extend(batch)
    assert set(seen) == set(range(5))
    assert len(set(Counter(seen).values())) == 1


def test_singletons_fill_physical_batches_and_groups_never_split() -> None:
    # One 2-image group plus fourteen singletons fills the first physical batch.
    groups = ["airport"] * 7 + [f"singleton-{index}" for index in range(30)]
    sampler = make_sampler(groups, shuffle=False)
    batches = list(sampler)
    assert len(sampler) == sampler.batches_per_epoch == 2
    assert all(len(batch) == 16 for batch in batches)
    assert all(len(set(batch)) == 16 for batch in batches)
    first_groups = Counter(groups[index] for index in batches[0])
    assert first_groups["airport"] == 2
    assert len(first_groups) == 15
    selected_rows = [index for batch in batches for index in batch]
    assert len(set(selected_rows)) == len(selected_rows)
    assert {groups[index] for index in selected_rows} == set(groups)


def test_free_slot_truncates_group_without_splitting_it() -> None:
    groups = ["singleton"] + ["airport"] * 3 + ["factory"] * 2
    sampler = make_sampler(groups, batch_size=2, images_per_group=2, shuffle=False)
    batches = list(sampler)
    assert [Counter(groups[index] for index in batch) for batch in batches] == [
        Counter({"singleton": 1, "airport": 1}),
        Counter({"factory": 2}),
    ]


def test_drop_last_does_not_rotate_discarded_tail_and_partial_mode_keeps_it() -> None:
    groups = ["airport"] * 3 + ["tail"] * 4
    sampler = make_sampler(groups, batch_size=3, images_per_group=2, shuffle=False)
    # First group contributes 2, second only 1 to fill the batch.
    assert len(list(sampler)[0]) == 3
    groups = ["a"] * 3 + ["b"] * 3 + ["tail"] * 4
    sampler = make_sampler(groups, batch_size=4, images_per_group=2, shuffle=False)
    state = sampler.state_dict()
    assert state["rotation_cursors"][-1] == state["plan_start_cursors"][-1] == 0
    partial = make_sampler(
        groups, batch_size=4, images_per_group=2, shuffle=False, drop_last=False
    )
    assert [len(batch) for batch in partial] == [4, 2]
    assert partial.state_dict()["rotation_cursors"][-1] == 2


def test_changed_row_mapping_or_seed_is_rejected() -> None:
    groups = ["a", "a", "b", "b"]
    sampler = make_sampler(groups, batch_size=2)
    state = sampler.state_dict()
    changed_mapping = make_sampler(["a", "b", "a", "b"], batch_size=2)
    with pytest.raises(ValueError, match="group_rows"):
        changed_mapping.load_state_dict(state)
    with pytest.raises(ValueError, match="seed"):
        make_sampler(groups, batch_size=2, seed=47).load_state_dict(state)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("group_order", [0, 0, 2]),
        ("caption_groups", ["a", "b", "wrong"]),
        ("group_rows", [[0, 2], [1, 3], [4, 5]]),
        ("batch_plan", [[0, 0], [2, 3], [4, 5]]),
        ("rotation_cursors", [3, 0, 0]),
        ("plan_start_cursors", [-1, 0, 0]),
        ("batch_offset", 3),
        ("epoch", -1),
        ("generator_state", torch.tensor([1, 2], dtype=torch.uint8)),
    ],
)
def test_corrupt_state_is_rejected_without_mutating_sampler(field: str, value: object) -> None:
    sampler = make_sampler(["a", "a", "b", "b", "c", "c"], batch_size=2)
    before = sampler.state_dict()
    invalid = deepcopy(before)
    invalid[field] = value
    with pytest.raises(ValueError):
        sampler.load_state_dict(invalid)
    after = sampler.state_dict()
    assert after["batch_plan"] == before["batch_plan"]
    assert after["epoch"] == before["epoch"]
    assert torch.equal(after["generator_state"], before["generator_state"])


def test_state_is_independent_copy_and_empty_epoch_is_visible() -> None:
    sampler = make_sampler(["a", "a"], batch_size=2)
    state = sampler.state_dict()
    state["batch_plan"][0][0] = 99
    state["rotation_cursors"][0] = 99
    state["caption_groups"][0] = "changed"
    state["group_rows"][0][0] = 99
    assert all(index < 2 for batch in sampler for index in batch)
    empty = make_sampler(["singleton"], batch_size=16, images_per_group=1)
    assert len(empty) == 0
    assert list(empty) == []
    empty.load_state_dict(empty.state_dict())
    with pytest.raises(RuntimeError, match="empty epoch"):
        empty.advance()


def test_checkpoint_mapping_stores_each_caption_once() -> None:
    groups = ["airport with two parallel runways"] * 4 + ["factory beside a river"] * 3
    sampler = make_sampler(groups, batch_size=2)
    state = sampler.state_dict()
    assert "group_ids" not in state
    assert state["caption_groups"] == [
        "airport with two parallel runways", "factory beside a river"
    ]
    assert state["group_rows"] == [[0, 1, 2, 3], [4, 5, 6]]
    restored = make_sampler(groups, batch_size=2)
    restored.load_state_dict(state)
    assert list(restored) == list(sampler)
