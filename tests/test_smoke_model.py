import pytest

from dinotxt_rs.cli.smoke_model import assert_shapes


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)

    def numel(self) -> int:
        total = 1
        for size in self.shape:
            total *= size
        return total


def test_smoke_shape_contract() -> None:
    assert_shapes(
        FakeTensor((2, 2048)),
        FakeTensor((2, 2048)),
        FakeTensor(()),
        FakeTensor((2, 196, 1024)),
        FakeTensor((2, 196, 1024)),
        batch_size=2,
    )


def test_smoke_shape_contract_rejects_wrong_feature_dimension() -> None:
    with pytest.raises(RuntimeError, match="image feature shape"):
        assert_shapes(
            FakeTensor((2, 1024)),
            FakeTensor((2, 2048)),
            FakeTensor(()),
            FakeTensor((2, 196, 1024)),
            FakeTensor((2, 196, 1024)),
            batch_size=2,
        )
