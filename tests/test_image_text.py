import pytest

try:
    import torch  # noqa: F401
except ImportError:
    pytest.skip("PyTorch is unavailable in this interpreter", allow_module_level=True)

from dinotxt_rs.data.image_text import make_transform


def test_training_transform_does_not_flip_directional_captions() -> None:
    transform = make_transform(image_size=224, backbone_domain="sat", train=True)
    names = {type(item).__name__ for item in transform.transforms}
    assert "RandomHorizontalFlip" not in names
    assert "RandomVerticalFlip" not in names
