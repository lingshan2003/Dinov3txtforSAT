from pathlib import Path

from dinotxt_rs.config import load_config


def test_load_mvp_config() -> None:
    config = load_config(Path("configs/train_mvp_web.toml"))
    assert config.model.backbone_domain == "web"
    assert config.model.image_size == 224
    assert config.model.text_last_k == 4
    assert config.train.queue_size == 4096

