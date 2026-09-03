from pathlib import Path

from dinotxt_rs.config import load_config


def test_load_mvp_config() -> None:
    config = load_config(Path("configs/train_mvp_web.toml"))
    assert config.model.backbone_domain == "web"
    assert config.model.image_size == 224
    assert config.model.text_last_k == 4
    assert config.data.train_manifest.name == "chatearthnet_35_train_10k_seed11_no_nodata.jsonl"
    assert config.data.val_manifest is not None
    assert config.data.val_manifest.name == "chatearthnet_35_val_no_nodata.jsonl"
    assert config.train.queue_size == 4096
