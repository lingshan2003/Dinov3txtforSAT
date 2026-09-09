from copy import deepcopy
from pathlib import Path

import pytest

from dinotxt_rs.config import load_config
from tools.verify_skyscript_m4_config_pair import load_toml, verify_config_pair

WEB_CONFIG = Path("configs/skyscript_web_adapter_500step_seed11.toml")
SAT_CONFIG = Path("configs/skyscript_sat_adapter_500step_seed11.toml")


def test_repository_m4_a_configs_match_frozen_protocol() -> None:
    verify_config_pair(load_toml(WEB_CONFIG), load_toml(SAT_CONFIG))

    web = load_config(WEB_CONFIG)
    sat = load_config(SAT_CONFIG)
    assert sat.experiment.seed == web.experiment.seed == 11
    assert sat.model.backbone_domain == "sat"
    assert sat.model.image_adapter_bottleneck == web.model.image_adapter_bottleneck == 256
    assert sat.data == web.data
    assert sat.train == web.train


def test_m4_a_protocol_rejects_an_unapproved_training_change() -> None:
    web = load_toml(WEB_CONFIG)
    sat = deepcopy(load_toml(SAT_CONFIG))
    sat["train"]["learning_rate"] = 5e-5

    with pytest.raises(ValueError, match=r"unexpected=\['train.learning_rate'\]"):
        verify_config_pair(web, sat)


def test_m4_a_protocol_rejects_wrong_sat_identity() -> None:
    web = load_toml(WEB_CONFIG)
    sat = deepcopy(load_toml(SAT_CONFIG))
    sat["model"]["backbone_weights"] = "assets/checkpoints/wrong-sat-weights.pth"

    with pytest.raises(ValueError, match="Unexpected sat model.backbone_weights"):
        verify_config_pair(web, sat)
