from copy import deepcopy
from pathlib import Path

import pytest

from dinotxt_rs.config import load_config
from tools.verify_skyscript_m4_b_configs import load_config_matrix, verify_config_matrix


def test_repository_m4_b_config_matrix_matches_frozen_protocol() -> None:
    configs = load_config_matrix(Path("configs"))
    verify_config_matrix(configs)
    for seed in (23, 47):
        web = load_config(Path(f"configs/skyscript_web_adapter_500step_seed{seed}.toml"))
        sat = load_config(Path(f"configs/skyscript_sat_adapter_500step_seed{seed}.toml"))
        assert web.experiment.seed == sat.experiment.seed == seed
        assert web.data == sat.data
        assert web.train == sat.train
        assert web.train.max_steps == 500
        assert web.train.warmup_steps == 50


def test_m4_b_matrix_rejects_seed_specific_training_change() -> None:
    configs = load_config_matrix(Path("configs"))
    mutated = {key: deepcopy(value) for key, value in configs.items()}
    mutated[("sat", 47)]["train"]["learning_rate"] = 5e-5
    with pytest.raises(ValueError, match="within-domain differences"):
        verify_config_matrix(mutated)
