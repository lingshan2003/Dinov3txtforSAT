import subprocess
from pathlib import Path

RUNNER = Path("scripts/run_skyscript_m4_sat_seed11.sh")


def test_m4_sat_runner_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_m4_sat_runner_defaults_to_stage100_and_uses_formal_paths() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "STOP_AFTER_STAGE=100" in source
    assert 'CONFIG="configs/skyscript_sat_adapter_500step_seed11.toml"' in source
    assert (
        'RUN_DIR="outputs/skyscript_images23_top30raw_sat_'
        'imageadapter256_36495_500step_seed11"'
    ) in source
    assert 'GATE_DIR="outputs/skyscript_gate_m4_sat_seed11"' in source
    assert "tools/summarize_skyscript_gate_m4_sat.py stage" in source


def test_m4_sat_runner_help_is_available_without_environment() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "default: 100" in result.stdout
    assert "never loads a Web adapter" in result.stdout
