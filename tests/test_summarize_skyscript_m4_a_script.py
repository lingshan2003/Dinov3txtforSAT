import subprocess
from pathlib import Path

SCRIPT = Path("scripts/summarize_skyscript_m4_a_seed11.sh")


def test_m4_a_summary_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_m4_a_summary_script_uses_frozen_inputs_and_output() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'WEB_SUMMARY="outputs/skyscript_gate_s2_seed11/summary.json"' in source
    assert 'SAT_SUMMARY="outputs/skyscript_gate_m4_sat_seed11/summary.json"' in source
    assert 'OUTPUT="outputs/skyscript_m4_rq1_seed11/summary.json"' in source
    assert "tools/summarize_skyscript_m4_rq1_seed11.py" in source
