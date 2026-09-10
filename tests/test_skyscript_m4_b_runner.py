import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.run_skyscript_m4_b import M4BRun

RUNNER = Path("scripts/run_skyscript_m4_b.sh")
MATRIX_RUNNER = Path("scripts/run_skyscript_m4_b_stage100.sh")
FULL_RUNNER = Path("scripts/run_skyscript_m4_b_full.sh")
TOOL = Path("tools/run_skyscript_m4_b.py")


def _args(domain: str = "sat", seed: int = 23) -> argparse.Namespace:
    return argparse.Namespace(
        domain=domain,
        seed=seed,
        stop_after_stage=100,
        rsicd_root=Path("assets/data/raw/rsicd"),
        batch_size=64,
        num_workers=4,
        chunk_size=256,
        skip_code_checks=False,
    )


def test_m4_b_runners_have_valid_bash_syntax_and_complete_matrix() -> None:
    for script in (RUNNER, MATRIX_RUNNER, FULL_RUNNER):
        subprocess.run(["bash", "-n", str(script)], check=True)
    source = MATRIX_RUNNER.read_text(encoding="utf-8")
    assert "web:23 sat:23 web:47 sat:47" in source
    assert "--stop-after-stage 100" in source
    assert "tools/summarize_skyscript_m4_b.py" in source


def test_m4_b_full_runner_automates_all_matrix_gates() -> None:
    source = FULL_RUNNER.read_text(encoding="utf-8")
    assert "for stage in 100 250 500" in source
    assert "web:23 sat:23 web:47 sat:47" in source
    assert '--stop-after-stage "${stage}"' in source
    assert 'm4_b_stage${stage}/summary.json' in source
    assert "No run will advance to the next stage" in source


def test_m4_b_runner_help_does_not_start_work() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--domain" in result.stdout
    assert "--stop-after-stage" in result.stdout


def test_m4_b_run_paths_are_domain_and_seed_specific(tmp_path: Path) -> None:
    run = M4BRun(_args(), tmp_path)
    assert run.config == Path("configs/skyscript_sat_adapter_500step_seed23.toml")
    assert run.run_dir.name.endswith("sat_imageadapter256_36495_500step_seed23")
    assert run.gate_dir == Path("outputs/skyscript_gate_m4_b_sat_seed23")


def test_current_step_rejects_partial_training_artifacts(tmp_path: Path) -> None:
    run = M4BRun(_args(), tmp_path)
    run.run_dir = tmp_path / "run"
    run.run_dir.mkdir()
    (run.run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Partial M4-B artifacts"):
        run.current_step()


def test_current_step_accepts_only_registered_boundaries(tmp_path: Path) -> None:
    run = M4BRun(_args(), tmp_path)
    run.run_dir = tmp_path / "run"
    run.run_dir.mkdir()
    (run.run_dir / "training_summary.json").write_text(
        json.dumps({"steps": 250, "completed": False, "target_steps": 500}),
        encoding="utf-8",
    )
    assert run.current_step() == 250
