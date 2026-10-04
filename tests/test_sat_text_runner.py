"""Behavioral tests for the sequential text-control runner.

The runner is copied into a temporary project with a tiny fake Python package.
No model code, GPU, checkpoint weights, or project outputs are touched.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_sat_text_3epoch_seed11.sh"
TRIALS = (
    "skyscript_sat_textproj_3epoch_seed11",
    "skyscript_sat_textlast2_3epoch_seed11",
    "skyscript_sat_textlora_3epoch_seed11",
)


@pytest.fixture(autouse=True, params=["text", "joint"])
def runner_variant(request, monkeypatch):
    if request.param == "joint":
        module = sys.modules[__name__]
        monkeypatch.setattr(module, "RUNNER", ROOT / "scripts/run_sat_joint_3epoch_seed11.sh")
        monkeypatch.setattr(module, "TRIALS", tuple(
            f"skyscript_sat_{name}_3epoch_seed11" for name in (
                "adapter_textlora", "visionhead_textlora", "adapter_textproj", "visionhead_textproj"
            )
        ))


def _fake_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / ".venv/bin").mkdir(parents=True)
    shutil.copy2(RUNNER, project / "scripts/run_sat_text_3epoch_seed11.sh")
    python = project / ".venv/bin/python"
    python.symlink_to(sys.executable)

    for trial in TRIALS:
        config = project / "configs" / f"{trial}.toml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"trial = '{trial}'\n", encoding="utf-8")

    package = project / "fake_lib/dinotxt_rs"
    (package / "cli").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli/__init__.py").write_text("", encoding="utf-8")
    (package / "config.py").write_text(
        """from dataclasses import dataclass
from pathlib import Path

@dataclass
class Experiment:
    output_dir: Path

@dataclass
class Config:
    experiment: Experiment
    source: Path

def load_config(path):
    source = Path(path)
    trial = source.stem
    return Config(Experiment(Path('outputs') / trial), source)

def required_paths(config):
    return []
""",
        encoding="utf-8",
    )
    (package / "cli/train.py").write_text(
        """import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
config = Path(args[args.index('--config') + 1])
trial = config.stem
log = Path(os.environ['FAKE_TRAIN_LOG'])
with log.open('a', encoding='utf-8') as stream:
    stream.write(trial + ' ' + ' '.join(args) + '\\n')
if os.environ.get('FAKE_FAIL_TRIAL') == trial:
    print('intentional fake training failure', file=sys.stderr)
    raise SystemExit(7)
root = Path('outputs') / trial
root.mkdir(parents=True, exist_ok=True)
for name in ('step_0000000.pt', 'best.pt', 'latest.pt'):
    (root / name).write_text('fake checkpoint', encoding='utf-8')
(root / 'training_summary.json').write_text(json.dumps({
    'completed': True,
    'steps': 1710,
    'target_steps': 1710,
    'validation': {'best_step': 10, 'best_loss': 1.25},
}), encoding='utf-8')
""",
        encoding="utf-8",
    )
    return project, tmp_path / "train_calls.log"


def _run(project: Path, log: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "fake_lib")
    env["FAKE_TRAIN_LOG"] = str(log)
    env.update(env_overrides)
    return subprocess.run(
        ["bash", "scripts/run_sat_text_3epoch_seed11.sh"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _calls(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def test_runs_all_controls_in_order(tmp_path: Path) -> None:
    project, log = _fake_project(tmp_path)

    result = _run(project, log)

    assert result.returncode == 0, result.stderr
    calls = _calls(log)
    assert [line.split()[0] for line in calls] == list(TRIALS)
    assert all("--resume" not in line for line in calls)


def test_latest_checkpoint_resumes_and_completed_run_is_skipped(tmp_path: Path) -> None:
    project, log = _fake_project(tmp_path)
    first, second = TRIALS[:2]
    first_dir = project / "outputs" / first
    first_dir.mkdir(parents=True)
    (first_dir / "latest.pt").write_text("fake latest", encoding="utf-8")
    second_dir = project / "outputs" / second
    second_dir.mkdir(parents=True)
    (second_dir / "training_summary.json").write_text(
        json.dumps({"completed": True, "steps": 1710, "target_steps": 1710}),
        encoding="utf-8",
    )
    for name in ("step_0000000.pt", "best.pt", "latest.pt"):
        (second_dir / name).write_text("fake checkpoint", encoding="utf-8")

    result = _run(project, log)

    assert result.returncode == 0, result.stderr
    calls = _calls(log)
    assert [line.split()[0] for line in calls] == [first, *TRIALS[2:]]
    assert "--resume outputs/" + first + "/latest.pt" in calls[0]
    assert "Already completed: " + second in result.stdout


def test_training_failure_stops_before_following_trials(tmp_path: Path) -> None:
    project, log = _fake_project(tmp_path)

    result = _run(project, log, FAKE_FAIL_TRIAL=TRIALS[0])

    assert result.returncode != 0
    assert [line.split()[0] for line in _calls(log)] == [TRIALS[0]]


def test_existing_output_without_latest_is_preserved_and_rejected(tmp_path: Path) -> None:
    project, log = _fake_project(tmp_path)
    run_dir = project / "outputs" / TRIALS[0]
    run_dir.mkdir(parents=True)
    artifact = run_dir / "keep-me.txt"
    artifact.write_text("existing result", encoding="utf-8")

    result = _run(project, log)

    assert result.returncode != 0
    assert "no latest.pt" in result.stderr
    assert artifact.read_text(encoding="utf-8") == "existing result"
    assert _calls(log) == []


def test_mismatched_saved_config_is_rejected_before_training(tmp_path: Path) -> None:
    project, log = _fake_project(tmp_path)
    snapshot = project / "outputs" / TRIALS[0] / "config.toml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("trial = 'different'\n", encoding="utf-8")

    result = _run(project, log)

    assert result.returncode != 0
    assert "Saved configuration differs" in result.stderr
    assert _calls(log) == []
