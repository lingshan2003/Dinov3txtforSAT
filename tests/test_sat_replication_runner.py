"""Behavioral tests for the multi-seed SAT replication runner.

The runner is copied into an isolated temporary project with a tiny fake
``dinotxt_rs`` package. These tests exercise orchestration and artifact safety
without loading model code, a GPU, or real checkpoint weights.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_sat_replications_3epoch.sh"
TRIALS = tuple(
    f"skyscript_sat_{method}_3epoch_seed{seed}"
    for seed in (23, 47)
    for method in ("adapter_textlora", "adapter", "visionhead_textlora")
)
MANIFESTS = (
    "skyscript_images23_val_raw_unique4055_seed23_global77.jsonl",
    "rsicd_val_retrieval_v1.jsonl",
)


def _fake_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / ".venv/bin").mkdir(parents=True)
    shutil.copy2(RUNNER, project / "scripts/run_sat_replications_3epoch.sh")
    (project / ".venv/bin/python").symlink_to(sys.executable)

    config_dir = project / "configs"
    config_dir.mkdir()
    for trial in TRIALS:
        (config_dir / f"{trial}.toml").write_text(
            f"trial = '{trial}'\n", encoding="utf-8"
        )

    manifest_dir = project / "assets/data/manifests"
    manifest_dir.mkdir(parents=True)
    for manifest in MANIFESTS:
        (manifest_dir / manifest).write_text("fake manifest\n", encoding="utf-8")

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
    return Config(Experiment(Path('outputs') / source.stem), source)

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
with Path(os.environ['FAKE_TRAIN_LOG']).open('a', encoding='utf-8') as stream:
    stream.write(trial + ' ' + ' '.join(args) + '\\n')
if os.environ.get('FAKE_FAIL_TRIAL') == trial:
    print('intentional fake training failure', file=sys.stderr)
    raise SystemExit(7)
root = Path('outputs') / trial
root.mkdir(parents=True, exist_ok=True)
(root / 'config.toml').write_bytes(config.read_bytes())
for name in ('step_0000000.pt', 'best.pt', 'latest.pt'):
    (root / name).write_bytes(b'fake checkpoint')
(root / 'training_summary.json').write_text(json.dumps({
    'completed': True,
    'steps': 1710,
    'target_steps': 1710,
    'validation': {'last_step': 1710, 'best_step': 900, 'best_loss': 1.25},
}), encoding='utf-8')
""",
        encoding="utf-8",
    )
    evaluate = """import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
def arg(name):
    return args[args.index(name) + 1]
config = Path(arg('--config'))
root = Path(arg('--training-output'))
manifest = Path(arg('--manifest'))
checkpoint = Path(arg('--checkpoint'))
dataset = 'skyscript' if 'skyscript' in manifest.name else 'rsicd'
tag = checkpoint.stem
with Path(os.environ['FAKE_EVAL_LOG']).open('a', encoding='utf-8') as stream:
    stream.write(f'{root.name} {dataset} {tag}\\n')
metrics = {
    'image_to_text': {'r1': 0.1, 'r5': 0.2, 'r10': 0.3},
    'text_to_image': {'r1': 0.4, 'r5': 0.5, 'r10': 0.6},
    'mean_recall': 0.35,
}
report = {
    'split': 'val',
    'task': ('paired_image_text_global_retrieval' if dataset == 'skyscript'
             else 'rsicd_image_text_retrieval'),
    'counts': ({'images': 4055, 'captions': 4055} if dataset == 'skyscript'
               else {'images': 1094, 'captions': 5470}),
    'manifest': {'path': str(manifest)},
    'model': {
        'checkpoint': {
            'step': 0 if tag == 'step_0000000' else 900,
            'path': str(checkpoint),
            'identity_check': {'status': 'match', 'blocking_mismatches': []},
        },
        'model_config': str(config),
    },
    'tie_policy': 'score_descending_then_candidate_index_ascending',
    'metrics': metrics,
}
Path(arg('--output')).write_text(json.dumps(report), encoding='utf-8')
"""
    (package / "cli/evaluate_skyscript.py").write_text(evaluate, encoding="utf-8")
    (package / "cli/evaluate_rsicd.py").write_text(evaluate, encoding="utf-8")
    return project, tmp_path / "train_calls.log", tmp_path / "eval_calls.log"


def _run(
    project: Path,
    train_log: Path,
    eval_log: Path,
    *args: str,
    **env_overrides: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(project / "fake_lib"),
        FAKE_TRAIN_LOG=str(train_log),
        FAKE_EVAL_LOG=str(eval_log),
    )
    env.update(env_overrides)
    return subprocess.run(
        ["bash", "scripts/run_sat_replications_3epoch.sh", *args],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_default_trains_all_seeds_then_evaluates_and_packages_reports(tmp_path: Path) -> None:
    project, train_log, eval_log = _fake_project(tmp_path)

    result = _run(project, train_log, eval_log)

    assert result.returncode == 0, result.stderr
    calls = _lines(train_log)
    assert [line.split()[0] for line in calls] == list(TRIALS)
    assert all("--resume" not in line for line in calls)
    evaluations = _lines(eval_log)
    assert len(evaluations) == 24
    assert evaluations == [
        f"{trial} {dataset} {tag}"
        for trial in TRIALS
        for dataset in ("skyscript", "rsicd")
        for tag in ("step_0000000", "best")
    ]
    for trial in TRIALS:
        run_dir = project / "outputs" / trial
        reports = list(run_dir.glob("*_val_*.json"))
        assert len(reports) == 4
    archive = project / "outputs/sat_replications_3epoch_reports.tar.gz"
    assert archive.is_file()
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert any(name.endswith("skyscript_val_step_0000000.json") for name in names)
    assert not any(name.endswith(".pt") for name in names)


def test_train_only_then_default_adds_reports_and_later_skips_all(tmp_path: Path) -> None:
    project, train_log, eval_log = _fake_project(tmp_path)

    train_only = _run(project, train_log, eval_log, "--train-only")
    assert train_only.returncode == 0, train_only.stderr
    assert [line.split()[0] for line in _lines(train_log)] == list(TRIALS)
    assert _lines(eval_log) == []
    assert not list((project / "outputs").glob("*/skyscript_val_*.json"))

    evaluate = _run(project, train_log, eval_log)
    assert evaluate.returncode == 0, evaluate.stderr
    assert [line.split()[0] for line in _lines(train_log)] == list(TRIALS)
    assert len(_lines(eval_log)) == 24

    rerun = _run(project, train_log, eval_log)
    assert rerun.returncode == 0, rerun.stderr
    assert [line.split()[0] for line in _lines(train_log)] == list(TRIALS)
    assert len(_lines(eval_log)) == 24
    assert rerun.stdout.count("Already completed:") == len(TRIALS)
    assert rerun.stdout.count("Already evaluated:") == 24


def test_latest_checkpoint_resumes_and_training_failure_preserves_old_archive(
    tmp_path: Path,
) -> None:
    project, train_log, eval_log = _fake_project(tmp_path)
    first, second = TRIALS[:2]
    first_dir = project / "outputs" / first
    first_dir.mkdir(parents=True)
    (first_dir / "latest.pt").write_text("interrupted checkpoint", encoding="utf-8")
    archive = project / "outputs/sat_replications_3epoch_reports.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"old archive")

    result = _run(project, train_log, eval_log, FAKE_FAIL_TRIAL=second)

    assert result.returncode != 0
    calls = _lines(train_log)
    assert [line.split()[0] for line in calls] == [first, second]
    assert "--resume outputs/" + first + "/latest.pt" in calls[0]
    assert _lines(eval_log) == []
    assert archive.read_bytes() == b"old archive"
    assert not (project / "outputs" / TRIALS[2]).exists()


@pytest.mark.parametrize(
    "bad_report",
    ["checkpoint_step", "malformed"],
)
def test_bad_existing_report_is_neither_skipped_nor_overwritten(
    tmp_path: Path, bad_report: str
) -> None:
    project, train_log, eval_log = _fake_project(tmp_path)
    trial = TRIALS[0]
    # Completed outputs for every trial isolate report validation from training.
    for completed_trial in TRIALS:
        completed_dir = project / "outputs" / completed_trial
        completed_dir.mkdir(parents=True)
        (completed_dir / "config.toml").write_bytes(
            (project / "configs" / f"{completed_trial}.toml").read_bytes()
        )
        (completed_dir / "training_summary.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "steps": 1710,
                    "target_steps": 1710,
                    "validation": {
                        "last_step": 1710,
                        "best_step": 900,
                        "best_loss": 1.25,
                    },
                }
            ),
            encoding="utf-8",
        )
        for name in ("step_0000000.pt", "best.pt", "latest.pt"):
            (completed_dir / name).write_text("checkpoint", encoding="utf-8")

    run_dir = project / "outputs" / trial
    report = run_dir / "skyscript_val_step_0000000.json"
    if bad_report == "checkpoint_step":
        report_contents = json.dumps({
            "split": "val",
            "task": "paired_image_text_global_retrieval",
            "counts": {"images": 4055, "captions": 4055},
            "manifest": {"path": str(
                project / "assets/data/manifests" / MANIFESTS[0]
            )},
            "model": {
                "checkpoint": {
                    "step": 9,
                    "path": str(run_dir / "step_0000000.pt"),
                    "identity_check": {
                        "status": "match", "blocking_mismatches": []
                    },
                },
                "model_config": str(run_dir / "config.toml"),
            },
            "tie_policy": "score_descending_then_candidate_index_ascending",
            "metrics": {
                "image_to_text": {"r1": 0.1, "r5": 0.2, "r10": 0.3},
                "text_to_image": {"r1": 0.4, "r5": 0.5, "r10": 0.6},
                "mean_recall": 0.35,
            },
        })
    else:
        report_contents = "{ malformed json"
    report.write_text(report_contents, encoding="utf-8")
    archive = project / "outputs/sat_replications_3epoch_reports.tar.gz"
    archive.write_bytes(b"previous archive")

    result = _run(project, train_log, eval_log)

    assert result.returncode != 0
    assert _lines(train_log) == []
    assert _lines(eval_log) == []
    assert report.read_text(encoding="utf-8") == report_contents
    assert archive.read_bytes() == b"previous archive"


def test_preflight_rejects_config_mismatch_or_missing_rs_manifest_before_training(
    tmp_path: Path,
) -> None:
    project, train_log, eval_log = _fake_project(tmp_path)
    snapshot = project / "outputs" / TRIALS[0] / "config.toml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("trial = 'mismatched'\n", encoding="utf-8")

    mismatch = _run(project, train_log, eval_log)
    assert mismatch.returncode != 0
    assert "Saved configuration differs" in mismatch.stderr
    assert _lines(train_log) == []
    assert _lines(eval_log) == []

    snapshot.unlink()
    snapshot.parent.rmdir()
    (project / "assets/data/manifests" / MANIFESTS[1]).unlink()
    missing = _run(project, train_log, eval_log)
    assert missing.returncode != 0
    assert MANIFESTS[1] in missing.stderr
    assert _lines(train_log) == []
    assert _lines(eval_log) == []
