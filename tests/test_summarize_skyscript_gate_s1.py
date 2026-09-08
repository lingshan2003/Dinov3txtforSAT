import json
from pathlib import Path

import pytest

from tools.summarize_skyscript_gate_s1 import summarize_seed_results

STEPS = (0, 25, 50, 75, 100)


def _metrics(mean_recall: float) -> dict:
    return {
        "image_to_text": {
            "r1": mean_recall,
            "r5": mean_recall,
            "r10": mean_recall,
            "median_rank": 10.0,
            "mean_rank": 20.0,
        },
        "text_to_image": {
            "r1": mean_recall,
            "r5": mean_recall,
            "r10": mean_recall,
            "median_rank": 11.0,
            "mean_rank": 21.0,
        },
        "mean_recall": mean_recall,
    }


def _seed_result(
    root: Path,
    seed: int,
    *,
    learning_rate: float = 0.0001,
    sky_improves: bool = True,
):
    run = root / f"run-{seed}"
    run.mkdir()
    (run / "config.toml").write_text(
        f"""[experiment]
name = "run-{seed}"
seed = {seed}
output_dir = "outputs/run-{seed}"
[model]
adapter = 256
[data]
train_manifest = "train.jsonl"
val_manifest = "val.jsonl"
[train]
max_steps = 100
learning_rate = {learning_rate}
""",
        encoding="utf-8",
    )
    losses = {0: 1.5, 25: 1.2, 50: 1.0, 75: 0.9, 100: 0.8 + seed / 10000}
    (run / "validation.jsonl").write_text(
        "".join(json.dumps({"step": step, "loss": losses[step]}) + "\n" for step in STEPS),
        encoding="utf-8",
    )
    (run / "verification_report.json").write_text(
        json.dumps(
            {
                "completed": True,
                "steps": 100,
                "best_validation_step": 100,
                "project_commit": f"commit-{seed}",
            }
        ),
        encoding="utf-8",
    )
    sky_direction = 1 if sky_improves else -1
    sky = {
        str(step): _metrics(0.1 + sky_direction * step / 10000 + seed / 100000)
        for step in STEPS
    }
    official = _metrics(0.16)
    rsicd = {str(step): _metrics(0.16 + step / 100000) for step in STEPS}
    gate = root / f"gate-{seed}.json"
    gate.write_text(
        json.dumps(
            {
                "status": "pass" if sky_improves else "fail",
                "checks": {
                    "training_artifacts": True,
                    "train_validation_overlap": True,
                    "step0_parity": True,
                    "training_rsicd_val_overlap": True,
                    "skyscript_step100_mean_recall_above_step0": sky_improves,
                    "rsicd_step100_mean_recall_drop_within_0.01": True,
                },
                "skyscript": {"steps": sky},
                "rsicd_val": {"official": official, "steps": rsicd},
            }
        ),
        encoding="utf-8",
    )
    return seed, run, gate


def test_summarize_seed_results_reports_per_seed_and_sample_std(tmp_path: Path) -> None:
    specs = [_seed_result(tmp_path, seed) for seed in (11, 23, 47)]

    report = summarize_seed_results(specs)

    assert report["status"] == "pass"
    assert list(report["per_seed"]) == ["11", "23", "47"]
    assert report["per_seed"]["23"]["criteria"]["validation_best_not_step0"]
    aggregate = report["aggregate"]
    assert aggregate["validation_loss_by_step"]["100"]["sample_std"] > 0
    assert aggregate["skyscript_step100_minus_step0_mean_recall"]["mean"] > 0
    assert aggregate["rsicd_step100_minus_official_mean_recall"]["mean"] > 0


def test_summarize_seed_results_rejects_protocol_drift(tmp_path: Path) -> None:
    specs = [
        _seed_result(tmp_path, 11),
        _seed_result(tmp_path, 23, learning_rate=0.0002),
        _seed_result(tmp_path, 47),
    ]

    with pytest.raises(ValueError, match="outside experiment"):
        summarize_seed_results(specs)


def test_summarize_seed_results_preserves_numerical_failure(tmp_path: Path) -> None:
    specs = [
        _seed_result(tmp_path, 11),
        _seed_result(tmp_path, 23, sky_improves=False),
        _seed_result(tmp_path, 47),
    ]

    report = summarize_seed_results(specs)

    assert report["status"] == "fail"
    assert report["per_seed"]["23"]["status"] == "fail"
    assert not report["per_seed"]["23"]["criteria"][
        "skyscript_step100_mean_recall_above_step0"
    ]
