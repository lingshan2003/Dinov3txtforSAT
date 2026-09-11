import json
import subprocess
import sys
from pathlib import Path

from tools.summarize_skyscript_m4_f3_sat import CANDIDATES, summarize
from tools.verify_skyscript_m4_f3_configs import verify

RUNNER = Path("scripts/run_skyscript_m4_f3_sat_seed11.sh")


def test_f3_sat_configs_match_f0_and_freeze_official_vision_modules() -> None:
    configs = verify()
    assert set(configs) == set(CANDIDATES)
    assert all(config.model.backbone_domain == "sat" for config in configs.values())
    assert all(config.model.image_adapter_bottleneck == 256 for config in configs.values())
    assert all(config.model.train_text_projection for config in configs.values())
    assert all(not config.model.train_vision_head for config in configs.values())
    assert all(config.model.text_last_k == 0 for config in configs.values())


def test_f3_runner_is_one_shot_reuses_f0_and_measures_text_drift() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    source = RUNNER.read_text(encoding="utf-8")
    assert 'F0_SUMMARY="outputs/skyscript_gate_m4_sat_seed11/summary.json"' in source
    assert "f0=verified_reused_not_retrained" in source
    assert "--required-optimizer-group image_adapter" in source
    assert "--required-optimizer-group text_projection" in source
    assert "evaluate_text_drift" in source
    assert "tools/summarize_skyscript_m4_f3_sat.py" in source


def test_f3_runner_help_needs_no_server_assets() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "all the way through 500 steps" in result.stdout
    assert "visual backbone and official dino.txt vision head remain" in result.stdout


def test_f3_config_verifier_cli() -> None:
    result = subprocess.run(
        [sys.executable, "tools/verify_skyscript_m4_f3_configs.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "m4_f3_sat_config_protocol=verified" in result.stdout


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evaluation(step: int | None, mean_recall: float) -> dict:
    return {
        "model": {
            "backbone_domain": "sat",
            "checkpoint": None if step is None else {"step": step},
        },
        "metrics": {"mean_recall": mean_recall},
    }


def _drift_report() -> dict:
    distribution = {
        "mean": 0.001,
        "std": 0.0,
        "min": 0.001,
        "p05": 0.001,
        "median": 0.001,
        "p95": 0.001,
        "max": 0.001,
    }
    aggregate = {
        "captions": 2,
        "embedding_dim": 4,
        "cosine_similarity": distribution,
        "cosine_distance": distribution,
        "l2_distance": distribution,
        "fraction_cosine_distance_above": {"1e-4": 1.0, "1e-3": 0.0, "1e-2": 0.0},
    }
    return {
        "task": "text_embedding_drift_against_frozen_official_reference",
        "reference_definition": "test reference",
        "model": {"backbone_domain": "sat"},
        "manifests": {"test": {"captions": 2}},
        "checkpoints": {
            str(step): {
                "checkpoint": {"step": step},
                "aggregate": aggregate,
                "datasets": {"test": aggregate},
            }
            for step in (0, 100, 250, 500)
        },
    }


def test_f3_summary_applies_preregistered_f0_comparison(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    baseline = tmp_path / "f0.json"
    _write_json(
        baseline,
        {
            "gate": "SkyScript_M4_SAT",
            "status": "pass",
            "stages": [100, 250, 500],
            "validation_loss": {
                "steps": {"0": 4.0, "100": 2.0, "250": 1.5, "500": 1.2}
            },
            "skyscript": {
                "steps": {
                    "100": {"mean_recall": 0.04},
                    "250": {"mean_recall": 0.05},
                    "500": {"mean_recall": 0.06},
                }
            },
            "rsicd_val": {
                "steps": {
                    "100": {"mean_recall": 0.04},
                    "250": {"mean_recall": 0.05},
                    "500": {"mean_recall": 0.06},
                }
            },
        },
    )
    for index, candidate in enumerate(CANDIDATES):
        run_dir = outputs / f"skyscript_sat_{candidate}_36495_500step_seed11"
        gate_dir = outputs / f"skyscript_gate_m4_{candidate}_sat_seed11"
        run_dir.mkdir(parents=True)
        eligible = index == 0
        validation = {0: 4.0, 100: 1.9, 250: 1.4, 500: 1.1 if eligible else 1.3}
        (run_dir / "validation.jsonl").write_text(
            "".join(
                json.dumps({"step": step, "loss": loss}) + "\n"
                for step, loss in validation.items()
            ),
            encoding="utf-8",
        )
        _write_json(
            gate_dir / "training_step500.json",
            {
                "steps": 500,
                "completed": True,
                "visual_backbone_permanently_frozen": True,
                "optimizer_parameter_groups": {
                    "groups": [{"name": "image_adapter"}, {"name": "text_projection"}]
                },
            },
        )
        _write_json(gate_dir / "rsicd_val_official.json", _evaluation(None, 0.01))
        for step in (0, 100, 250, 500):
            sky = 0.01 if step == 0 else step / 10000 + (0.03 if eligible else 0.0)
            rsicd = 0.01 if step == 0 else step / 10000 + 0.02
            _write_json(
                gate_dir / f"skyscript_val_step{step:03d}.json",
                _evaluation(step, sky),
            )
            _write_json(
                gate_dir / f"rsicd_val_step{step:03d}.json",
                _evaluation(step, rsicd),
            )
        _write_json(gate_dir / "text_embedding_drift.json", _drift_report())

    report = summarize(
        baseline_path=baseline,
        candidate_root=outputs,
        output_root=outputs,
    )

    assert report["status"] == "complete"
    assert report["visual_backbone_permanently_frozen"]
    assert report["vision_head_frozen"]
    assert report["eligible_candidates"] == [CANDIDATES[0]]
    assert report["ranking_by_skyscript_then_validation_then_rsicd"] == [CANDIDATES[0]]
