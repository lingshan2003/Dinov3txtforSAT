import json
import subprocess
import sys
from pathlib import Path

from tools.summarize_skyscript_m4_f1_f2_sat import CANDIDATES, summarize
from tools.verify_skyscript_m4_f1_f2_configs import verify

RUNNER = Path("scripts/run_skyscript_m4_f1_f2_sat_seed11.sh")


def test_f1_f2_sat_configs_match_the_archived_f0_protocol() -> None:
    configs = verify()
    assert set(configs) == {
        "f1_visionhead_lr1e5",
        "f1_visionhead_lr5e6",
        "f2_adapter256_visionhead_lr1e5",
        "f2_adapter256_visionhead_lr5e6",
    }
    assert all(config.model.backbone_domain == "sat" for config in configs.values())
    assert all(config.train.max_steps == 500 for config in configs.values())


def test_f1_f2_runner_has_valid_syntax_and_reuses_f0() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    source = RUNNER.read_text(encoding="utf-8")
    assert 'F0_SUMMARY="outputs/skyscript_gate_m4_sat_seed11/summary.json"' in source
    assert "f0=verified_reused_not_retrained" in source
    assert "--require-visual-backbone-frozen" in source
    assert "tools/summarize_skyscript_m4_f1_f2_sat.py" in source


def test_f1_f2_runner_help_needs_no_server_assets() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "never retrained" in result.stdout
    assert "visual backbone remains frozen" in result.stdout


def test_f1_f2_config_verifier_cli() -> None:
    result = subprocess.run(
        [sys.executable, "tools/verify_skyscript_m4_f1_f2_configs.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "m4_f1_f2_sat_config_protocol=verified" in result.stdout


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


def test_f1_f2_summary_reuses_f0_and_compares_all_candidates(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    baseline = tmp_path / "f0.json"
    stages = (100, 250, 500)
    _write_json(
        baseline,
        {
            "gate": "SkyScript_M4_SAT",
            "status": "pass",
            "stages": list(stages),
            "validation_loss": {
                "steps": {"0": 4.0, "100": 3.0, "250": 2.0, "500": 1.5}
            },
            "skyscript": {
                "steps": {str(step): {"mean_recall": 0.05} for step in stages}
            },
            "rsicd_val": {
                "steps": {str(step): {"mean_recall": 0.06} for step in stages}
            },
        },
    )
    for index, candidate in enumerate(CANDIDATES):
        run_dir = outputs / f"skyscript_sat_{candidate}_36495_500step_seed11"
        gate_dir = outputs / f"skyscript_gate_m4_{candidate}_sat_seed11"
        run_dir.mkdir(parents=True)
        validation = {0: 4.0, 100: 2.9, 250: 1.9, 500: 1.4 - index / 100}
        (run_dir / "validation.jsonl").write_text(
            "".join(
                json.dumps({"step": step, "loss": loss}) + "\n"
                for step, loss in validation.items()
            ),
            encoding="utf-8",
        )
        group_names = ["vision_head"]
        if candidate.startswith("f2_"):
            group_names.insert(0, "image_adapter")
        _write_json(
            gate_dir / "training_step500.json",
            {
                "steps": 500,
                "completed": True,
                "visual_backbone_permanently_frozen": True,
                "optimizer_parameter_groups": {
                    "groups": [{"name": name} for name in group_names]
                },
            },
        )
        _write_json(gate_dir / "rsicd_val_official.json", _evaluation(None, 0.01))
        for step in (0, *stages):
            gain = 0.0 if step == 0 else step / 10000 + index / 1000
            _write_json(
                gate_dir / f"skyscript_val_step{step:03d}.json",
                _evaluation(step, 0.01 + gain),
            )
            _write_json(
                gate_dir / f"rsicd_val_step{step:03d}.json",
                _evaluation(step, 0.01 + gain),
            )

    report = summarize(
        baseline_path=baseline,
        candidate_root=outputs,
        output_root=outputs,
    )

    assert report["status"] == "complete"
    assert report["f0"]["reused_not_retrained"]
    assert report["visual_backbone_permanently_frozen"]
    assert set(report["candidates"]) == set(CANDIDATES)
    assert report["step500_ranking_by_skyscript_then_rsicd"][0] == CANDIDATES[-1]
