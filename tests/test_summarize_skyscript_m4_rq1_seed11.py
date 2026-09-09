import json
import subprocess
import sys
from pathlib import Path

import pytest

import tools.summarize_skyscript_m4_rq1_seed11 as m4

TOOL = Path("tools/summarize_skyscript_m4_rq1_seed11.py")
WEB_CONFIG = Path("configs/skyscript_web_adapter_500step_seed11.toml")
SAT_CONFIG = Path("configs/skyscript_sat_adapter_500step_seed11.toml")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _metrics(value: float) -> dict:
    return {
        "image_to_text": {
            "r1": value,
            "r5": value + 0.01,
            "r10": value + 0.02,
            "median_rank": 100.0 - value,
            "mean_rank": 200.0 - value,
        },
        "text_to_image": {
            "r1": value + 0.001,
            "r5": value + 0.011,
            "r10": value + 0.021,
            "median_rank": 90.0 - value,
            "mean_rank": 190.0 - value,
        },
        "mean_recall": value + 0.0105,
    }


def _model(domain: str, step: int | None) -> dict:
    result = {
        "backbone_domain": domain,
        "backbone_weights": {"sha256": m4.EXPECTED_MODELS[domain]},
        "dinotxt_weights": {"sha256": m4.DINOTXT_SHA256},
        "bpe_vocab": {"sha256": m4.BPE_SHA256},
        "trainable_parameters": {"trainable": m4.ADAPTER_TRAINABLE_PARAMETERS},
    }
    result["checkpoint"] = (
        None
        if step is None
        else {
            "step": step,
            "run_identity": {"project_commit": f"{domain}-checkpoint-commit"},
        }
    )
    return result


def _prepare_domain(root: Path, domain: str) -> Path:
    gate_dir = root / domain
    gate_dir.mkdir()
    baseline = 0.08 if domain == "web" else 0.01
    sky = {
        "step0": _metrics(baseline),
        "steps": {
            str(stage): _metrics(baseline + stage / 10_000) for stage in m4.STAGES
        },
    }
    rsicd_baseline = 0.16 if domain == "web" else 0.02
    rsicd = {
        "official": _metrics(rsicd_baseline),
        "step0": _metrics(rsicd_baseline),
        "steps": {
            str(stage): _metrics(rsicd_baseline + stage / 20_000)
            for stage in m4.STAGES
        },
    }
    stage_paths = {}
    for stage in m4.STAGES:
        stage_path = gate_dir / f"stage_{stage}_summary.json"
        _write_json(
            stage_path,
            {
                "gate": (
                    "SkyScript_S2_stage"
                    if domain == "web"
                    else "SkyScript_M4_SAT_stage"
                ),
                "stage": stage,
                "status": "pass",
                "training": {
                    "project_commit": f"{domain}-training-commit",
                    "dinov3_commit": m4.DINOV3_COMMIT,
                    "train_manifest_sha256": m4.TRAIN_MANIFEST_SHA256,
                    "val_manifest_sha256": m4.VAL_MANIFEST_SHA256,
                },
            },
        )
        stage_paths[str(stage)] = str(stage_path)

    summary = {
        "gate": "SkyScript_S2" if domain == "web" else "SkyScript_M4_SAT",
        "status": "pass",
        "stages": list(m4.STAGES),
        "validation_loss": {
            "steps": {
                str(step): 2.0 - step / 1_000 + (0.5 if domain == "sat" else 0.0)
                for step in range(0, 501, 50)
            },
            "best_step": 500,
        },
        "skyscript": sky,
        "rsicd_val": rsicd,
        "stage_reports": stage_paths,
    }
    if domain == "sat":
        summary["domain"] = "sat"
    summary_path = gate_dir / "summary.json"
    _write_json(summary_path, summary)

    report_metrics = {
        "skyscript_val_step000.json": (sky["step0"], 0),
        "rsicd_val_official.json": (rsicd["official"], None),
        "rsicd_val_step000.json": (rsicd["step0"], 0),
    }
    for stage in m4.STAGES:
        report_metrics[f"skyscript_val_step{stage:03d}.json"] = (
            sky["steps"][str(stage)],
            stage,
        )
        report_metrics[f"rsicd_val_step{stage:03d}.json"] = (
            rsicd["steps"][str(stage)],
            stage,
        )
    for name, (metrics, step) in report_metrics.items():
        _write_json(gate_dir / name, {"model": _model(domain, step), "metrics": metrics})
    return summary_path


def test_direct_entrypoint_can_import_project_modules() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "matched-seed Web/SAT comparison" in result.stdout


def test_m4_a_summary_contains_absolute_and_delta_comparisons(tmp_path: Path) -> None:
    web_summary = _prepare_domain(tmp_path, "web")
    sat_summary = _prepare_domain(tmp_path, "sat")

    report = m4.summarize_m4_a(
        web_summary_path=web_summary,
        sat_summary_path=sat_summary,
        web_config=WEB_CONFIG,
        sat_config=SAT_CONFIG,
    )

    assert report["status"] == "pass"
    assert report["scope"].startswith("matched_seed11_pilot")
    assert report["domains"]["web"]["model"]["backbone_domain"] == "web"
    assert report["domains"]["sat"]["model"]["backbone_domain"] == "sat"
    assert report["domains"]["sat"]["model"]["checkpoint_project_commits"] == {
        "0": "sat-checkpoint-commit",
        "100": "sat-checkpoint-commit",
        "250": "sat-checkpoint-commit",
        "500": "sat-checkpoint-commit",
    }
    step500 = report["skyscript"]["steps"]["500"]
    assert set(step500["absolute"]["web"]["image_to_text"]) == {
        "r1",
        "r5",
        "r10",
        "median_rank",
        "mean_rank",
    }
    assert step500["absolute"]["sat_minus_web"]["mean_recall"] == pytest.approx(-0.07)
    assert step500["within_domain_delta"]["web"]["mean_recall"] == pytest.approx(0.05)
    assert step500["within_domain_delta"]["sat"]["mean_recall"] == pytest.approx(0.05)
    assert step500["sat_delta_minus_web_delta"]["mean_recall"] == pytest.approx(0.0)
    assert report["headline_step500"]["skyscript"]["sat_minus_web_mean_recall"] == (
        pytest.approx(-0.07)
    )
    assert report["validation_loss"]["500"]["within_domain_delta"]["web"] == -0.5


def test_m4_a_summary_rejects_evaluation_metrics_from_another_run(tmp_path: Path) -> None:
    web_summary = _prepare_domain(tmp_path, "web")
    sat_summary = _prepare_domain(tmp_path, "sat")
    path = sat_summary.parent / "skyscript_val_step500.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["metrics"] = _metrics(0.99)
    _write_json(path, report)

    with pytest.raises(ValueError, match="metrics do not match"):
        m4.summarize_m4_a(
            web_summary_path=web_summary,
            sat_summary_path=sat_summary,
            web_config=WEB_CONFIG,
            sat_config=SAT_CONFIG,
        )
