import json
from pathlib import Path

import pytest

import tools.summarize_skyscript_gate_m4_sat as m4_sat


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _model_identity(trainable: int, *, domain: str = "sat") -> dict:
    return {
        "backbone_domain": domain,
        "backbone_weights": {"sha256": m4_sat.SAT_BACKBONE_SHA256},
        "dinotxt_weights": {"sha256": m4_sat.DINOTXT_SHA256},
        "bpe_vocab": {"sha256": m4_sat.BPE_SHA256},
        "trainable_parameters": {"trainable": trainable},
    }


def _prepare_identity_evidence(root: Path, stage: int) -> tuple[Path, Path]:
    run_dir = root / "run"
    gate_dir = root / "gate"
    run_dir.mkdir()
    gate_dir.mkdir()
    _write_json(
        run_dir / "skyscript_step0_parity.json",
        {
            "backbone_domain": "sat",
            "official_model": _model_identity(0),
            "step0_model": _model_identity(m4_sat.ADAPTER_TRAINABLE_PARAMETERS),
        },
    )
    for name in (
        "skyscript_val_step000.json",
        f"skyscript_val_step{stage:03d}.json",
        "rsicd_val_official.json",
        "rsicd_val_step000.json",
        f"rsicd_val_step{stage:03d}.json",
    ):
        _write_json(
            gate_dir / name,
            {"model": _model_identity(m4_sat.ADAPTER_TRAINABLE_PARAMETERS)},
        )
    return run_dir, gate_dir


def test_sat_identity_validation_accepts_frozen_assets(tmp_path: Path) -> None:
    run_dir, gate_dir = _prepare_identity_evidence(tmp_path, 100)
    m4_sat.validate_sat_evidence_identity(run_dir=run_dir, gate_dir=gate_dir, stage=100)


def test_sat_identity_validation_rejects_web_evaluation(tmp_path: Path) -> None:
    run_dir, gate_dir = _prepare_identity_evidence(tmp_path, 100)
    path = gate_dir / "rsicd_val_step100.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["model"] = _model_identity(m4_sat.ADAPTER_TRAINABLE_PARAMETERS, domain="web")
    _write_json(path, report)

    with pytest.raises(ValueError, match="backbone_domain='sat'"):
        m4_sat.validate_sat_evidence_identity(run_dir=run_dir, gate_dir=gate_dir, stage=100)


def test_stage_wrapper_reuses_s2_gate_and_adds_sat_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, gate_dir = _prepare_identity_evidence(tmp_path, 100)
    base = {
        "format_version": 1,
        "gate": "SkyScript_S2_stage",
        "stage": 100,
        "status": "pass",
        "checks": {"shared": True},
    }
    monkeypatch.setattr(m4_sat, "summarize_s2_stage", lambda **_: base.copy())

    result = m4_sat.summarize_stage(
        run_dir=run_dir,
        gate_dir=gate_dir,
        stage=100,
        training_report_path=gate_dir / "training_step100.json",
    )

    assert result["gate"] == "SkyScript_M4_SAT_stage"
    assert result["domain"] == "sat"
    assert result["shared_gate_implementation"] == "SkyScript_S2_stage"
    assert result["model_identity"]["trainable_parameters"] == 1_054_976


def test_final_summary_combines_three_passing_sat_stages(tmp_path: Path) -> None:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    model_identity = {
        "backbone_domain": "sat",
        "backbone_weights_sha256": m4_sat.SAT_BACKBONE_SHA256,
    }
    for stage in m4_sat.STAGES:
        _write_json(
            gate_dir / f"stage_{stage}_summary.json",
            {
                "gate": "SkyScript_M4_SAT_stage",
                "domain": "sat",
                "stage": stage,
                "status": "pass",
                "checks": {"stage_gate": True},
                "training": {"project_commit": "project-commit"},
                "model_identity": model_identity,
                "validation_loss": {"best_step": stage},
                "skyscript": {
                    "step0": {"mean_recall": 0.08},
                    f"step{stage}": {"mean_recall": 0.08 + stage / 10_000},
                    "current_minus_step0_mean_recall": stage / 10_000,
                },
                "rsicd_val": {
                    "official": {"mean_recall": 0.16},
                    "step0": {"mean_recall": 0.16},
                    f"step{stage}": {"mean_recall": 0.159},
                    "current_minus_official_mean_recall": -0.001,
                },
            },
        )

    result = m4_sat.summarize_final(gate_dir=gate_dir)

    assert result["gate"] == "SkyScript_M4_SAT"
    assert result["status"] == "pass"
    assert result["stages"] == [100, 250, 500]
    assert result["skyscript"]["current_minus_step0_mean_recall"]["500"] == 0.05
