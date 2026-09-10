from pathlib import Path

import pytest

import tools.summarize_skyscript_gate_m4_b as gate


def _model(domain: str, trainable: int) -> dict:
    identity = gate.MODEL_IDENTITIES[domain]
    return {
        "backbone_domain": domain,
        "backbone_weights": {"sha256": identity["backbone_weights_sha256"]},
        "dinotxt_weights": {"sha256": identity["dinotxt_weights_sha256"]},
        "bpe_vocab": {"sha256": identity["bpe_vocab_sha256"]},
        "trainable_parameters": {"trainable": trainable},
    }


def test_m4_b_model_identity_accepts_each_frozen_domain() -> None:
    for domain in ("web", "sat"):
        gate._validate_model_identity(
            _model(domain, 1_054_976), domain=domain, trainable=1_054_976
        )


def test_m4_b_model_identity_rejects_domain_cross_contamination() -> None:
    with pytest.raises(ValueError, match="backbone_domain"):
        gate._validate_model_identity(
            _model("web", 1_054_976), domain="sat", trainable=1_054_976
        )


def test_m4_b_stage_summary_adds_registered_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate,
        "summarize_s2_stage",
        lambda **_: {"gate": "SkyScript_S2_stage", "status": "pass"},
    )
    monkeypatch.setattr(gate, "validate_evidence_identity", lambda **_: None)
    report = gate.summarize_stage(
        run_dir=Path("run"),
        gate_dir=Path("gate"),
        stage=100,
        training_report_path=Path("training.json"),
        domain="sat",
        seed=47,
    )
    assert report["gate"] == "SkyScript_M4_B_stage"
    assert report["domain"] == "sat"
    assert report["seed"] == 47
    assert report["model_identity"]["trainable_parameters"] == 1_054_976
