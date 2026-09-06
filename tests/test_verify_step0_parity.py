from __future__ import annotations

import json

import pytest

from tools.verify_step0_parity import verify_step0_parity


def _report(domain: str, manifest_hash: str = "manifest") -> dict[str, object]:
    return {
        "status": "pass",
        "backbone_domain": domain,
        "checkpoint_step": 0,
        "project_commit": "commit",
        "precision": "bf16",
        "input": {
            "manifest_sha256": manifest_hash,
            "identity_sha256": f"{domain}-input",
        },
        "step0_model": {"checkpoint": {"sha256": f"{domain}-checkpoint"}},
        "tokens": {"passed": True},
        "trainable_parameters": {"passed": True},
        "comparisons": {
            "image_features": {"passed": True, "max_absolute_error": 0.0},
            "text_features": {"passed": True, "max_absolute_error": 0.0},
            "logit_scale": {"passed": True, "max_absolute_error": 0.0},
            "patch_tokens": {"passed": True, "max_absolute_error": 0.0},
            "backbone_patch_tokens": {"passed": True, "max_absolute_error": 0.0},
            "similarity_logits": {"passed": True, "max_absolute_error": 0.0},
            "symmetric_contrastive_loss": {"passed": True, "max_absolute_error": 0.0},
        },
    }


def test_verify_step0_parity_accepts_two_passing_reports(tmp_path) -> None:
    for domain in ("web", "sat"):
        (tmp_path / f"{domain}.json").write_text(
            json.dumps(_report(domain)), encoding="utf-8"
        )

    result = verify_step0_parity(tmp_path)

    assert result["status"] == "complete"
    assert result["gate"] == "A_step0_parity"
    assert result["input_manifest_sha256"] == "manifest"


def test_verify_step0_parity_rejects_different_inputs(tmp_path) -> None:
    (tmp_path / "web.json").write_text(json.dumps(_report("web")), encoding="utf-8")
    (tmp_path / "sat.json").write_text(
        json.dumps(_report("sat", manifest_hash="different")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="different input manifests"):
        verify_step0_parity(tmp_path)
