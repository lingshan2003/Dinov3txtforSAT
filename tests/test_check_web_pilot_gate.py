import json

import pytest

from tools.check_web_pilot_gate import check_web_pilot_gate


def _report(*, evidence_status: str = "complete", improved: bool = True) -> dict[str, object]:
    return {
        "artifact_evidence_status": evidence_status,
        "training_summary": {"steps": 500, "target_steps": 5000, "completed": False},
        "validation": {
            "best_relative_change_from_step_zero": -0.2 if improved else 0.1,
            "best_step_from_records": 300,
        },
        "best_checkpoint_consistency": {
            "best_step_matches_validation_records": True,
            "best_loss_matches_validation_records": True,
            "best_file_matches_summary_best_step": True,
        },
    }


def test_web_gate_permits_reviewed_degraded_evidence_for_sat_pilot(tmp_path) -> None:
    report_path = tmp_path / "recovery_report.json"
    report_path.write_text(json.dumps(_report(evidence_status="degraded")), encoding="utf-8")

    with pytest.raises(ValueError, match="degraded"):
        check_web_pilot_gate(report_path)

    result = check_web_pilot_gate(report_path, allow_degraded_artifacts=True)
    assert result["decision"] == "proceed_to_sat_pilot"
    assert result["web_validation_best_step"] == 300
    assert result["warning"] is not None


def test_web_gate_rejects_non_improving_validation(tmp_path) -> None:
    report_path = tmp_path / "recovery_report.json"
    report_path.write_text(json.dumps(_report(improved=False)), encoding="utf-8")

    with pytest.raises(ValueError, match="did not improve"):
        check_web_pilot_gate(report_path)
