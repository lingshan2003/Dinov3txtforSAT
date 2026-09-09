from dinotxt_rs.evaluation.common import _checkpoint_identity_check
from dinotxt_rs.training.provenance import compare_run_identities


def _identity(commit: str, *, files: dict[str, str] | None = None) -> dict:
    return {
        "format_version": 1,
        "config_sha256": "config",
        "project_commit": commit,
        "dinov3_commit": "dinov3",
        "files": files or {"manifest": "manifest-sha"},
    }


def test_project_commit_difference_is_advisory() -> None:
    blocking, advisory = compare_run_identities(
        _identity("current-commit"),
        _identity("checkpoint-commit"),
    )

    assert blocking == []
    assert advisory == ["project_commit"]


def test_input_hash_difference_remains_blocking() -> None:
    blocking, advisory = compare_run_identities(
        _identity("current", files={"manifest": "current-sha"}),
        _identity("checkpoint", files={"manifest": "checkpoint-sha"}),
    )

    assert blocking == ["files"]
    assert advisory == ["project_commit"]


def test_evaluation_records_current_checkout_commit_difference() -> None:
    blocking, report = _checkpoint_identity_check(
        _identity("training-commit"),
        _identity("training-commit"),
        "evaluation-bugfix-commit",
    )

    assert blocking == []
    assert report["status"] == "warning"
    assert report["advisory_mismatches"] == ["project_commit"]
    assert report["checkpoint_project_commit"] == "training-commit"
    assert report["current_project_commit"] == "evaluation-bugfix-commit"
