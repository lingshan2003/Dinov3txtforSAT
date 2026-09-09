from tools.verify_preflight_compatibility import compare_preflights


def test_preflight_commit_change_is_advisory() -> None:
    result = compare_preflights(
        {"project_commit": "old", "config_sha256": "same", "stage": "100"},
        {"project_commit": "new", "config_sha256": "same", "stage": "100"},
    )

    assert result["status"] == "warning"
    assert result["blocking_fields"] == []
    assert result["advisory_fields"] == ["project_commit"]


def test_preflight_protocol_change_is_blocking() -> None:
    result = compare_preflights(
        {"project_commit": "old", "config_sha256": "first"},
        {"project_commit": "new", "config_sha256": "second"},
    )

    assert result["status"] == "fail"
    assert result["blocking_fields"] == ["config_sha256"]
