import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dinotxt_rs.training import checkpoint


def _identity(**changes) -> dict:
    return {
        "format_version": 1,
        "config_sha256": "synthetic-config-identity",
        "project_commit": "commit-a",
        "dinov3_commit": "upstream-a",
        "files": {"backbone_weights": "synthetic-weight-identity"},
        **changes,
    }


def _checkpoint(path: Path, step: int, identity: dict | None = None) -> Path:
    torch.save(
        {
            "format_version": 2,
            "step": step,
            "run_identity": _identity() if identity is None else identity,
            "trainable_model": {"weight": torch.tensor([float(step)])},
        },
        path,
    )
    return path


def _save(output_dir: Path, *, step: int = 1, name: str | None = None) -> Path:
    model = torch.nn.Module()
    model.visual_model = torch.nn.Module()
    model.visual_model.backbone = torch.nn.Identity()
    model.visual_model.head = torch.nn.Linear(2, 2)
    empty_state = SimpleNamespace(state_dict=lambda: {})
    return checkpoint.save_checkpoint(
        output_dir,
        model=model,
        optimizer=empty_state,
        scheduler=empty_state,
        scaler=empty_state,
        queue=empty_state,
        sampler_state={},
        loader_generator_state=torch.Generator().get_state(),
        run_state={},
        run_identity=_identity(),
        optimizer_parameter_groups={},
        step=step,
        config_text="synthetic configuration",
        name=name,
    )


def _log(path: Path, steps: list[int]) -> bytes:
    content = "".join(json.dumps({"step": step, "loss": float(step)}) + "\n" for step in steps)
    path.write_text(content, encoding="utf-8")
    return content.encode()


def test_save_checkpoint_ignores_stale_hard_link_and_supports_name(tmp_path) -> None:
    old = _checkpoint(tmp_path / "old.pt", 0)
    old_content = old.read_bytes()
    (tmp_path / "latest.pt.part").hardlink_to(old)

    result = _save(tmp_path, name="latest.pt")

    assert result == tmp_path / "latest.pt"
    assert old.read_bytes() == old_content
    assert torch.load(result, weights_only=False)["step"] == 1
    assert not list(tmp_path.glob(".latest.pt.*.part"))


def test_best_save_ignores_stale_hard_link_and_preserves_old_checkpoint(tmp_path) -> None:
    old = _checkpoint(tmp_path / "step_0000000.pt", 0)
    new = _checkpoint(tmp_path / "step_0000001.pt", 1)
    checkpoint.save_best_checkpoint(tmp_path, old)
    (tmp_path / "best.pt.part").hardlink_to(old)
    old_content = old.read_bytes()

    result = checkpoint.save_best_checkpoint(tmp_path, new)

    assert old.read_bytes() == old_content
    assert result.stat().st_ino == new.stat().st_ino
    assert torch.load(result, weights_only=False)["step"] == 1


def test_best_save_copy_fallback_is_atomic(tmp_path, monkeypatch) -> None:
    source = _checkpoint(tmp_path / "step_0000001.pt", 1)

    def no_hard_links(*args):
        raise OSError("hard links unavailable")

    monkeypatch.setattr(checkpoint.os, "link", no_hard_links)
    result = checkpoint.save_best_checkpoint(tmp_path, source)

    assert result.read_bytes() == source.read_bytes()
    assert result.stat().st_ino != source.stat().st_ino
    assert not list(tmp_path.glob(".best.pt.*.part"))


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_checkpoint_failure_preserves_destination_and_cleans_temporary(
    tmp_path, monkeypatch, failure
) -> None:
    destination = _checkpoint(tmp_path / "latest.pt", 0)
    previous = destination.read_bytes()

    def fail(*args, **kwargs):
        if failure == "write":
            args[1].write(b"partial failed serialization")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(checkpoint.torch if failure == "write" else checkpoint.os,
                        "save" if failure == "write" else "replace", fail)
    with pytest.raises(OSError, match="simulated disk failure"):
        _save(tmp_path, name="latest.pt")

    assert destination.read_bytes() == previous
    assert not list(tmp_path.glob(".latest.pt.*.part"))


def test_best_replace_failure_preserves_best_and_cleans_temporary(tmp_path, monkeypatch) -> None:
    old = _checkpoint(tmp_path / "old.pt", 0)
    new = _checkpoint(tmp_path / "new.pt", 1)
    best = checkpoint.save_best_checkpoint(tmp_path, old)
    previous = best.read_bytes()

    def fail(*args):
        raise OSError("replace failure")

    monkeypatch.setattr(checkpoint.os, "replace", fail)
    with pytest.raises(OSError, match="replace failure"):
        checkpoint.save_best_checkpoint(tmp_path, new)

    assert best.read_bytes() == previous
    assert not list(tmp_path.glob(".best.pt.*.part"))


def test_best_copy_failure_preserves_best_and_cleans_temporary(tmp_path, monkeypatch) -> None:
    old = _checkpoint(tmp_path / "old.pt", 0)
    new = _checkpoint(tmp_path / "new.pt", 1)
    best = checkpoint.save_best_checkpoint(tmp_path, old)
    previous = best.read_bytes()

    def fail_link(*args):
        raise OSError("no hard links")

    def fail_sync(*args):
        raise OSError("write failure")

    monkeypatch.setattr(checkpoint.os, "link", fail_link)
    monkeypatch.setattr(checkpoint.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="write failure"):
        checkpoint.save_best_checkpoint(tmp_path, new)

    assert best.read_bytes() == previous
    assert not list(tmp_path.glob(".best.pt.*.part"))


def test_resume_archives_future_logs_and_recovers_recorded_best(tmp_path) -> None:
    old = _checkpoint(tmp_path / "step_0000000.pt", 0)
    latest = _checkpoint(tmp_path / "latest.pt", 1)
    _checkpoint(tmp_path / "best.pt", 3)
    for filename in ("metrics.jsonl", "validation.jsonl", "fixed_monitor.jsonl"):
        _log(tmp_path / filename, [0, 1, 2, 3])

    result = checkpoint.prepare_resume_artifacts(
        tmp_path, 1, {"best_validation_step": 0}, latest, _identity()
    )

    assert result["best_recovered_from"] == str(old.resolve())
    assert result["discarded_records"] == {
        "metrics.jsonl": 2, "validation.jsonl": 2, "fixed_monitor.jsonl": 2
    }
    assert torch.load(tmp_path / "best.pt", weights_only=False)["step"] == 0
    for filename in result["discarded_records"]:
        records = [json.loads(line) for line in (tmp_path / filename).read_text().splitlines()]
        assert [record["step"] for record in records] == [0, 1]
    archived = json.loads((tmp_path / "resume_discarded.jsonl").read_text())
    assert archived["resume_step"] == 1
    assert {entry["filename"] for entry in archived["logs"]} == set(result["discarded_records"])
    assert all([record["step"] for record in entry["records"]] == [2, 3]
               for entry in archived["logs"])
    repeated = checkpoint.prepare_resume_artifacts(
        tmp_path, 1, {"best_validation_step": 0}, latest, _identity()
    )
    assert not any(repeated["discarded_records"].values())
    assert len((tmp_path / "resume_discarded.jsonl").read_text().splitlines()) == 1


def test_resume_can_recover_best_from_the_resume_checkpoint(tmp_path) -> None:
    source = _checkpoint(tmp_path / "latest.pt", 2)
    _checkpoint(tmp_path / "best.pt", 3)

    result = checkpoint.prepare_resume_artifacts(
        tmp_path, 2, {"best_validation_step": 2}, source, _identity()
    )

    assert result["best_recovered_from"] == str(source.resolve())
    assert torch.load(tmp_path / "best.pt", weights_only=False)["step"] == 2


def test_resume_allows_advisory_project_commit_change_in_best(tmp_path) -> None:
    _checkpoint(tmp_path / "best.pt", 1, _identity(project_commit="old-project-commit"))

    result = checkpoint.prepare_resume_artifacts(
        tmp_path, 2, {"best_validation_step": 1}, tmp_path / "latest.pt", _identity()
    )

    assert result["best_recovered_from"] is None


def test_best_recovery_write_failure_does_not_truncate_logs(tmp_path, monkeypatch) -> None:
    _checkpoint(tmp_path / "step_0000000.pt", 0)
    _checkpoint(tmp_path / "best.pt", 3)
    previous = _log(tmp_path / "metrics.jsonl", [1, 2, 3])

    def fail(*args):
        raise OSError("best recovery write failure")

    monkeypatch.setattr(checkpoint, "save_best_checkpoint", fail)
    with pytest.raises(OSError, match="best recovery write failure"):
        checkpoint.prepare_resume_artifacts(
            tmp_path, 1, {"best_validation_step": 0}, tmp_path / "latest.pt", _identity()
        )

    assert (tmp_path / "metrics.jsonl").read_bytes() == previous
    assert not (tmp_path / "resume_discarded.jsonl").exists()
    assert not list(tmp_path.glob(".*.part"))


def test_resume_from_best_does_not_rewrite_its_loaded_source(tmp_path) -> None:
    best = _checkpoint(tmp_path / "best.pt", 2)
    previous = best.read_bytes()
    inode = best.stat().st_ino
    _log(tmp_path / "metrics.jsonl", [1, 2, 3])

    result = checkpoint.prepare_resume_artifacts(
        tmp_path, 2, {"best_validation_step": 2}, best, _identity()
    )

    assert result["best_recovered_from"] is None
    assert best.read_bytes() == previous
    assert best.stat().st_ino == inode


@pytest.mark.parametrize("bad_identity", [False, True])
def test_unrecoverable_best_refuses_before_changing_logs(tmp_path, bad_identity) -> None:
    identity = _identity(dinov3_commit="other-upstream") if bad_identity else _identity()
    _checkpoint(tmp_path / "best.pt", 0 if bad_identity else 3, identity)
    latest = _checkpoint(tmp_path / "latest.pt", 1)
    previous = _log(tmp_path / "metrics.jsonl", [1, 2, 3])

    with pytest.raises(ValueError, match="Cannot recover best.pt"):
        checkpoint.prepare_resume_artifacts(
            tmp_path, 1, {"best_validation_step": 0}, latest, _identity()
        )

    assert (tmp_path / "metrics.jsonl").read_bytes() == previous
    assert not (tmp_path / "resume_discarded.jsonl").exists()


@pytest.mark.parametrize("record", [{"step": True}, {"step": -1}, {"step": 1.5}, {"loss": 1}])
def test_invalid_log_step_refuses_before_mutation(tmp_path, record) -> None:
    log = tmp_path / "metrics.jsonl"
    log.write_text(json.dumps(record) + "\n")
    previous = log.read_bytes()

    with pytest.raises(ValueError, match="strictly increasing integers"):
        checkpoint.prepare_resume_artifacts(
            tmp_path, 1, {}, tmp_path / "latest.pt", _identity()
        )

    assert log.read_bytes() == previous


def test_resume_rejects_duplicate_log_steps_before_mutation(tmp_path) -> None:
    previous = _log(tmp_path / "metrics.jsonl", [1, 1, 2])

    with pytest.raises(ValueError, match="strictly increasing integers"):
        checkpoint.prepare_resume_artifacts(
            tmp_path, 1, {}, tmp_path / "latest.pt", _identity()
        )

    assert (tmp_path / "metrics.jsonl").read_bytes() == previous


def test_resume_refuses_stale_best_when_restored_state_has_no_best(tmp_path) -> None:
    _checkpoint(tmp_path / "best.pt", 3)
    previous = _log(tmp_path / "metrics.jsonl", [1, 2, 3])

    with pytest.raises(ValueError, match="no best_validation_step but best.pt exists"):
        checkpoint.prepare_resume_artifacts(
            tmp_path, 1, {}, tmp_path / "latest.pt", _identity()
        )

    assert (tmp_path / "metrics.jsonl").read_bytes() == previous
