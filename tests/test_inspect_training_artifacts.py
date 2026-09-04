import json
from pathlib import Path

import torch

from tools.inspect_training_artifacts import inspect_training_artifacts


def _write_checkpoint(path: Path, step: int) -> None:
    torch.save(
        {
            "format_version": 2,
            "step": step,
            "run_state": {
                "micro_step": step * 4,
                "validation_evaluations": 2,
                "best_validation_loss": 1.0,
                "best_validation_step": step,
            },
            "run_identity": {
                "format_version": 1,
                "config_sha256": "config",
                "project_commit": "project",
                "dinov3_commit": "dinov3",
            },
        },
        path,
    )


def test_inspection_marks_pruned_resume_checkpoint_as_degraded(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    step_zero = output / "step_0000000.pt"
    _write_checkpoint(step_zero, 0)
    best = output / "best.pt"
    _write_checkpoint(best, 500)
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "steps": 500,
                "target_steps": 5000,
                "completed": False,
                "final_checkpoint": str(output / "step_0000500.pt"),
                "validation": {"best_step": 500, "best_loss": 2.0},
            }
        ),
        encoding="utf-8",
    )
    (output / "validation.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"step": 0, "loss": 3.0},
                {"step": 250, "loss": 2.5},
                {"step": 500, "loss": 2.0},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "resume_history.jsonl").write_text(
        json.dumps({"checkpoint_step": 250, "checkpoint_sha256": "hash"}) + "\n",
        encoding="utf-8",
    )

    report = inspect_training_artifacts(
        output_dir=output,
        expected_checkpoint_steps=(0, 250, 500),
    )

    assert report["artifact_evidence_status"] == "degraded"
    assert report["missing_required_checkpoint_steps"] == [250, 500]
    assert report["validation"]["best_step_from_records"] == 500
    assert report["validation"]["best_relative_change_from_step_zero"] == -1 / 3
    assert report["best_checkpoint_consistency"]["best_file_matches_summary_best_step"]
    assert report["resume_evidence"]["not_replayable_from_retained_checkpoint_steps"] == [250]
    assert report["recovery_options"][0]["status"] == "not_recoverable_from_remaining_checkpoints"
    assert report["recovery_options"][1]["status"] == "recoverable_from_retained_same_step_payload"
