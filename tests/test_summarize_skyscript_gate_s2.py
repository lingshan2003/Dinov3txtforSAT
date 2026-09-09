import json
from pathlib import Path

from tools.summarize_skyscript_gate_s2 import summarize_final, summarize_stage

STAGES = (100, 250, 500)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _metrics(mean_recall: float) -> dict:
    return {
        "image_to_text": {
            "r1": mean_recall,
            "r5": mean_recall,
            "r10": mean_recall,
            "median_rank": 10.0,
            "mean_rank": 20.0,
        },
        "text_to_image": {
            "r1": mean_recall,
            "r5": mean_recall,
            "r10": mean_recall,
            "median_rank": 11.0,
            "mean_rank": 21.0,
        },
        "mean_recall": mean_recall,
    }


def _evaluation(
    *, task: str, manifest_sha: str, step: int | None, metrics: dict, source: str | None
) -> dict:
    report = {
        "task": task,
        "split": "val",
        "manifest": {"sha256": manifest_sha},
        "model": {"checkpoint": None if step is None else {"step": step}},
        "metrics": metrics,
    }
    if source is not None:
        report["sources"] = [source]
    return report


def _prepare_evidence(root: Path) -> tuple[Path, Path, dict[int, Path]]:
    run = root / "run"
    gate = root / "gate"
    run.mkdir()
    gate.mkdir()
    train_sha = "train-sha"
    val_sha = "val-sha"
    rsicd_sha = "rsicd-sha"

    validation = [
        {"step": step, "loss": 1.5 - step / 1000} for step in range(0, 501, 50)
    ]
    (run / "validation.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in validation), encoding="utf-8"
    )
    _write_json(
        run / "skyscript_step0_parity.json",
        {
            "format_version": 2,
            "status": "pass",
            "checkpoint_step": 0,
            "input": {"manifest_sha256": val_sha},
        },
    )
    for name, right_sha in (
        ("train_vs_val_overlap.json", val_sha),
        ("train_vs_rsicd_val_overlap.json", rsicd_sha),
    ):
        _write_json(
            gate / name,
            {
                "status": "clear",
                "exact_file_overlap_count": 0,
                "exact_decoded_pixel_overlap_count": 0,
                "left_manifest": {"sha256": train_sha},
                "right_manifest": {"sha256": right_sha},
            },
        )

    sky_zero = _metrics(0.08)
    rsicd_zero = _metrics(0.16)
    _write_json(
        gate / "skyscript_val_step000.json",
        _evaluation(
            task="paired_image_text_global_retrieval",
            manifest_sha=val_sha,
            step=0,
            metrics=sky_zero,
            source="SkyScript",
        ),
    )
    _write_json(
        gate / "rsicd_val_official.json",
        _evaluation(
            task="rsicd_image_text_retrieval",
            manifest_sha=rsicd_sha,
            step=None,
            metrics=rsicd_zero,
            source=None,
        ),
    )
    _write_json(
        gate / "rsicd_val_step000.json",
        _evaluation(
            task="rsicd_image_text_retrieval",
            manifest_sha=rsicd_sha,
            step=0,
            metrics=rsicd_zero,
            source=None,
        ),
    )

    training_reports: dict[int, Path] = {}
    for stage in STAGES:
        resume_steps = () if stage == 100 else ((100,) if stage == 250 else (100, 250))
        _write_json(
            gate / f"skyscript_val_step{stage:03d}.json",
            _evaluation(
                task="paired_image_text_global_retrieval",
                manifest_sha=val_sha,
                step=stage,
                metrics=_metrics(0.08 + stage / 10000),
                source="SkyScript",
            ),
        )
        _write_json(
            gate / f"rsicd_val_step{stage:03d}.json",
            _evaluation(
                task="rsicd_image_text_retrieval",
                manifest_sha=rsicd_sha,
                step=stage,
                metrics=_metrics(0.16 + stage / 100000),
                source=None,
            ),
        )
        report_path = gate / f"training_step{stage}.json"
        _write_json(
            report_path,
            {
                "steps": stage,
                "completed": stage == 500,
                "final_queue_size": 0,
                "best_validation_step": stage,
                "train_manifest_sha256": train_sha,
                "val_manifest_sha256": val_sha,
                "project_commit": "project",
                "dinov3_commit": "dinov3",
                "resume_history": [
                    {"checkpoint_step": resume_step}
                    for resume_step in resume_steps
                ],
            },
        )
        training_reports[stage] = report_path
    return run, gate, training_reports


def test_summarize_stages_and_final_gate(tmp_path: Path) -> None:
    run, gate, training_reports = _prepare_evidence(tmp_path)
    for stage in STAGES:
        summary = summarize_stage(
            run_dir=run,
            gate_dir=gate,
            stage=stage,
            training_report_path=training_reports[stage],
        )
        assert summary["status"] == "pass"
        assert summary["validation_loss"]["best_step"] == stage
        _write_json(gate / f"stage_{stage}_summary.json", summary)

    s1 = tmp_path / "s1.json"
    _write_json(s1, {"gate": "SkyScript_S1", "status": "pass", "seeds": [11, 23, 47]})
    final = summarize_final(s1_summary_path=s1, gate_dir=gate)

    assert final["status"] == "pass"
    assert final["stages"] == [100, 250, 500]
    assert final["validation_loss"]["best_step"] == 500
    assert final["skyscript"]["current_minus_step0_mean_recall"]["500"] > 0


def test_stage_summary_records_numerical_failure(tmp_path: Path) -> None:
    run, gate, training_reports = _prepare_evidence(tmp_path)
    _write_json(
        gate / "skyscript_val_step100.json",
        _evaluation(
            task="paired_image_text_global_retrieval",
            manifest_sha="val-sha",
            step=100,
            metrics=_metrics(0.07),
            source="SkyScript",
        ),
    )

    summary = summarize_stage(
        run_dir=run,
        gate_dir=gate,
        stage=100,
        training_report_path=training_reports[100],
    )

    assert summary["status"] == "fail"
    assert not summary["checks"]["skyscript_current_mean_recall_above_step0"]
