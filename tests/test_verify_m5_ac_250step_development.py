from __future__ import annotations

import json

from tools.verify_m5_ac_250step_development import (
    DESIGNS,
    verify_m5_ac_250step_development,
)


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _retrieval(step, mean_recall, trainable, project_commit="m4-commit"):
    checkpoint = None
    if step is not None:
        checkpoint = {
            "step": step,
            "sha256": f"checkpoint-{step}",
            "run_identity": {"project_commit": project_commit},
        }
    return {
        "task": "rsicd_image_text_retrieval",
        "split": "val",
        "manifest": {"sha256": "manifest"},
        "metrics": {"mean_recall": mean_recall},
        "counts": {"images": 10, "captions": 50},
        "model": {
            "checkpoint": checkpoint,
            "trainable_parameters": {"trainable": trainable},
        },
    }


def _write_run(path, label, validation) -> None:
    path.mkdir()
    design = DESIGNS[label]
    (path / "config.toml").write_text(
        "\n".join(
            (
                "[model]",
                f"text_last_k = {design['text_last_k']}",
                f"train_vision_head = {str(design['train_vision_head']).lower()}",
                f"train_text_projection = {str(design['train_text_projection']).lower()}",
                f"train_logit_scale = {str(design['train_logit_scale']).lower()}",
                "[train]",
                f"learning_rate = {design['learning_rate']}",
            )
        ),
        encoding="utf-8",
    )
    _write_json(
        path / "verification_report.json",
        {
            "steps": 250,
            "completed": False,
            "project_commit": "m4-commit",
            "resume_history": [
                {"checkpoint_step": 50},
                {"checkpoint_step": 100},
            ],
        },
    )
    (path / "validation.jsonl").write_text(
        "".join(
            json.dumps({"step": step, "loss": loss}) + "\n"
            for step, loss in zip((0, 50, 100, 150, 200, 250), validation, strict=True)
        ),
        encoding="utf-8",
    )


def _setup(tmp_path, final_metrics):
    development = tmp_path / "development"
    development.mkdir()
    _write_json(
        development / "source_m4_verification_report.json",
        {
            "status": "complete",
            "recommended_for_250step": "a",
            "promotion_eligible": ["a", "c", "b"],
            "rsicd_val": {
                "manifest_sha256": "manifest",
                "counts": {"images": 10, "captions": 50},
                "official_mean_recall": 0.2,
            },
        },
    )
    _write_json(development / "train_vs_rsicd_val_overlap.json", {"status": "clear"})
    _write_json(development / "rsicd_val_official.json", _retrieval(None, 0.2, 100))

    runs = {}
    for label, (loss, recall, trainable) in final_metrics.items():
        run = tmp_path / f"run-{label}"
        _write_run(run, label, (3.8, 3.7, 3.6, 3.55, 3.5, loss))
        runs[label] = run
        recalls = (0.2, 0.199, 0.198, recall)
        for step, observed_recall in zip((100, 150, 200, 250), recalls, strict=True):
            _write_json(
                development / f"rsicd_val_{label}_step{step}.json",
                _retrieval(step, observed_recall, trainable),
            )
    return development, runs


def test_verify_m5_selects_unique_dominant_candidate(tmp_path) -> None:
    development, runs = _setup(
        tmp_path,
        {
            "a": (3.4, 0.205, 100),
            "c": (3.5, 0.195, 10),
        },
    )

    report = verify_m5_ac_250step_development(development, runs)

    assert report["status"] == "complete"
    assert report["eligible_for_500step"] == ["a", "c"]
    assert report["pareto_frontier"] == ["a"]
    assert report["recommended_for_500step"] == "a"
    assert report["decision_status"] == "candidate_selected"


def test_verify_m5_requires_review_for_metric_tradeoff(tmp_path) -> None:
    development, runs = _setup(
        tmp_path,
        {
            "a": (3.4, 0.195, 100),
            "c": (3.5, 0.205, 10),
        },
    )

    report = verify_m5_ac_250step_development(development, runs)

    assert report["pareto_frontier"] == ["a", "c"]
    assert report["recommended_for_500step"] is None
    assert report["decision_status"] == "tradeoff_review_required"
