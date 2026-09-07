from __future__ import annotations

import json

from tools.verify_m4_abc_development import DESIGNS, verify_m4_abc_development


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _retrieval(step, mean_recall, trainable) -> dict[str, object]:
    checkpoint = None if step is None else {"step": step, "sha256": f"checkpoint-{step}"}
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
    _write_json(path / "verification_report.json", {"steps": 100, "completed": False})
    (path / "validation.jsonl").write_text(
        "".join(
            json.dumps({"step": step, "loss": loss}) + "\n"
            for step, loss in zip((0, 50, 100), validation, strict=True)
        ),
        encoding="utf-8",
    )


def test_verify_m4_abc_development_selects_retained_improving_runs(tmp_path) -> None:
    development = tmp_path / "development"
    development.mkdir()
    _write_json(development / "train_vs_rsicd_val_overlap.json", {"status": "clear"})
    _write_json(development / "rsicd_val_official.json", _retrieval(None, 0.2, 100))
    runs = {}
    retrieval = {
        "a": (0.19, 0.18, 100),
        "b": (0.2, 0.195, 10),
        "c": (0.205, 0.21, 10),
    }
    for label in DESIGNS:
        run = tmp_path / f"run-{label}"
        _write_run(run, label, (3.8, 3.7, 3.6))
        runs[label] = run
        step50, step100, trainable = retrieval[label]
        _write_json(
            development / f"rsicd_val_{label}_step50.json",
            _retrieval(50, step50, trainable),
        )
        _write_json(
            development / f"rsicd_val_{label}_step100.json",
            _retrieval(100, step100, trainable),
        )

    report = verify_m4_abc_development(development, runs, retention_margin=0.01)

    assert report["status"] == "complete"
    assert report["runs"]["a"]["rsicd_val_mean_recall"]["retention_pass"] is False
    assert report["promotion_eligible"] == ["c", "b"]
    assert report["recommended_for_250step"] == "c"
