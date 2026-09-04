import json

from tools.verify_downstream_evaluation import LABELS, verify_downstream_evaluation


def _eurosat(label: str) -> dict[str, object]:
    return {
        "task": "eurosat_zero_shot_classification",
        "manifest": {"sha256": "eurosat"},
        "metrics": {"top1_accuracy": 0.4, "mean_per_class_accuracy": 0.3},
        "model": {"checkpoint": None if label.endswith("official") else {"step": 300}},
    }


def _rsicd() -> dict[str, object]:
    metric = {"r1": 0.1, "r5": 0.2, "r10": 0.3, "median_rank": 9.0}
    return {
        "task": "rsicd_image_text_retrieval",
        "split": "test",
        "manifest": {"sha256": "rsicd"},
        "metrics": {"image_to_text": metric, "text_to_image": metric},
    }


def test_verify_downstream_evaluation_accepts_comparable_reports(tmp_path) -> None:
    for label in LABELS:
        (tmp_path / f"eurosat_{label}.json").write_text(
            json.dumps(_eurosat(label)), encoding="utf-8"
        )
        (tmp_path / f"rsicd_{label}.json").write_text(json.dumps(_rsicd()), encoding="utf-8")

    report = verify_downstream_evaluation(tmp_path)

    assert report["status"] == "complete"
    assert report["models"]["web_500step_best"]["checkpoint_step"] == 300
