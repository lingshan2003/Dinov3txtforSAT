import hashlib
import json
from pathlib import Path

import pytest

import tools.summarize_skyscript_m4_b as m4b


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _metrics(value: float) -> dict:
    return {
        "image_to_text": {
            "r1": value,
            "r5": value + 0.01,
            "r10": value + 0.02,
            "median_rank": 50 - value,
            "mean_rank": 100 - value,
        },
        "text_to_image": {
            "r1": value + 0.001,
            "r5": value + 0.011,
            "r10": value + 0.021,
            "median_rank": 40 - value,
            "mean_rank": 90 - value,
        },
        "mean_recall": value + 0.0105,
    }


def _stage_report(domain: str, seed: int, stage: int) -> dict:
    baseline = (0.08 if domain == "web" else 0.01) + seed / 100_000
    gate = (
        "SkyScript_S2_stage"
        if (domain, seed) == ("web", 11)
        else "SkyScript_M4_SAT_stage"
        if (domain, seed) == ("sat", 11)
        else "SkyScript_M4_B_stage"
    )
    report = {
        "gate": gate,
        "stage": stage,
        "status": "pass",
        "checks": {"all": True},
        "training": {
            "train_manifest_sha256": m4b.TRAIN_SHA,
            "val_manifest_sha256": m4b.VAL_SHA,
            "dinov3_commit": m4b.DINOV3_COMMIT,
        },
        "validation_loss": {
            "steps": {"0": 2.0 + baseline, str(stage): 1.0 + baseline}
        },
        "skyscript": {
            "step0": _metrics(baseline),
            f"step{stage}": _metrics(baseline + 0.03),
        },
        "rsicd_val": {
            "official": _metrics(baseline + 0.05),
            f"step{stage}": _metrics(baseline + 0.06),
        },
    }
    if seed != 11:
        report.update({"domain": domain, "seed": seed})
    return report


def test_three_seed_summary_reports_per_seed_stats_and_paired_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m4b, "load_config_matrix", lambda _: {})
    monkeypatch.setattr(m4b, "verify_config_matrix", lambda _: None)
    prerequisite = tmp_path / "m4_a.json"
    _write_json(prerequisite, {"gate": "SkyScript_M4_RQ1_seed11", "status": "pass"})
    monkeypatch.setattr(
        m4b, "M4_A_SUMMARY_SHA", hashlib.sha256(prerequisite.read_bytes()).hexdigest()
    )
    for domain in m4b.DOMAINS:
        for seed in m4b.SEEDS:
            config = m4b.config_path(Path("configs"), domain, seed)
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text("# synthetic\n", encoding="utf-8")
            _write_json(
                m4b.stage_report_path(domain, seed, 100),
                _stage_report(domain, seed, 100),
            )

    summary = m4b.summarize(100, prerequisite)

    assert summary["status"] == "pass"
    assert summary["seeds"] == [11, 23, 47]
    sky = summary["skyscript"]
    assert set(sky["per_seed"]) == {"11", "23", "47"}
    assert sky["domain_statistics"]["web"]["absolute"]["mean_recall"][
        "sample_std"
    ] > 0
    assert sky["paired_sat_minus_web_statistics"]["absolute"]["mean_recall"][
        "mean"
    ] == pytest.approx(-0.07)
    assert sky["paired_sat_minus_web_statistics"]["within_domain_delta"][
        "mean_recall"
    ]["mean"] == pytest.approx(0.0)
