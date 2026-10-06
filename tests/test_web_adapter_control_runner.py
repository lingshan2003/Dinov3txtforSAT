from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest

from dinotxt_rs.config import load_config
from dinotxt_rs.evaluation.retrieval import RETRIEVAL_TIE_POLICY
from tools import run_sat_full_image_epoch as shared
from tools import run_web_adapter_control as runner
from tools import run_web_native as native
from tools.run_sat_multi_positive import Trial

ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "control project"
    (root / "configs").mkdir(parents=True)
    for stem in native.VARIANTS.values():
        source = ROOT / "configs" / f"{stem}.toml"
        (root / "configs" / source.name).write_bytes(source.read_bytes())
    reference = ROOT / "configs/skyscript_sat_adapter_textlora_maskpos_fullimage1epoch_seed11.toml"
    (root / "configs" / reference.name).write_bytes(reference.read_bytes())
    return root


def _trials(root: Path) -> list[Trial]:
    return native.validated_trials(root, runner.METHODS)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_train_only_selects_two_trials_and_does_not_read_baseline(tmp_path, monkeypatch):
    root = _project(tmp_path)
    events = []
    monkeypatch.setattr(runner, "_training_state", lambda root, trial: "complete")
    monkeypatch.setattr(runner.native, "_official_baseline", lambda *args: pytest.fail("baseline"))
    monkeypatch.setattr(runner, "_copy_cached_reports", lambda *args: pytest.fail("cache"))
    monkeypatch.setattr(shared, "run_pipeline", lambda *args, **kwargs: events.append(
        (kwargs["mode"], [trial.method for trial in kwargs["trials"]])
    ))

    runner.run_pipeline(root, sys.executable, mode="train-only")

    assert events == [("train-only", ["adapter_lora", "adapter_only"])]


@pytest.mark.parametrize("state", ["fresh", "resume", "missing"])
def test_control_refuses_unfinished_primary_before_preflight_or_gpu_work(
    tmp_path, monkeypatch, state,
):
    root = _project(tmp_path)
    events = []
    monkeypatch.setattr(runner, "_training_state", lambda root, trial: state)
    monkeypatch.setattr(shared, "run_pipeline", lambda *args, **kwargs: events.append("shared"))

    with pytest.raises(RuntimeError, match=r"requires completed adapter\+LoRA training"):
        runner.run_pipeline(root, sys.executable)

    assert events == []


def test_completed_primary_allows_control_and_rerun_delegates_both_for_skip_logic(
    tmp_path, monkeypatch,
):
    root = _project(tmp_path)
    states = {"adapter_lora": "complete", "adapter_only": "fresh"}
    events = []
    monkeypatch.setattr(runner, "_training_state", lambda root, trial: states[trial.method])
    monkeypatch.setattr(
        runner.native, "_official_baseline", lambda *args: events.append("baseline")
    )
    monkeypatch.setattr(runner, "_copy_cached_reports", lambda *args: events.append("copy"))
    monkeypatch.setattr(runner, "_write_comparison", lambda *args: events.append("comparison"))
    monkeypatch.setattr(shared, "_pool_counts", lambda root: {})
    monkeypatch.setattr(shared, "run_pipeline", lambda *args, **kwargs: events.append(
        (kwargs["mode"], [trial.method for trial in kwargs["trials"]])
    ))
    monkeypatch.setattr(shared, "_package", lambda *args, **kwargs: Path("archive.tar.gz"))

    assert runner.run_pipeline(root, sys.executable) == Path("archive.tar.gz")
    assert events == [
        ("preflight-only", ["adapter_lora", "adapter_only"]),
        "baseline", "copy", ("all", ["adapter_lora", "adapter_only"]), "comparison",
    ]

    events.clear()
    states["adapter_only"] = "complete"
    runner.run_pipeline(root, sys.executable)
    assert ("all", ["adapter_lora", "adapter_only"]) in events


def test_preflight_is_read_only_and_evaluate_only_requires_completed_control(
    tmp_path, monkeypatch,
):
    root = _project(tmp_path)
    states = {"adapter_lora": "complete", "adapter_only": "fresh"}
    events = []
    monkeypatch.setattr(runner, "_training_state", lambda root, trial: states[trial.method])
    monkeypatch.setattr(shared, "run_pipeline", lambda *args, **kwargs: events.append(
        ("shared", kwargs["mode"])
    ))
    monkeypatch.setattr(
        runner.native, "_official_baseline", lambda *args: events.append("baseline")
    )

    assert runner.run_pipeline(root, sys.executable, mode="preflight-only") is None
    assert events == [("shared", "preflight-only")]

    events.clear()
    with pytest.raises(RuntimeError, match="requires completed adapter-only training"):
        runner.run_pipeline(root, sys.executable, mode="evaluate-only")
    assert events == [("shared", "preflight-only")]


def test_cached_reports_are_validated_before_copy_and_identity_failure_stops_copy(
    tmp_path, monkeypatch,
):
    root = _project(tmp_path)
    primary = _trials(root)[0]
    counts = {name: {"images": 2, "texts": 3} for name in shared.MANIFESTS}
    seen = []

    def validate_official(path, *args):
        seen.append(("official", path))
        if path.name == "rsicd.json":
            raise ValueError("Unexpected native official reference identity")

    monkeypatch.setattr(native, "validate_official_report", validate_official)
    monkeypatch.setattr(shared, "validate_report", lambda path, *args: seen.append(("model", path)))
    for dataset in shared.MANIFESTS:
        _json(root / native.REPORT_DIR / "official" / f"{dataset}.json", {})
        for tag in shared.TAGS:
            _json(root / native.REPORT_DIR / primary.method / f"{dataset}_{tag}.json", {})

    with pytest.raises(ValueError, match="Unexpected native official"):
        runner._copy_cached_reports(root, primary, counts)

    assert ("official", root / native.REPORT_DIR / "official/rsicd.json") in seen
    assert not (root / runner.REPORT_DIR / "official/rsicd.json").exists()


def _metrics(base: float, *, grouped: bool = False) -> dict:
    direction = {"r1": base, "r5": base + .1, "r10": base + .2,
                 "mean_rank": 4.0, "median_rank": 2.0}
    value = {"image_to_text": dict(direction), "text_to_image": dict(direction),
             "mean_recall": base + .1}
    if grouped:
        value["image_to_text_group_balanced"] = {
            "r1": base, "r5": base + .1, "r10": base + .2,
        }
        value["group_balanced_mean_recall"] = base + .1
    return value


def _seed_comparison_inputs(
    root: Path, monkeypatch,
) -> tuple[list[Trial], dict[str, dict[str, int]]]:
    trials = _trials(root)
    manifests = {name: Path("manifests") / f"{name}.jsonl" for name in shared.MANIFESTS}
    monkeypatch.setattr(shared, "MANIFESTS", manifests)
    counts = {name: {"images": 2, "texts": 3} for name in manifests}
    monkeypatch.setattr(shared, "_pool_counts", lambda root: counts)
    monkeypatch.setattr(native, "validate_official_report", lambda *args: None)
    monkeypatch.setattr(shared, "validate_report", lambda *args: None)
    lora_config = load_config(trials[0].config)
    assets = {
        key: {
            "path": str(
                (lora_config.source.parent.parent / getattr(lora_config.model, key)).resolve()
            )
        }
        for key in ("backbone_weights", "dinotxt_weights", "bpe_vocab")
    }
    for dataset, relative in manifests.items():
        manifest = root / relative
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("", encoding="utf-8")
        definitions = {}
        if dataset == "skyscript_unique":
            definitions["positive_definition"] = "manifest_row_one_to_one"
        elif dataset == "skyscript_group":
            definitions.update(
                positive_definition="normalized_complete_caption_group",
                recall_definition="fraction_of_queries_with_any_positive_in_top_k",
            )
        official = {
            "manifest": {"path": str(manifest)}, "counts": counts[dataset],
            "tie_policy": RETRIEVAL_TIE_POLICY, **definitions,
            "model": assets, "metrics": _metrics(.2, grouped=dataset == "skyscript_group"),
        }
        _json(root / runner.REPORT_DIR / "official" / f"{dataset}.json", official)
        for trial, base in zip(trials, (.6, .4), strict=True):
            _json(trial.output / "training_summary.json", {
                "steps": 5269, "validation": {"best_step": 5000, "final_loss": 1.25},
                "optimizer_parameter_groups": {"trainable_parameters": 1234},
            })
            for tag in shared.TAGS:
                report = {
                    "manifest": {"path": str(manifest)}, "counts": counts[dataset],
                    "tie_policy": RETRIEVAL_TIE_POLICY, **definitions,
                    "model": assets,
                    "metrics": _metrics(base, grouped=dataset == "skyscript_group"),
                }
                _json(root / runner.REPORT_DIR / trial.method / f"{dataset}_{tag}.json", report)
    return trials, counts


def test_comparison_reports_lora_minus_adapter_and_rejects_protocol_mismatch(
    tmp_path, monkeypatch,
):
    root = _project(tmp_path)
    _seed_comparison_inputs(root, monkeypatch)
    runner._write_comparison(root, _trials(root))

    result = json.loads((root / runner.REPORT_DIR / "comparison.json").read_text())
    unique = result["comparisons"]["latest"]["skyscript_unique"]
    assert unique["delta_lora_minus_adapter_only_pp"]["mean_recall"] == pytest.approx(20.0)
    assert unique["delta_to_official_pp"]["adapter_lora"]["mean_recall"] == pytest.approx(40.0)
    assert unique["delta_to_official_pp"]["adapter_only"]["mean_recall"] == pytest.approx(20.0)
    assert "| unique-val |" in (root / runner.REPORT_DIR / "comparison.md").read_text()
    assert "image_to_text" in (root / runner.REPORT_DIR / "comparison.md").read_text()

    path = root / runner.REPORT_DIR / "adapter_only/skyscript_unique_latest.json"
    value = json.loads(path.read_text())
    value["tie_policy"] = "different-policy"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="protocols differ"):
        runner._write_comparison(root, _trials(root))


def test_report_package_contains_metadata_and_comparison_but_no_checkpoint_weights(
    tmp_path,
):
    root = _project(tmp_path)
    trials = _trials(root)
    audit = root / shared.GROUP_DIR / "audit.json"
    _json(audit, {"splits": {"train": {"records": 2}}})
    for trial in trials:
        _json(trial.output / "training_summary.json", {"completed": True})
        (trial.output / "latest.pt").parent.mkdir(parents=True, exist_ok=True)
        (trial.output / "latest.pt").write_bytes(b"checkpoint")
    report_dir = root / runner.REPORT_DIR
    _json(report_dir / "comparison.json", {"ok": True})
    (report_dir / "comparison.md").write_text("comparison", encoding="utf-8")
    (report_dir / "official/config.toml").parent.mkdir(parents=True, exist_ok=True)
    (report_dir / "official/config.toml").write_text("config", encoding="utf-8")
    (report_dir / "ignored.pt").write_bytes(b"weights")

    archive = shared._package(root, trials, report_dir=runner.REPORT_DIR, series=runner.SERIES)

    with tarfile.open(archive) as package:
        members = set(package.getnames())
    assert str(trials[0].output.relative_to(root) / "training_summary.json") in members
    assert str(trials[1].output.relative_to(root) / "training_summary.json") in members
    assert str(runner.REPORT_DIR / "comparison.json") in members
    assert str(runner.REPORT_DIR / "comparison.md") in members
    assert str(runner.REPORT_DIR / "official/config.toml") in members
    assert not any(name.endswith((".pt", ".part")) for name in members)
