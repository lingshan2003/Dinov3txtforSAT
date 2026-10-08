from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from tools import run_web_mixed_pilot as runner
from tools.bundle_skyscript_mixed_pilot import bundle_mixed_pilot, reconstruct_group_split
from tools.install_skyscript_mixed_pilot import install_mixed_pilot

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(root: Path, monkeypatch) -> Path:
    monkeypatch.setattr(runner, "TRAIN_IMAGES", 2)
    monkeypatch.setattr(runner, "VAL_IMAGES", 2)
    config_path = root / runner.CONFIG
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes((ROOT / runner.CONFIG).read_bytes())
    reference_config = (
        ROOT / "configs/skyscript_web_adapter_textlora_maskpos_fullimage1epoch_seed11.toml"
    )
    target_reference = (
        root / "configs/skyscript_web_adapter_textlora_maskpos_fullimage1epoch_seed11.toml"
    )
    target_reference.write_bytes(reference_config.read_bytes())

    for relative in (
        "external/dinov3",
        "assets/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "assets/checkpoints/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        "assets/checkpoints/bpe_simple_vocab_16e6.txt.gz",
    ):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True) if relative.endswith(
            "dinov3"
        ) else path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            path.write_bytes(b"fixture")

    image_dir = root / "images"
    image_dir.mkdir()
    groups = {"train": [], "val": []}
    rows_by_split: dict[str, list[dict]] = {"train": [], "val": []}
    alternatives = {}
    for split in ("train", "val"):
        for index in range(2):
            image = image_dir / f"{split}-{index}.jpg"
            image.write_bytes(b"image")
            short = f"short {split} {index} unique caption with original wording"
            detail = f"detail {split} {index} unique caption with original wording"
            group = " ".join(short.split()).casefold()
            reference = {
                "id": f"original:{split}:{index}",
                "image": str(image),
                "caption": short,
                "split": split,
                "source": "SkyScript",
            }
            groups[split].append(reference)
            alternatives[(split, index)] = (short, detail)
            kind = ("short", "detail")[index]
            original_caption = short if kind == "short" else detail
            caption = original_caption[:24]
            row = {
                **reference,
                "id": f"skyscript:{split}:{index}",
                "caption": caption,
                "group_id": " ".join(caption.split()).casefold(),
                "caption_kind": kind,
                "selected_caption_group": " ".join(group.split()).casefold(),
                "short_caption": short,
                "detail_caption": detail,
                "original_caption": original_caption,
                "csv_filepath": f"images/{split}-{index}.jpg",
                "image_id": f"skyscript:{split}:{index}",
            }
            rows_by_split[split].append(row)
    _write_jsonl(root / runner.TRAIN_REFERENCE, groups["train"])
    _write_jsonl(root / runner.VAL_REFERENCE, groups["val"])
    _write_jsonl(root / runner.ASSET_DIR / "train_mixed.jsonl", rows_by_split["train"])

    mixed_rows = rows_by_split["val"]
    short_rows, detail_rows = [], []
    for index, row in enumerate(mixed_rows):
        short, detail = alternatives[("val", index)]
        short_rows.append(
            {
                **row,
                "caption": short[:24],
                "original_caption": short,
                "group_id": short[:24].casefold(),
                "caption_kind": "short",
            }
        )
        detail_rows.append(
            {
                **row,
                "caption": detail[:24],
                "original_caption": detail,
                "group_id": detail[:24].casefold(),
                "caption_kind": "detail",
            }
        )
    _write_jsonl(root / runner.ASSET_DIR / "val_mixed.jsonl", mixed_rows)
    _write_jsonl(root / runner.ASSET_DIR / "val_short.jsonl", short_rows)
    _write_jsonl(root / runner.ASSET_DIR / "val_detail.jsonl", detail_rows)
    audit = {
        "schema": "skyscript_mixed_pilot_v1",
        "splits": {"train": 2, "val": 2, "val_short": 2, "val_detail": 2},
        "selected_kinds": {
            "train": {"short": 1, "detail": 1},
            "val": {"short": 1, "detail": 1},
        },
        "mixed_ratio": {
            "train": {"short": 1, "detail": 1},
            "val": {"short": 1, "detail": 1},
        },
        "selected_group_provenance": [
            {
                "group": row["selected_caption_group"],
                "split": row["split"],
                "filepath": row["csv_filepath"],
                "caption_kind": row["caption_kind"],
            }
            for split_rows in rows_by_split.values()
            for row in split_rows
        ],
    }
    (root / runner.ASSET_DIR / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    rs_image = image_dir / "rs.jpg"
    rs_image.write_bytes(b"image")
    _write_jsonl(
        root / runner.RSICD_MANIFEST,
        [
            {
                "id": "rs-cap",
                "image_id": "rs-image",
                "image": str(rs_image),
                "caption": "rs caption",
                "split": "val",
                "source": "RSICD",
            }
        ],
    )
    return root


def test_pilot_config_keeps_native_web_recipe_and_expected_budget(tmp_path):
    trial = runner.validated_trial(ROOT)
    assert trial.method == "adapter_lora"
    assert trial.output.name == "skyscript_web_mixed_pilot_16k_1to1_3epoch_seed11"
    from dinotxt_rs.config import load_config

    config = load_config(trial.config)
    assert (config.train.image_epochs, config.train.max_steps, config.train.warmup_steps) == (
        3,
        750,
        75,
    )
    assert config.data.caption_sampling == "image_epoch"
    assert config.data.images_per_caption == 2
    assert config.train.contrastive_objective == "mask_same_caption"


def test_data_preflight_checks_original_split_fixed_ratio_and_alternatives(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "pilot", monkeypatch)
    train, pools, audit = runner._preflight_data(root)
    assert len(train) == 2
    assert set(pools) == {"pilotmixed", "pilotshort", "pilotdetail"}
    assert audit["schema"] == "skyscript_mixed_pilot_v1"

    rows = runner._read_manifest_rows(root / runner.ASSET_DIR / "train_mixed.jsonl")
    rows[1]["selected_caption_group"] = rows[0]["selected_caption_group"]
    _write_jsonl(root / runner.ASSET_DIR / "train_mixed.jsonl", rows)
    with pytest.raises(ValueError, match="duplicate|original train groups|ratio|selected group"):
        runner._preflight_data(root)


class _WordTokenizer:
    def encode(self, text):
        return text.split()


def test_installer_output_with_truncated_captions_passes_runner_preflight(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "pilot-project", monkeypatch)
    monkeypatch.setattr(runner, "TRAIN_IMAGES", 4)
    short_rows, detail_rows = [], []
    grouped = {2: [], 3: []}
    for index in range(8):
        filepath = f"images{2 + index % 2}/a{100000 + index}_US_{10 + index}.jpg"
        short_caption = f"Building zone{index} sector"
        detail_caption = (
            f"Satellite view zone{index} reveals warehouse beside a service road "
            "and storage tanks across the industrial district today clearly."
        )
        short_rows.append(
            {"filepath": filepath, "title_raw": short_caption, "title": short_caption}
        )
        detail_rows.append({"filepath": filepath, "title_multi_objects": detail_caption})
        grouped[2 + index % 2].append((filepath, f"image-bytes-{index}".encode()))

    short_csv, detail_csv = tmp_path / "short.csv", tmp_path / "detail.csv"
    with short_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filepath", "title_raw", "title"))
        writer.writeheader()
        writer.writerows(short_rows)
    with detail_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filepath", "title_multi_objects"))
        writer.writeheader()
        writer.writerows(detail_rows)
    archives = []
    for prefix in (2, 3):
        archive_path = tmp_path / f"images{prefix}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for filepath, content in grouped[prefix]:
                archive.writestr(filepath, content)
        archives.append(archive_path)

    bundle = tmp_path / "mixed-pilot.zip"
    bundle_mixed_pilot(
        csv_path=short_csv,
        detail_csv_path=detail_csv,
        archives=archives,
        output=bundle,
        train_count=4,
        val_count=2,
        split_val_count=2,
        split_seed=23,
        seed=11,
    )
    split = reconstruct_group_split(short_rows, val_count=2, seed=23)
    image_root = tmp_path / "source-images"
    for index, row in enumerate(short_rows):
        image = image_root / row["filepath"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-bytes-{index}".encode())
    references = {}
    for split_name in ("train", "val"):
        rows = []
        for index, row in enumerate(short_rows):
            group = row["title_raw"].casefold()
            if split[group] != split_name:
                continue
            rows.append(
                {
                    "id": f"original:{split_name}:{index}",
                    "image": str(image_root / row["filepath"]),
                    "caption": row["title_raw"],
                    "split": split_name,
                    "source": "SkyScript",
                }
            )
        path = tmp_path / f"{split_name}-reference.jsonl"
        _write_jsonl(path, rows)
        references[split_name] = path
        _write_jsonl(
            root / (runner.TRAIN_REFERENCE if split_name == "train" else runner.VAL_REFERENCE), rows
        )

    shutil.rmtree(root / runner.ASSET_DIR)
    install_mixed_pilot(
        bundle=bundle,
        train_reference=references["train"],
        val_reference=references["val"],
        output_dir=root / runner.ASSET_DIR,
        tokenizer=_WordTokenizer(),
        context_length=5,
    )
    train, pools, audit = runner._preflight_data(root)
    assert len(train) == 4
    assert all(len(pool) == 2 for pool in pools.values())
    assert audit["complete_word_backoff"]["detail"] > 0
    assert all(
        row["group_id"] == " ".join(row["caption"].split()).casefold()
        for name in ("pilotmixed", "pilotshort", "pilotdetail")
        for row in pools[name]
    )


def test_preflight_only_has_no_output_and_never_trains(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "pilot", monkeypatch)
    executions = []
    monkeypatch.setattr(runner.shared, "_execute", lambda *args, **kwargs: executions.append(args))
    assert runner.run_pipeline(root, "python", mode="preflight-only") is None
    assert not (root / runner.REPORT_DIR).exists()
    assert executions == []


def test_train_only_runs_once_then_skips_completed_run(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "pilot", monkeypatch)
    monkeypatch.setattr(
        runner,
        "_pool_counts",
        lambda *args, **kwargs: {
            "pilotmixed": {"images": 2, "captions": 2, "pairs": 2},
            "pilotshort": {"images": 2, "captions": 2, "pairs": 2},
            "pilotdetail": {"images": 2, "captions": 2, "pairs": 2},
            "rsicd": {"images": 1, "captions": 1},
        },
    )
    state = {"value": "fresh"}
    monkeypatch.setattr(runner, "_validate_training_state", lambda *args, **kwargs: state["value"])
    calls = []

    def execute(command, log, *, root):
        calls.append(command)
        state["value"] = "complete"

    monkeypatch.setattr(runner.shared, "_execute", execute)
    monkeypatch.setattr(runner, "_package", lambda root, trial: root / "reports.tar.gz")
    result = runner.run_pipeline(root, "python", mode="train-only")
    assert result == root / "reports.tar.gz"
    assert len(calls) == 1 and "dinotxt_rs.cli.train" in calls[0]
    runner.run_pipeline(root, "python", mode="train-only")
    assert len(calls) == 1


def test_default_mode_orders_baseline_train_and_evaluates_each_tag_once(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "pilot", monkeypatch)
    monkeypatch.setattr(
        runner,
        "_pool_counts",
        lambda *args, **kwargs: {
            "pilotmixed": {"images": 2, "captions": 2, "pairs": 2},
            "pilotshort": {"images": 2, "captions": 2, "pairs": 2},
            "pilotdetail": {"images": 2, "captions": 2, "pairs": 2},
            "rsicd": {"images": 1, "captions": 1},
        },
    )
    state = {"value": "fresh"}
    monkeypatch.setattr(runner, "_validate_training_state", lambda *args, **kwargs: state["value"])
    monkeypatch.setattr(runner, "_validate_pilot_report", lambda *args, **kwargs: None)
    events = []

    def official(root, python, trial, counts):
        events.append("official")
        directory = root / runner.REPORT_DIR / "official"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.toml").write_bytes(trial.config.read_bytes())
        for pool in runner.POOLS:
            (directory / f"{pool}.json").write_text("{}", encoding="utf-8")

    def execute(command, log, *, root):
        if "dinotxt_rs.cli.train" in command:
            events.append("train")
            state["value"] = "complete"
            return
        events.append("evaluate")
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")

    def comparison(root, trial, counts):
        reports = list((root / runner.REPORT_DIR / trial.method).glob("*.json"))
        official_reports = list((root / runner.REPORT_DIR / "official").glob("*.json"))
        assert len(reports) == 12
        assert len(official_reports) == 4
        if "train" in events:
            assert events.index("official") < events.index("train")
            assert events.index("train") < events.index("evaluate")
        else:
            assert "evaluate" not in events
        events.append("comparison")

    monkeypatch.setattr(runner, "_official_baseline", official)
    monkeypatch.setattr(runner.shared, "_execute", execute)
    monkeypatch.setattr(runner, "_write_comparison", comparison)
    monkeypatch.setattr(
        runner, "_package", lambda root, trial: events.append("package") or root / "reports.tar.gz"
    )

    assert runner.run_pipeline(root, "python") == root / "reports.tar.gz"
    assert events.count("official") == 1
    assert events.count("train") == 1
    assert events.count("evaluate") == 12
    assert events.count("comparison") == 1

    events.clear()
    assert runner.run_pipeline(root, "python") == root / "reports.tar.gz"
    assert "train" not in events and "evaluate" not in events
    assert events.count("comparison") == 1
    assert events.count("package") == 1


def test_resume_uses_latest_checkpoint_and_evaluate_only_rejects_incomplete(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "pilot", monkeypatch)
    monkeypatch.setattr(
        runner,
        "_pool_counts",
        lambda *args, **kwargs: {
            "pilotmixed": {"images": 2, "captions": 2, "pairs": 2},
            "pilotshort": {"images": 2, "captions": 2, "pairs": 2},
            "pilotdetail": {"images": 2, "captions": 2, "pairs": 2},
            "rsicd": {"images": 1, "captions": 1},
        },
    )
    state = {"value": "resume"}
    monkeypatch.setattr(runner, "_validate_training_state", lambda *args, **kwargs: state["value"])
    monkeypatch.setattr(runner, "_package", lambda root, trial: root / "reports.tar.gz")
    commands = []

    def execute(command, log, *, root):
        commands.append(command)
        state["value"] = "complete"

    monkeypatch.setattr(runner.shared, "_execute", execute)
    runner.run_pipeline(root, "python", mode="train-only")
    assert commands == [
        [
            "python",
            "-u",
            "-m",
            "dinotxt_rs.cli.train",
            "--config",
            str(root / runner.CONFIG),
            "--resume",
            str(root / "outputs" / runner.SERIES / "latest.pt"),
        ]
    ]

    state["value"] = "fresh"
    with pytest.raises(RuntimeError, match="requires completed pilot training"):
        runner.run_pipeline(root, "python", mode="evaluate-only")


def test_existing_reports_without_pool_fingerprint_are_rejected(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "pilot", monkeypatch)
    trial = runner.validated_trial(root)
    report_dir = root / runner.REPORT_DIR
    (report_dir / "official").mkdir(parents=True)
    (report_dir / "official/pilotmixed.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="without a manifest/config identity"):
        runner._ensure_cache_identity(root, trial)


def test_cached_pilot_reports_reject_changed_manifest_content(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "pilot", monkeypatch)
    trial = runner.validated_trial(root)
    runner._ensure_cache_identity(root, trial)
    path = root / runner.MANIFESTS["pilotmixed"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cache identity changed"):
        runner._ensure_cache_identity(root, trial)
