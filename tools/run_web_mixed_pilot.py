#!/usr/bin/env python3
"""Run and report the fixed-caption 16k SkyScript Web pilot."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinotxt_rs.config import load_config, required_paths  # noqa: E402
from dinotxt_rs.evaluation.common import write_json_atomic  # noqa: E402
from dinotxt_rs.evaluation.retrieval import load_paired_records, load_rsicd_records  # noqa: E402
from tools import run_sat_full_image_epoch as shared  # noqa: E402
from tools import run_web_native as native  # noqa: E402
from tools.run_sat_multi_positive import Trial  # noqa: E402

SERIES = "skyscript_web_mixed_pilot_16k_1to1_3epoch_seed11"
REPORT_DIR = Path("outputs/web_mixed_pilot_seed11_reports")
ASSET_DIR = Path("assets/data/manifests/skyscript_mixed_pilot_v1")
CONFIG = Path(f"configs/{SERIES}.toml")
TRAIN_REFERENCE = Path(
    "assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl"
)
VAL_REFERENCE = Path(
    "assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
)
RSICD_MANIFEST = Path("assets/data/manifests/rsicd_val_retrieval_v1.jsonl")
TRAIN_IMAGES = 16_000
VAL_IMAGES = 1_600
EPOCHS = 3
STEPS = 750
TAGS = ("step_0000000", "best", "latest")
SKY_POOLS = ("pilotmixed", "pilotshort", "pilotdetail")
POOLS = (*SKY_POOLS, "rsicd")
MANIFESTS = {
    "pilotmixed": ASSET_DIR / "val_mixed.jsonl",
    "pilotshort": ASSET_DIR / "val_short.jsonl",
    "pilotdetail": ASSET_DIR / "val_detail.jsonl",
    "rsicd": RSICD_MANIFEST,
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL row {path}:{number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {path}:{number}")
        rows.append(row)
    return rows


def validated_trial(root: Path) -> Trial:
    root = root.resolve()
    trial = Trial("adapter_lora", root / CONFIG, root / "outputs" / SERIES)
    config = load_config(trial.config)
    reference = load_config(
        root / "configs/skyscript_web_adapter_textlora_maskpos_fullimage1epoch_seed11.toml"
    )
    expected_data = replace(
        reference.data,
        train_manifest=ASSET_DIR / "train_mixed.jsonl",
        val_manifest=ASSET_DIR / "val_mixed.jsonl",
    )
    expected_train = replace(
        reference.train,
        max_steps=STEPS,
        image_epochs=EPOCHS,
        warmup_steps=75,
    )
    if (
        config.experiment
        != replace(
            reference.experiment,
            name=SERIES,
            output_dir=Path("outputs") / SERIES,
        )
        or config.model != reference.model
        or config.data != expected_data
        or config.train != expected_train
    ):
        raise ValueError(f"Unexpected mixed-pilot training protocol: {trial.config}")
    if (
        config.model.backbone_domain != "web"
        or config.model.image_adapter_bottleneck != 256
        or config.model.text_last_k != 0
        or config.model.text_lora_rank != 8
        or config.model.text_lora_alpha != 16.0
        or not config.model.text_lora_include_projection
        or config.data.caption_sampling != "image_epoch"
        or config.data.images_per_caption != 2
        or config.data.num_workers != 0
        or config.data.train_augmentation
        or config.train.contrastive_objective != "mask_same_caption"
        or config.train.queue_size != 0
    ):
        raise ValueError(f"Unexpected sampling/objective protocol: {trial.config}")
    return trial


def _validate_audit(root: Path) -> dict[str, Any]:
    path = root / ASSET_DIR / "audit.json"
    audit = _json(path)
    # The installer audit is part of the provenance record; verify it describes the
    # exact fixed pools and split selection consumed below.
    if audit.get("schema") not in {"skyscript_mixed_pilot_v1", "skyscript_mixed_pilot_v1_audit"}:
        raise ValueError(f"Unexpected mixed-pilot audit schema: {path}")
    return audit


def _crc32(path: Path) -> str:
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return f"{crc & 0xFFFFFFFF:08x}"


def _ensure_cache_identity(root: Path, trial: Trial) -> None:
    directory = root / REPORT_DIR
    sidecar = directory / "cache_identity.json"
    sources = {"config": trial.config, **{name: root / path for name, path in MANIFESTS.items()}}
    identity = {
        "scheme": "crc32-size-v1",
        "files": {
            name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "crc32": _crc32(path)}
            for name, path in sources.items()
        },
    }
    if sidecar.exists():
        if _json(sidecar) != identity:
            raise ValueError(f"Pilot evaluation cache identity changed: {sidecar}")
        return
    cached_reports = [path for path in directory.rglob("*.json") if path != sidecar]
    if cached_reports:
        raise ValueError(
            f"Pilot reports exist without a manifest/config identity record: {cached_reports[0]}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(sidecar, identity)


def _check_audit_counts(audit: dict[str, Any], split: str, *, records: int) -> None:
    splits = audit.get("splits", {})
    entry = splits.get(split) if isinstance(splits, dict) else None
    if entry is None:
        counts = audit.get("counts", {})
        entry = counts.get(split) if isinstance(counts, dict) else None
    if isinstance(entry, int):
        count = entry
    elif isinstance(entry, dict):
        count = entry.get("records", entry.get("images"))
    else:
        raise ValueError(f"Mixed-pilot audit is missing split counts for {split}")
    if count != records:
        raise ValueError(f"Mixed-pilot audit count mismatch for {split}: {count} != {records}")


def _validate_rows(
    rows: list[dict[str, Any]], *, split: str, kind: str | None = None
) -> list[dict[str, str]]:
    if len(rows) != (TRAIN_IMAGES if split == "train" else VAL_IMAGES):
        raise ValueError(
            f"Expected {TRAIN_IMAGES if split == 'train' else VAL_IMAGES} {split} rows"
        )
    for row in rows:
        for key in ("id", "image_id", "image", "caption", "source", "split", "group_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Pilot manifest row is missing {key}: {split}")
        if row["id"] != row["image_id"]:
            raise ValueError(f"Pilot {split} sample id must remain stable per image")
        if row["split"] != split or row["source"] != "SkyScript":
            raise ValueError(f"Unexpected pilot row source/split in {split}")
        if not Path(row["image"]).is_file():
            raise FileNotFoundError(f"Pilot image does not exist: {row['image']}")
        if kind is not None and row.get("caption_kind") != kind:
            raise ValueError(f"Expected caption_kind={kind} in {split} manifest")
        actual_caption = row.get("original_caption")
        caption_kind = row.get("caption_kind")
        if not isinstance(actual_caption, str) or not actual_caption.strip():
            raise ValueError(f"Pilot {split} row is missing original_caption")
        if row.get("group_id") != _normalized(row["caption"]):
            raise ValueError(f"Pilot {split} group_id does not match the fitted caption")
        alternative = row.get(f"{caption_kind}_caption")
        if actual_caption != alternative:
            raise ValueError(
                f"Pilot {split} selected caption does not match {caption_kind} alternative"
            )
        if row.get("selected_caption_group") != _normalized(row.get("short_caption", "")):
            raise ValueError(
                f"Pilot {split} selected group is not the original short caption group"
            )
    ids = [row["id"] for row in rows]
    image_ids = [row["image_id"] for row in rows]
    images = [Path(row["image"]).resolve() for row in rows]
    captions = [_normalized(row["caption"]) for row in rows]
    if (
        len(set(ids)) != len(rows)
        or len(set(image_ids)) != len(rows)
        or len(set(images)) != len(rows)
        or len(set(captions)) != len(rows)
    ):
        raise ValueError(f"Pilot {split} rows must have unique ids, images, and captions")
    return rows


def _preflight_data(
    root: Path,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    trial = validated_trial(root)
    config = load_config(trial.config)
    missing = [path if path.is_absolute() else root / path for path in required_paths(config)]
    missing += [root / path for path in (CONFIG, TRAIN_REFERENCE, VAL_REFERENCE, RSICD_MANIFEST)]
    missing += [
        root / ASSET_DIR / name
        for name in (
            "train_mixed.jsonl",
            "val_mixed.jsonl",
            "val_short.jsonl",
            "val_detail.jsonl",
            "audit.json",
        )
    ]
    absent = [path for path in missing if not path.exists()]
    if absent:
        raise FileNotFoundError("Missing required pilot inputs: " + ", ".join(map(str, absent)))

    audit = _validate_audit(root)
    train_rows = _read_manifest_rows(root / ASSET_DIR / "train_mixed.jsonl")
    mixed_rows = _read_manifest_rows(root / ASSET_DIR / "val_mixed.jsonl")
    short_rows = _read_manifest_rows(root / ASSET_DIR / "val_short.jsonl")
    detail_rows = _read_manifest_rows(root / ASSET_DIR / "val_detail.jsonl")
    train = _validate_rows(train_rows, split="train")
    pools = {
        "pilotmixed": _validate_rows(mixed_rows, split="val"),
        "pilotshort": _validate_rows(short_rows, split="val", kind="short"),
        "pilotdetail": _validate_rows(detail_rows, split="val", kind="detail"),
    }
    # Require exact one-to-one identity and equal short/detail sample coverage.
    mixed_images = {(row["image_id"], Path(row["image"]).resolve()) for row in pools["pilotmixed"]}
    if any(
        {(row["image_id"], Path(row["image"]).resolve()) for row in pools[name]} != mixed_images
        for name in ("pilotshort", "pilotdetail")
    ):
        raise ValueError("Mixed, short, and detail validation pools must cover identical images")
    for split_name, rows in (("train", train_rows), ("val", mixed_rows)):
        kinds = {
            kind: sum(row.get("caption_kind") == kind for row in rows)
            for kind in ("short", "detail")
        }
        expected = TRAIN_IMAGES // 2 if split_name == "train" else VAL_IMAGES // 2
        if kinds != {"short": expected, "detail": expected}:
            raise ValueError(f"Pilot {split_name} must be a fixed 1:1 short/detail mix")

    # Confirm the selected original caption groups stay in their source split, and
    # that neither image paths nor ids cross train/validation.
    original_train = load_paired_records(
        root / TRAIN_REFERENCE, expected_split="train", expected_source="SkyScript"
    )
    original_val = load_paired_records(
        root / VAL_REFERENCE, expected_split="val", expected_source="SkyScript"
    )
    train_groups = {_normalized(row["caption"]) for row in original_train}
    val_groups = {_normalized(row["caption"]) for row in original_val}
    selected_train_groups = {row.get("selected_caption_group") for row in train_rows}
    selected_val_groups = {row.get("selected_caption_group") for row in mixed_rows}
    if (
        None in selected_train_groups
        or len(selected_train_groups) != len(train_rows)
        or not selected_train_groups <= train_groups
    ):
        raise ValueError("Pilot train selected groups are not a subset of original train groups")
    if (
        None in selected_val_groups
        or len(selected_val_groups) != len(mixed_rows)
        or not selected_val_groups <= val_groups
    ):
        raise ValueError("Pilot validation selected groups are not a subset of original val groups")
    if selected_train_groups & selected_val_groups:
        raise ValueError("Pilot selected original caption groups overlap train and validation")
    train_images = {Path(row["image"]).resolve() for row in train}
    val_images = {Path(row["image"]).resolve() for row in pools["pilotmixed"]}
    if train_images & val_images or {row["image_id"] for row in train_rows} & {
        row["image_id"] for row in mixed_rows
    }:
        raise ValueError("Pilot training and validation image/id sets overlap")
    if {row["group_id"] for row in train_rows} & {row["group_id"] for row in mixed_rows}:
        raise ValueError("Pilot fixed training and validation captions overlap")

    # Every original short/detail alternative is unique across both splits, while
    # the train manifest uses only its fixed selected caption.
    alternatives = []
    for row in [*train_rows, *mixed_rows]:
        for field in ("short_caption", "detail_caption"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Pilot row is missing the {field} provenance field")
            alternatives.append(_normalized(value))
    if len(set(alternatives)) != len(alternatives):
        raise ValueError("Short/detail alternative captions must all be unique")
    _check_audit_counts(audit, "train", records=TRAIN_IMAGES)
    _check_audit_counts(audit, "val", records=VAL_IMAGES)
    _check_audit_counts(audit, "val_short", records=VAL_IMAGES)
    _check_audit_counts(audit, "val_detail", records=VAL_IMAGES)
    expected_kinds = {
        "train": {"short": TRAIN_IMAGES // 2, "detail": TRAIN_IMAGES // 2},
        "val": {"short": VAL_IMAGES // 2, "detail": VAL_IMAGES // 2},
    }
    if audit.get("selected_kinds") != expected_kinds or audit.get("mixed_ratio") != expected_kinds:
        raise ValueError("Mixed-pilot audit does not confirm the fixed 1:1 caption ratio")
    provenance = audit.get("selected_group_provenance")
    expected_provenance = {
        (
            row["selected_caption_group"],
            row["split"],
            row["csv_filepath"],
            row["caption_kind"],
        )
        for row in (*train_rows, *mixed_rows)
    }
    actual_provenance = {
        (
            row.get("group"),
            row.get("split"),
            row.get("filepath"),
            row.get("caption_kind"),
        )
        for row in provenance or []
        if isinstance(row, dict)
    }
    if actual_provenance != expected_provenance:
        raise ValueError("Audit selected-group provenance differs from the fixed manifests")
    return train, pools, audit


def _validate_training_state(trial: Trial, *, images: int = TRAIN_IMAGES) -> str:
    config = load_config(trial.config)
    return shared.full_training_state(
        trial,
        images=images,
        epochs=config.train.image_epochs or 0,
        steps=config.train.max_steps,
    )


def _pool_counts(root: Path, *, training_images: set[Path]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for name in SKY_POOLS:
        records = load_paired_records(
            root / MANIFESTS[name], expected_split="val", expected_source="SkyScript"
        )
        images = {Path(row["image"]).resolve() for row in records}
        if images & training_images:
            raise ValueError(f"Pilot train and {name} image sets overlap")
        counts[name] = {"images": len(records), "captions": len(records), "pairs": len(records)}
    rs_images, rs_texts = load_rsicd_records(root / RSICD_MANIFEST, expected_split="val")
    if training_images & {Path(row["image"]).resolve() for row in rs_images}:
        raise ValueError("Pilot training and RSICD validation images overlap")
    counts["rsicd"] = {"images": len(rs_images), "captions": len(rs_texts)}
    return counts


def _validate_pilot_report(
    report: Path,
    trial: Trial,
    pool: str,
    tag: str,
    manifest: Path,
    counts: dict[str, int],
    *,
    official: bool = False,
) -> None:
    if official:
        alias = "rsicd" if pool == "rsicd" else "skyscript_unique"
        native.validate_official_report(report, load_config(trial.config), alias, manifest, counts)
        return
    alias = "rsicd" if pool == "rsicd" else "skyscript_unique"
    shared.validate_report(report, trial, alias, tag, manifest, counts)


def _official_baseline(
    root: Path, python: str, trial: Trial, counts: dict[str, dict[str, int]]
) -> None:
    config = load_config(trial.config)
    directory = root / REPORT_DIR / "official"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "config.toml"
    if snapshot.exists() and snapshot.read_bytes() != config.source.read_bytes():
        raise ValueError(f"Official baseline config identity mismatch: {snapshot}")
    if not snapshot.exists():
        if any(directory.glob("*.json")):
            raise ValueError(f"Cached official baseline has no config snapshot: {directory}")
        snapshot.write_bytes(config.source.read_bytes())
    for pool in POOLS:
        report = directory / f"{pool}.json"
        manifest = root / MANIFESTS[pool]
        if not report.exists():
            cli = "evaluate_rsicd" if pool == "rsicd" else "evaluate_skyscript"
            command = [
                python,
                "-u",
                "-m",
                f"dinotxt_rs.cli.{cli}",
                "--config",
                str(config.source),
                "--manifest",
                str(manifest),
                "--official",
                "--split",
                "val",
                "--output",
                str(report),
            ]
            shared._execute(command, directory / "retrieval.log", root=root)
        _validate_pilot_report(
            report, trial, pool, "official", manifest, counts[pool], official=True
        )


def _delta_pp(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    values = {"mean_recall": 100 * (after["mean_recall"] - before["mean_recall"])}
    for direction in ("image_to_text", "text_to_image"):
        for key in ("r1", "r5", "r10"):
            values[f"{direction}_{key}"] = 100 * (after[direction][key] - before[direction][key])
    return values


def _write_comparison(root: Path, trial: Trial, counts: dict[str, dict[str, int]]) -> None:
    directory = root / REPORT_DIR
    official = {}
    reports: dict[str, dict[str, Any]] = {}
    parity = {}
    for pool in POOLS:
        manifest = root / MANIFESTS[pool]
        official_path = directory / "official" / f"{pool}.json"
        _validate_pilot_report(
            official_path, trial, pool, "official", manifest, counts[pool], official=True
        )
        baseline = _json(official_path)
        official[pool] = baseline["metrics"]
        reports[pool], parity[pool] = {}, False
        for tag in TAGS:
            report_path = directory / trial.method / f"{pool}_{tag}.json"
            _validate_pilot_report(report_path, trial, pool, tag, manifest, counts[pool])
            report = _json(report_path)
            if any(
                report.get(key) != baseline.get(key)
                for key in (
                    "manifest",
                    "counts",
                    "tie_policy",
                    "positive_definition",
                    "recall_definition",
                )
            ):
                raise ValueError(f"Pilot retrieval protocols differ: {report_path}")
            for key in ("backbone_weights", "dinotxt_weights", "bpe_vocab"):
                if report["model"][key] != baseline["model"][key]:
                    raise ValueError(f"Pilot model assets differ: {report_path}")
            reports[pool][tag] = report["metrics"]
        parity[pool] = reports[pool]["step_0000000"] == official[pool]
        if not parity[pool]:
            raise ValueError(f"Step-0 evaluation does not match the official model on {pool}")
    training = _json(trial.output / "training_summary.json")
    comparisons = {
        tag: {
            pool: {
                "official": official[pool],
                "adapter_lora": reports[pool][tag],
                "adapter_lora_delta_to_official_pp": _delta_pp(reports[pool][tag], official[pool]),
            }
            for pool in POOLS
        }
        for tag in ("best", "latest")
    }
    write_json_atomic(
        directory / "comparison.json",
        {
            "series": SERIES,
            "primary_checkpoint": "latest",
            "counts": counts,
            "training": training,
            "official": official,
            "step0_metrics_match_official": parity,
            "comparisons": comparisons,
            "interpretation": (
                "Compare each adapted checkpoint with its same-pool official baseline; "
                "differences in pool difficulty are not method gains."
            ),
        },
    )
    lines = [
        "# SkyScript Web fixed-caption pilot",
        "",
        (
            "16,000 training images, one fixed caption per image, 8,000 short and "
            "8,000 detail; three complete image epochs (48,000 exposures, 750 steps)."
        ),
        (
            "Compare each validation pool only with its own official baseline. "
            "Pool Recall differences reflect pool composition, not method gains."
        ),
        "",
        (
            "| Checkpoint | Pool | Official mean Recall | Adapted mean Recall | "
            "Delta to same-pool official (pp) | I→T R@1/5/10 delta (pp) | "
            "T→I R@1/5/10 delta (pp) |"
        ),
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for tag in ("best", "latest"):
        for pool in POOLS:
            value = comparisons[tag][pool]
            delta = value["adapter_lora_delta_to_official_pp"]
            lines.append(
                f"| {tag} | {pool} | {value['official']['mean_recall'] * 100:.4f}% | "
                f"{value['adapter_lora']['mean_recall'] * 100:.4f}% | "
                f"{delta['mean_recall']:+.4f} | "
                f"{delta['image_to_text_r1']:+.4f}/"
                f"{delta['image_to_text_r5']:+.4f}/"
                f"{delta['image_to_text_r10']:+.4f} | "
                f"{delta['text_to_image_r1']:+.4f}/"
                f"{delta['text_to_image_r5']:+.4f}/"
                f"{delta['text_to_image_r10']:+.4f} |"
            )
    lines += ["", f"Step 0 matches official on all pools: {all(parity.values())}", ""]
    (directory / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _package(root: Path, trial: Trial) -> Path:
    archive = root / "outputs/web_mixed_pilot_seed11_reports.tar.gz"
    temporary = archive.with_name(archive.name + ".part")
    paths: set[Path] = {
        root / ASSET_DIR / "audit.json",
        root / ASSET_DIR / "train_mixed.jsonl",
    }
    paths.update(root / MANIFESTS[name] for name in POOLS if name != "rsicd")
    paths.add(root / RSICD_MANIFEST)
    if trial.output.exists():
        paths.update(
            path
            for path in trial.output.iterdir()
            if path.is_file()
            and path.suffix
            in {
                ".json",
                ".jsonl",
                ".toml",
                ".log",
            }
        )
    if (root / REPORT_DIR).exists():
        paths.update(
            path
            for path in (root / REPORT_DIR).rglob("*")
            if path.is_file()
            and path.suffix
            in {
                ".json",
                ".jsonl",
                ".toml",
                ".log",
                ".md",
            }
        )
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(temporary, "w:gz") as bundle:
            for path in sorted(paths):
                if path.exists() and not path.is_symlink():
                    bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"reports_archive={archive}", flush=True)
    return archive


def run_pipeline(root: Path, python: str, *, mode: str = "all") -> Path | None:
    if mode not in {"all", "train-only", "evaluate-only", "preflight-only"}:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    root = root.resolve()
    trial = validated_trial(root)
    train, pools, audit = _preflight_data(root)
    state = _validate_training_state(trial)
    if mode == "evaluate-only" and state != "complete":
        raise RuntimeError(f"--evaluate-only requires completed pilot training: {trial.output}")
    training_images = {Path(row["image"]).resolve() for row in train}
    counts = _pool_counts(root, training_images=training_images)
    for pool in SKY_POOLS:
        if len(pools[pool]) != counts[pool]["images"]:
            raise ValueError(f"Pilot manifest count changed during preflight: {pool}")
    if mode == "preflight-only":
        print(
            f"preflight_complete=true train_images={len(train)} "
            f"val_images={VAL_IMAGES} state={state}",
            flush=True,
        )
        return None

    _ensure_cache_identity(root, trial)

    report_directory = root / REPORT_DIR
    for pool in POOLS:
        for tag in TAGS:
            report = report_directory / trial.method / f"{pool}_{tag}.json"
            if report.exists() and state != "complete":
                raise RuntimeError(f"Cached evaluation exists for incomplete training: {report}")
            if report.exists():
                _validate_pilot_report(
                    report, trial, pool, tag, root / MANIFESTS[pool], counts[pool]
                )
        baseline = report_directory / "official" / f"{pool}.json"
        if baseline.exists():
            _validate_pilot_report(
                baseline,
                trial,
                pool,
                "official",
                root / MANIFESTS[pool],
                counts[pool],
                official=True,
            )

    if mode != "train-only":
        _official_baseline(root, python, trial, counts)

    if mode != "evaluate-only" and state != "complete":
        command = [python, "-u", "-m", "dinotxt_rs.cli.train", "--config", str(trial.config)]
        if state == "resume":
            command += ["--resume", str(trial.output / "latest.pt")]
        shared._execute(command, trial.output / "train.log", root=root)
        if _validate_training_state(trial) != "complete":
            raise RuntimeError(
                f"Pilot training did not complete full image coverage: {trial.output}"
            )
    if mode != "train-only":
        for pool in POOLS:
            for tag in TAGS:
                report = report_directory / trial.method / f"{pool}_{tag}.json"
                if report.exists():
                    _validate_pilot_report(
                        report, trial, pool, tag, root / MANIFESTS[pool], counts[pool]
                    )
                    continue
                cli = "evaluate_rsicd" if pool == "rsicd" else "evaluate_skyscript"
                command = [
                    python,
                    "-u",
                    "-m",
                    f"dinotxt_rs.cli.{cli}",
                    "--config",
                    str(trial.output / "config.toml"),
                    "--manifest",
                    str(root / MANIFESTS[pool]),
                    "--checkpoint",
                    str(trial.output / f"{tag}.pt"),
                    "--training-output",
                    str(trial.output),
                    "--split",
                    "val",
                    "--output",
                    str(report),
                ]
                if pool != "rsicd":
                    command += ["--positive-definition", "one-to-one"]
                shared._execute(command, report.parent / "retrieval.log", root=root)
                _validate_pilot_report(
                    report, trial, pool, tag, root / MANIFESTS[pool], counts[pool]
                )
        _write_comparison(root, trial, counts)
    return _package(root, trial)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ("preflight-only", "train-only", "evaluate-only"):
        modes.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    mode = next(
        (
            name
            for name in ("preflight-only", "train-only", "evaluate-only")
            if getattr(args, name.replace("-", "_"))
        ),
        "all",
    )
    run_pipeline(Path(__file__).resolve().parents[1], sys.executable, mode=mode)


if __name__ == "__main__":
    main()
