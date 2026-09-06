from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DOMAINS = ("web", "sat")
EXPECTED_COMPARISONS = {
    "image_features",
    "text_features",
    "logit_scale",
    "patch_tokens",
    "backbone_patch_tokens",
    "similarity_logits",
    "symmetric_contrastive_loss",
}


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_step0_parity(output_dir: Path) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    manifest_hash: str | None = None
    for domain in DOMAINS:
        path = output_dir / f"{domain}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing parity report: {path}")
        report = _read_report(path)
        if report.get("status") != "pass":
            raise ValueError(f"{domain} step-0 parity did not pass")
        if report.get("backbone_domain") != domain:
            raise ValueError(f"Unexpected backbone domain in {path}")
        if report.get("checkpoint_step") != 0:
            raise ValueError(f"{domain} report did not use a step-0 checkpoint")
        input_record = report.get("input")
        if not isinstance(input_record, dict) or not isinstance(
            input_record.get("manifest_sha256"), str
        ):
            raise ValueError(f"{domain} report has no input manifest SHA-256")
        observed_hash = input_record["manifest_sha256"]
        if manifest_hash is None:
            manifest_hash = observed_hash
        elif observed_hash != manifest_hash:
            raise ValueError("Web and SAT parity reports used different input manifests")
        comparisons = report.get("comparisons")
        if not isinstance(comparisons, dict) or set(comparisons) != EXPECTED_COMPARISONS:
            raise ValueError(f"{domain} report has incomplete tensor comparisons")
        if not all(
            isinstance(value, dict) and value.get("passed") is True
            for value in comparisons.values()
        ):
            raise ValueError(f"{domain} report contains a failed tensor comparison")
        tokens = report.get("tokens")
        trainable_parameters = report.get("trainable_parameters")
        if not isinstance(tokens, dict) or tokens.get("passed") is not True:
            raise ValueError(f"{domain} report did not verify identical tokens")
        if (
            not isinstance(trainable_parameters, dict)
            or trainable_parameters.get("passed") is not True
        ):
            raise ValueError(f"{domain} report did not verify trainable parameter identity")
        step0_model = report.get("step0_model")
        checkpoint = step0_model.get("checkpoint") if isinstance(step0_model, dict) else None
        reports[domain] = {
            "report": str(path.resolve()),
            "project_commit": report.get("project_commit"),
            "precision": report.get("precision"),
            "checkpoint_sha256": (
                checkpoint.get("sha256") if isinstance(checkpoint, dict) else None
            ),
            "input_identity_sha256": input_record.get("identity_sha256"),
            "max_absolute_errors": {
                name: value.get("max_absolute_error") for name, value in comparisons.items()
            },
        }
    return {
        "format_version": 1,
        "status": "complete",
        "gate": "A_step0_parity",
        "input_manifest_sha256": manifest_hash,
        "models": reports,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Web and SAT step-0 parity reports")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite parity verification: {args.report}")
    report = verify_step0_parity(args.output_dir)
    _write_json_atomic(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
