#!/usr/bin/env python3
"""Verify the frozen Web/SAT configuration pair for SkyScript M4-A."""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

APPROVED_DIFFERENCES = {
    ("experiment", "name"),
    ("experiment", "output_dir"),
    ("model", "backbone_domain"),
    ("model", "backbone_weights"),
}

EXPECTED_IDENTITIES = {
    "web": {
        ("experiment", "name"): (
            "skyscript_images23_top30raw_imageadapter256_36495_500step_seed11"
        ),
        ("experiment", "output_dir"): (
            "outputs/skyscript_images23_top30raw_imageadapter256_36495_500step_seed11"
        ),
        ("model", "backbone_domain"): "web",
        ("model", "backbone_weights"): (
            "assets/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        ),
    },
    "sat": {
        ("experiment", "name"): (
            "skyscript_images23_top30raw_sat_imageadapter256_36495_500step_seed11"
        ),
        ("experiment", "output_dir"): (
            "outputs/skyscript_images23_top30raw_sat_imageadapter256_36495_500step_seed11"
        ),
        ("model", "backbone_domain"): "sat",
        ("model", "backbone_weights"): (
            "assets/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-config", required=True, type=Path)
    parser.add_argument("--sat-config", required=True, type=Path)
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _flatten(
    value: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], Any]:
    leaves: dict[tuple[str, ...], Any] = {}
    for key, child in value.items():
        path = (*prefix, key)
        if isinstance(child, Mapping):
            leaves.update(_flatten(child, path))
        else:
            leaves[path] = child
    return leaves


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def verify_config_pair(web: Mapping[str, Any], sat: Mapping[str, Any]) -> None:
    web_leaves = _flatten(web)
    sat_leaves = _flatten(sat)
    if web_leaves.keys() != sat_leaves.keys():
        only_web = sorted(_format_path(path) for path in web_leaves.keys() - sat_leaves.keys())
        only_sat = sorted(_format_path(path) for path in sat_leaves.keys() - web_leaves.keys())
        raise ValueError(
            f"Web/SAT config fields differ: only_web={only_web}, only_sat={only_sat}"
        )

    observed_differences = {
        path for path in web_leaves if web_leaves[path] != sat_leaves[path]
    }
    if observed_differences != APPROVED_DIFFERENCES:
        missing = sorted(
            _format_path(path) for path in APPROVED_DIFFERENCES - observed_differences
        )
        unexpected = sorted(
            _format_path(path) for path in observed_differences - APPROVED_DIFFERENCES
        )
        raise ValueError(
            "M4-A config differences do not match the frozen protocol: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for side, leaves in (("web", web_leaves), ("sat", sat_leaves)):
        for path, expected in EXPECTED_IDENTITIES[side].items():
            observed = leaves[path]
            if observed != expected:
                raise ValueError(
                    f"Unexpected {side} {_format_path(path)}: "
                    f"expected {expected!r}, got {observed!r}"
                )

    if web_leaves[("experiment", "seed")] != sat_leaves[("experiment", "seed")]:
        raise ValueError("Web/SAT experiment seeds must match")
    if web_leaves[("experiment", "seed")] != 11:
        raise ValueError("M4-A is frozen to seed 11")


def main() -> None:
    args = parse_args()
    verify_config_pair(load_toml(args.web_config), load_toml(args.sat_config))
    print("m4_a_config_protocol=verified")
    print("approved_differences=" + ",".join(map(_format_path, sorted(APPROVED_DIFFERENCES))))


if __name__ == "__main__":
    main()
