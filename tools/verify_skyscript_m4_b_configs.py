#!/usr/bin/env python3
"""Verify the complete frozen Web/SAT seed matrix for SkyScript M4-B."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.verify_skyscript_m4_config_pair import _flatten, load_toml

SEEDS = (11, 23, 47)
DOMAINS = ("web", "sat")
SEED_DIFFERENCES = {
    ("experiment", "name"),
    ("experiment", "seed"),
    ("experiment", "output_dir"),
}
DOMAIN_DIFFERENCES = {
    ("experiment", "name"),
    ("experiment", "output_dir"),
    ("model", "backbone_domain"),
    ("model", "backbone_weights"),
}
BACKBONES = {
    "web": "assets/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "sat": "assets/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    return parser.parse_args()


def config_path(config_dir: Path, domain: str, seed: int) -> Path:
    return config_dir / f"skyscript_{domain}_adapter_500step_seed{seed}.toml"


def experiment_name(domain: str, seed: int) -> str:
    infix = "sat_" if domain == "sat" else ""
    return (
        f"skyscript_images23_top30raw_{infix}imageadapter256_36495_"
        f"500step_seed{seed}"
    )


def _differences(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> set[tuple[str, ...]]:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    if left_flat.keys() != right_flat.keys():
        raise ValueError("M4-B configs do not contain the same fields")
    return {path for path in left_flat if left_flat[path] != right_flat[path]}


def verify_config_matrix(configs: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    expected_keys = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
    if set(configs) != expected_keys:
        raise ValueError("M4-B config matrix must contain Web/SAT seeds 11, 23, and 47")

    for domain in DOMAINS:
        reference = configs[(domain, 11)]
        for seed in SEEDS:
            observed = configs[(domain, seed)]
            name = experiment_name(domain, seed)
            expected_identity = {
                ("experiment", "name"): name,
                ("experiment", "seed"): seed,
                ("experiment", "output_dir"): f"outputs/{name}",
                ("model", "backbone_domain"): domain,
                ("model", "backbone_weights"): BACKBONES[domain],
            }
            flat = _flatten(observed)
            for path, expected in expected_identity.items():
                if flat.get(path) != expected:
                    dotted = ".".join(path)
                    raise ValueError(
                        f"Unexpected {domain} seed{seed} {dotted}: "
                        f"expected {expected!r}, got {flat.get(path)!r}"
                    )
            expected_differences = set() if seed == 11 else SEED_DIFFERENCES
            differences = _differences(reference, observed)
            if differences != expected_differences:
                dotted = sorted(".".join(path) for path in differences)
                raise ValueError(
                    f"Unexpected within-domain differences for {domain} seed{seed}: {dotted}"
                )

    for seed in SEEDS:
        differences = _differences(configs[("web", seed)], configs[("sat", seed)])
        if differences != DOMAIN_DIFFERENCES:
            dotted = sorted(".".join(path) for path in differences)
            raise ValueError(f"Unexpected Web/SAT differences for seed{seed}: {dotted}")


def load_config_matrix(config_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (domain, seed): load_toml(config_path(config_dir, domain, seed))
        for domain in DOMAINS
        for seed in SEEDS
    }


def main() -> None:
    args = parse_args()
    verify_config_matrix(load_config_matrix(args.config_dir))
    print("m4_b_config_matrix=verified")
    print("domains=web,sat seeds=11,23,47 schedule=500 warmup=50")


if __name__ == "__main__":
    main()
