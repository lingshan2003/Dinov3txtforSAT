#!/usr/bin/env python3
"""Compare staged-run preflight files with commit changes treated as advisory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ADVISORY_FIELDS = {"project_commit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stored", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    return parser.parse_args()


def read_preflight(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "=" not in line:
            raise ValueError(f"Invalid preflight line: {path}:{line_number}")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"Invalid or duplicate preflight key: {path}:{line_number}")
        result[key] = value
    return result


def compare_preflights(stored: dict[str, str], current: dict[str, str]) -> dict[str, object]:
    blocking_fields = sorted(
        key
        for key in stored.keys() | current.keys()
        if key not in ADVISORY_FIELDS and stored.get(key) != current.get(key)
    )
    advisory_fields = sorted(
        key
        for key in ADVISORY_FIELDS
        if key in stored or key in current
        if stored.get(key) != current.get(key)
    )
    return {
        "status": "fail" if blocking_fields else ("warning" if advisory_fields else "match"),
        "blocking_fields": blocking_fields,
        "advisory_fields": advisory_fields,
        "stored_project_commit": stored.get("project_commit"),
        "current_project_commit": current.get("project_commit"),
    }


def main() -> None:
    args = parse_args()
    result = compare_preflights(read_preflight(args.stored), read_preflight(args.current))
    if result["advisory_fields"]:
        print(
            "WARNING: project commit changed since the staged run began: "
            f"stored={result['stored_project_commit']!r}, "
            f"current={result['current_project_commit']!r}. Continuing because all frozen "
            "protocol fields match; the invocation will be recorded.",
            file=sys.stderr,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["blocking_fields"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
