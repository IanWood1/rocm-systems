#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check compiled workload metadata against the built-in case catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROCJITSU_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROCJITSU_ROOT))

from benchmarks.catalog import parameters_for  # noqa: E402
from benchmarks.runner.manifest import load_suite  # noqa: E402


def _description(executable: Path, case_id: str) -> dict[str, object]:
    completed = subprocess.run(
        [str(executable), "--describe-case", case_id],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError(f"compiled metadata for {case_id} is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--hipblaslt", type=Path, required=True)
    arguments = parser.parse_args()

    suite = load_suite(arguments.manifest)
    executables = {
        "native": arguments.native,
        "hipblaslt": arguments.hipblaslt,
    }
    mismatches: dict[str, object] = {}
    checked = 0
    for case in suite.cases:
        executable = executables.get(case.provider)
        if executable is None:
            continue
        checked += 1
        compiled = _description(executable, case.id)
        expected = parameters_for(case.id)
        if compiled != expected:
            mismatches[case.id] = {"catalog": expected, "compiled": compiled}
    if mismatches:
        raise RuntimeError(
            "compiled workload metadata differs from the catalog: "
            + json.dumps(mismatches, sort_keys=True)
        )
    print(f"validated compiled metadata for {checked} selected cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
