#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Verify the pinned Python benchmark dependency cohort."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import tomllib


def _module_version_matches_distribution(
    module_version: str, distribution_version: str
) -> bool:
    """Accept a module's public version after the exact wheel pin is checked.

    ROCm Triton wheels use a PEP 440 local-version suffix to identify the source
    and ROCm cohort in package metadata, while ``triton.__version__`` reports
    only the public release (for example, ``3.8.0``).  The distribution metadata
    remains the authoritative exact pin; this secondary import check verifies
    that the imported module belongs to the same public release.
    """

    public_version = distribution_version.partition("+")[0]
    return module_version in {distribution_version, public_version}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    arguments = parser.parse_args()

    lock_path = arguments.lock.expanduser().resolve()
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"cannot read dependency lock {lock_path}: {error}", file=sys.stderr)
        return 2

    try:
        expected = {
            "rocm_sdk": lock["rocm_sdk"]["version"],
            "torch": lock["pytorch"]["version"],
            "triton": lock["triton"]["version"],
            "hipblaslt": lock["hipblaslt"]["distribution_version"],
        }
        distributions = {
            "rocm_sdk": lock["rocm_sdk"]["distribution"],
            "torch": lock["pytorch"]["distribution"],
            "triton": lock["triton"]["distribution"],
            "hipblaslt": lock["hipblaslt"]["distribution"],
        }
        required_python = str(lock["rocm_sdk"]["python"])
        expected_hipblaslt_version = str(lock["hipblaslt"]["package_version"])
    except (KeyError, TypeError) as error:
        print(f"invalid dependency lock {lock_path}: missing {error}", file=sys.stderr)
        return 2
    actual: dict[str, str | None] = {}
    errors: list[str] = []
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != required_python:
        errors.append(f"python=={actual_python}; expected {required_python}")
    for package, distribution in distributions.items():
        try:
            actual[package] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual[package] = None
        if actual[package] != expected[package]:
            errors.append(
                f"{distribution}=={actual[package]!s}; expected {expected[package]}"
            )

    if not errors:
        try:
            import torch
            import triton

            if not torch.version.hip:
                errors.append("torch is not a ROCm build")
            if not _module_version_matches_distribution(
                triton.__version__, expected["triton"]
            ):
                errors.append(
                    f"triton.__version__={triton.__version__}; "
                    f"expected public version "
                    f"{expected['triton'].partition('+')[0]}"
                )
        except Exception as error:  # Return imports cleanly to CMake/preflight.
            errors.append(f"cannot import pinned torch/triton cohort: {error}")

    report = {
        "schema": "rocjitsu.benchmark.environment.v1",
        "lock": str(lock_path),
        "expected": expected,
        "actual": actual,
        "python": {"actual": actual_python, "expected": required_python},
        "hipblaslt": {"expected_package_version": expected_hipblaslt_version},
        "passed": not errors,
        "errors": errors,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
