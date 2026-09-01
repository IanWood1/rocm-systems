# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parsing for path-free Rocjitsu benchmark suite manifests."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..catalog import CASE_BY_ID, DEPENDENCY_LOCK, CaseSpec

MANIFEST_SCHEMA = "rocjitsu.benchmark.suite.v1"
DEPENDENCY_SCHEMA = "rocjitsu.benchmark.dependencies.v1"
_TARGET = re.compile(r"^gfx[0-9A-Za-z]+$")
_REQUIRED_TARGETS = frozenset({"gfx950", "gfx1250"})
_ROOT_FIELDS = frozenset(
    {
        "schema",
        "name",
        "description",
        "targets",
        "budget_seconds",
        "cases",
        "measurement",
    }
)
_MEASUREMENT_FIELDS = frozenset({"warmups", "samples"})


class ManifestError(ValueError):
    """Raised when a suite manifest violates the public schema."""


@dataclasses.dataclass(frozen=True)
class MeasurementPolicy:
    warmups: int
    samples: int
    metric: str = "synchronized_dispatch_ns"


@dataclasses.dataclass(frozen=True)
class Suite:
    path: Path
    digest: str
    name: str
    description: str | None
    targets: tuple[str, ...]
    cases: tuple[CaseSpec, ...]
    measurement: MeasurementPolicy
    budget_seconds: float
    dependency_lock: Path
    dependencies: Mapping[str, Any]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field} must be a TOML table")
    return value


def _reject_unknown(
    value: Mapping[str, Any], allowed: frozenset[str], field: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ManifestError(f"{field} contains unsupported fields: {unknown}")


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ManifestError(f"{field} must be an array of strings")
    result = tuple(_string(item, f"{field}[]") for item in value)
    if not result:
        raise ManifestError(f"{field} must not be empty")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{field} must be a non-negative integer")
    return value


def _positive_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ManifestError(f"{field} must be a positive number")
    return float(value)


def _load_dependencies() -> Mapping[str, Any]:
    try:
        value = tomllib.loads(DEPENDENCY_LOCK.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(
            f"cannot parse dependency lock {DEPENDENCY_LOCK}: {exc}"
        ) from exc
    if value.get("schema") != DEPENDENCY_SCHEMA:
        raise ManifestError(f"dependency lock schema must be {DEPENDENCY_SCHEMA!r}")
    return value


def load_suite(path: str | Path) -> Suite:
    """Load a suite whose case definitions come exclusively from the catalog."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        raw_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    try:
        raw = tomllib.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"cannot parse manifest {manifest_path}: {exc}") from exc
    _reject_unknown(raw, _ROOT_FIELDS, "manifest")

    if raw.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(
            f"schema must be {MANIFEST_SCHEMA!r}, got {raw.get('schema')!r}"
        )
    name = _string(raw.get("name"), "name")
    description = (
        _string(raw["description"], "description")
        if "description" in raw
        else None
    )

    targets = _string_list(raw.get("targets"), "targets")
    if len(set(targets)) != len(targets):
        raise ManifestError("targets must not contain duplicates")
    invalid_targets = [target for target in targets if not _TARGET.fullmatch(target)]
    if invalid_targets:
        raise ManifestError(f"invalid target names: {invalid_targets}")
    if set(targets) != _REQUIRED_TARGETS:
        raise ManifestError(
            "v1 suites must declare exactly the symmetric gfx950/gfx1250 target set"
        )

    case_ids = _string_list(raw.get("cases"), "cases")
    if len(set(case_ids)) != len(case_ids):
        raise ManifestError("cases must not contain duplicate IDs")
    unknown_cases = sorted(set(case_ids) - set(CASE_BY_ID))
    if unknown_cases:
        raise ManifestError(f"unknown built-in benchmark cases: {unknown_cases}")

    measurement_raw = _mapping(raw.get("measurement"), "measurement")
    _reject_unknown(measurement_raw, _MEASUREMENT_FIELDS, "measurement")
    warmups = _nonnegative_int(
        measurement_raw.get("warmups"), "measurement.warmups"
    )
    samples = _nonnegative_int(
        measurement_raw.get("samples"), "measurement.samples"
    )
    if samples == 0:
        raise ManifestError("measurement.samples must be at least one")

    return Suite(
        path=manifest_path,
        digest=hashlib.sha256(raw_bytes).hexdigest(),
        name=name,
        description=description,
        targets=targets,
        cases=tuple(CASE_BY_ID[case_id] for case_id in case_ids),
        measurement=MeasurementPolicy(warmups=warmups, samples=samples),
        budget_seconds=_positive_number(
            raw.get("budget_seconds", 30 * 60), "budget_seconds"
        ),
        dependency_lock=DEPENDENCY_LOCK,
        dependencies=_load_dependencies(),
    )
