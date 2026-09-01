# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Strict comparison of synchronized-dispatch benchmark artifacts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .runtime import (
    CASE_RESULT_SCHEMA,
    DEPENDENCY_COHORT_PACKAGES,
    DEPENDENCY_COHORT_SCHEMA,
    ENVIRONMENT_PROVENANCE_POLICY,
    RUN_SCHEMA,
    write_json,
)
from .statistics import summarize

COMPARISON_SCHEMA = "rocjitsu.benchmark.comparison.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PACKAGE_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


class ComparisonError(ValueError):
    """Raised when artifacts cannot be compared without misleading results."""


def load_run(path: str | Path) -> dict[str, Any]:
    run_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read run artifact {run_path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != RUN_SCHEMA:
        raise ComparisonError(f"{run_path} is not a {RUN_SCHEMA} artifact")
    return value


def _at(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _config_digests(run: Mapping[str, Any]) -> dict[str, str]:
    targets = _at(run, "suite.targets")
    configs = _at(run, "provenance.configs")
    if (
        not isinstance(targets, Sequence)
        or isinstance(targets, (str, bytes))
        or not isinstance(configs, Mapping)
    ):
        raise ComparisonError("run has invalid target/config provenance")
    if set(configs) != set(targets):
        raise ComparisonError("provenance.configs targets do not match suite.targets")
    result: dict[str, str] = {}
    for target, record in configs.items():
        digest = record.get("sha256") if isinstance(record, Mapping) else None
        if not isinstance(target, str) or not _valid_digest(digest):
            raise ComparisonError(f"invalid config digest for target {target!r}")
        result[target] = digest
    return result


def _validate_dependency_provenance(run: Mapping[str, Any], label: str) -> None:
    dependencies = _at(run, "provenance.dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != {
        "lock_sha256",
        "verifier_sha256",
        "cohort",
    }:
        raise ComparisonError(f"{label} has invalid dependency provenance")
    if not _valid_digest(dependencies.get("lock_sha256")) or not _valid_digest(
        dependencies.get("verifier_sha256")
    ):
        raise ComparisonError(f"{label} has invalid dependency provenance digests")
    cohort = dependencies.get("cohort")
    if not isinstance(cohort, Mapping) or cohort.get("schema") != DEPENDENCY_COHORT_SCHEMA:
        raise ComparisonError(f"{label} has invalid dependency cohort")
    expected = cohort.get("expected")
    actual = cohort.get("actual")
    python = cohort.get("python")
    hipblaslt = cohort.get("hipblaslt")
    if (
        not isinstance(expected, Mapping)
        or set(expected) != DEPENDENCY_COHORT_PACKAGES
        or any(
            not isinstance(name, str)
            or not isinstance(version, str)
            or not version
            for name, version in expected.items()
        )
        or not isinstance(actual, Mapping)
        or actual != expected
        or not isinstance(python, Mapping)
        or not isinstance(python.get("actual"), str)
        or python.get("actual") != python.get("expected")
        or not isinstance(hipblaslt, Mapping)
        or not isinstance(hipblaslt.get("expected_package_version"), str)
        or _PACKAGE_VERSION.fullmatch(
            hipblaslt.get("expected_package_version", "")
        )
        is None
        or cohort.get("passed") is not True
        or cohort.get("errors") != []
    ):
        raise ComparisonError(f"{label} dependency cohort did not pass exactly")


def _compatibility_errors(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    if baseline.get("status") != "passed":
        errors.append("baseline run did not pass")
    if candidate.get("status") != "passed":
        errors.append("candidate run did not pass")
    stable_paths = (
        "mode",
        "suite.name",
        "suite.manifest_sha256",
        "suite.catalog_sha256",
        "suite.targets",
        "suite.cases",
        "execution",
        "provenance.runner.version",
        "provenance.runner.source_sha256",
        "provenance.runner.python",
        "provenance.catalog_sha256",
        "provenance.host.hostname",
        "provenance.host.system",
        "provenance.host.release",
        "provenance.host.machine",
        "provenance.host.cpu_model",
        "provenance.host.cpu_count",
        "provenance.dependencies",
        "provenance.build_type",
        "provenance.explicit_environment",
    )
    for path in stable_paths:
        before = _at(baseline, path)
        after = _at(candidate, path)
        if before != after:
            errors.append(f"provenance mismatch at {path}: {before!r} != {after!r}")
    try:
        before_configs = _config_digests(baseline)
        after_configs = _config_digests(candidate)
    except ComparisonError as exc:
        errors.append(str(exc))
    else:
        if before_configs != after_configs:
            errors.append("target config digests differ")
    return errors


def _result_key(result: Mapping[str, Any]) -> tuple[str, str]:
    case = result.get("case")
    case_id = case.get("id") if isinstance(case, Mapping) else None
    target = result.get("target")
    if not isinstance(case_id, str) or not isinstance(target, str):
        raise ComparisonError("case result is missing case.id or target")
    return case_id, target


def _group_results(
    run: Mapping[str, Any], field: str = "results"
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if field not in run:
        raise ComparisonError(f"run is missing required {field} entries")
    raw = run[field]
    if not isinstance(raw, list):
        raise ComparisonError(f"run {field} must be an array")
    groups: dict[tuple[str, str], Mapping[str, Any]] = {}
    for result in raw:
        if not isinstance(result, Mapping):
            raise ComparisonError(f"run contains a non-object {field} entry")
        if result.get("schema") != CASE_RESULT_SCHEMA:
            raise ComparisonError(f"{field} entry has an unsupported schema")
        key = _result_key(result)
        if key in groups:
            raise ComparisonError(f"run contains multiple {field} entries for {key}")
        if result.get("status") != "passed":
            raise ComparisonError(f"{field} entry {key} did not pass")
        groups[key] = result
    return groups


def _expected_matrix(run: Mapping[str, Any]) -> set[tuple[str, str]]:
    cases = _at(run, "suite.cases")
    targets = _at(run, "suite.targets")
    if (
        not isinstance(cases, list)
        or not cases
        or not all(isinstance(case, str) for case in cases)
        or len(set(cases)) != len(cases)
        or not isinstance(targets, list)
        or not targets
        or not all(isinstance(target, str) for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ComparisonError("run has an invalid suite case/target matrix")
    return {(case, target) for case in cases for target in targets}


def _validate_run_shape(run: Mapping[str, Any], label: str) -> None:
    if run.get("schema") != RUN_SCHEMA:
        raise ComparisonError(f"{label} has unsupported run schema")
    _validate_dependency_provenance(run, label)
    measurement = _at(run, "execution.measurement")
    if not isinstance(measurement, Mapping):
        raise ComparisonError(f"{label} is missing execution.measurement")
    warmups = measurement.get("warmups")
    samples = measurement.get("samples")
    metric = measurement.get("metric")
    if (
        isinstance(warmups, bool)
        or not isinstance(warmups, int)
        or warmups < 0
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples <= 0
        or not isinstance(metric, str)
        or not metric
    ):
        raise ComparisonError(f"{label} has invalid execution.measurement")

    expected = _expected_matrix(run)
    results = _group_results(run)
    if set(results) != expected:
        raise ComparisonError(f"{label} result matrix does not match its suite")
    primary_phase = "timing" if run.get("mode") == "run" else "profile"
    for key, result in results.items():
        for field in ("metric", "warmups", "samples"):
            if _at(result, f"measurement.{field}") != measurement.get(field):
                raise ComparisonError(
                    f"{label} {key} measurement.{field} contradicts execution"
                )
        if result.get("phase") != primary_phase:
            raise ComparisonError(f"{label} {key} has invalid primary phase")
        _timings(result, key)
    throughput = _group_results(run, "throughput")
    mode = run.get("mode")
    if mode == "run" and set(throughput) != expected:
        raise ComparisonError(f"{label} throughput matrix does not match its suite")
    if mode == "profile" and throughput:
        raise ComparisonError(f"{label} profile unexpectedly contains throughput entries")
    if mode not in {"run", "profile"}:
        raise ComparisonError(f"{label} has unsupported mode {mode!r}")
    for key, result in throughput.items():
        if (
            _at(result, "measurement.metric") != measurement.get("metric")
            or _at(result, "measurement.warmups") != 0
            or _at(result, "measurement.samples") != 1
            or result.get("phase") != "throughput"
        ):
            raise ComparisonError(f"{label} {key} has invalid throughput policy")
        _timings(result, key)


def _timings(result: Mapping[str, Any], key: tuple[str, str]) -> list[float]:
    raw = _at(result, "measurement.values")
    expected = _at(result, "measurement.samples")
    if (
        not isinstance(raw, list)
        or isinstance(expected, bool)
        or not isinstance(expected, int)
        or len(raw) != expected
        or expected <= 0
    ):
        raise ComparisonError(f"{key} has an invalid timing vector")
    values: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ComparisonError(f"{key} contains a non-numeric timing")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ComparisonError(f"{key} contains a non-positive timing")
        values.append(numeric)
    return values


def _validated_result_provenance(
    result: Mapping[str, Any], key: tuple[str, str], label: str
) -> tuple[str, dict[str, str], str, dict[str, str]]:
    payload = _at(result, "provenance.payload_executable.sha256")
    config = _at(result, "provenance.config_sha256")
    environment = _at(result, "provenance.effective_environment")
    if not _valid_digest(payload):
        raise ComparisonError(f"{key} {label} result has invalid payload provenance")
    if not _valid_digest(config):
        raise ComparisonError(f"{key} {label} result has invalid config provenance")
    if (
        not isinstance(environment, Mapping)
        or environment.get("policy") != ENVIRONMENT_PROVENANCE_POLICY
        or not isinstance(environment.get("values"), Mapping)
    ):
        raise ComparisonError(f"{key} {label} result has invalid environment provenance")
    values = environment["values"]
    if any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in values.items()
    ):
        raise ComparisonError(f"{key} {label} result has invalid environment values")
    provider = _at(result, "case.provider")
    raw_dependencies = _at(result, "provenance.dynamic_dependencies")
    dependencies: dict[str, str] = {}
    if provider == "hipblaslt":
        if not isinstance(raw_dependencies, Mapping) or set(raw_dependencies) != {
            "hipblaslt",
            "tensile_library",
        }:
            raise ComparisonError(
                f"{key} {label} result has invalid hipBLASLt dependencies"
            )
        for name, record in raw_dependencies.items():
            digest = record.get("sha256") if isinstance(record, Mapping) else None
            if not _valid_digest(digest):
                raise ComparisonError(
                    f"{key} {label} result has invalid {name} digest"
                )
            dependencies[name] = digest
    elif raw_dependencies not in (None, {}):
        raise ComparisonError(f"{key} {label} result has unexpected dependencies")
    return payload, dict(values), config, dependencies


def _ensure_compatible(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], key: tuple[str, str]
) -> None:
    before_provenance = _validated_result_provenance(baseline, key, "baseline")
    after_provenance = _validated_result_provenance(candidate, key, "candidate")
    if before_provenance != after_provenance:
        raise ComparisonError(f"{key} result provenance differs")
    for path in (
        "case.provider",
        "case.parameters",
        "case.sources",
        "case.source_sha256",
        "case.expected_dispatch_count",
        "measurement.metric",
        "measurement.warmups",
        "measurement.samples",
    ):
        if _at(baseline, path) != _at(candidate, path):
            raise ComparisonError(f"{key} metadata mismatch at {path}")


def _check_throughput(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    before = _group_results(baseline, "throughput")
    after = _group_results(candidate, "throughput")
    if set(before) != set(after):
        raise ComparisonError("throughput case/target matrix mismatch")
    for key in before:
        _ensure_compatible(before[key], after[key], key)
        before_summary = before[key].get("throughput_summary")
        after_summary = after[key].get("throughput_summary")
        if not isinstance(before_summary, Mapping) or not isinstance(after_summary, Mapping):
            raise ComparisonError(f"throughput entry {key} has no summary")
        for label, summary in (
            ("baseline", before_summary),
            ("candidate", after_summary),
        ):
            dispatches = summary.get("dispatches")
            wave_instructions = summary.get("wave_instructions")
            dispatch_seconds = summary.get("dispatch_seconds_sum")
            if (
                isinstance(dispatches, bool)
                or not isinstance(dispatches, int)
                or dispatches <= 0
                or isinstance(wave_instructions, bool)
                or not isinstance(wave_instructions, int)
                or wave_instructions < 0
                or isinstance(dispatch_seconds, bool)
                or not isinstance(dispatch_seconds, (int, float))
                or not math.isfinite(float(dispatch_seconds))
                or dispatch_seconds < 0
            ):
                raise ComparisonError(
                    f"{key} {label} throughput summary is incomplete"
                )
        for field in ("dispatches", "wave_instructions"):
            if before_summary.get(field) != after_summary.get(field):
                raise ComparisonError(f"{key} throughput {field} differs")


def compare_runs(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    baseline_path: str | None = None,
    candidate_path: str | None = None,
) -> dict[str, Any]:
    _validate_run_shape(baseline, "baseline")
    _validate_run_shape(candidate, "candidate")
    errors = _compatibility_errors(baseline, candidate)
    if errors:
        raise ComparisonError("; ".join(errors))
    baseline_results = _group_results(baseline)
    candidate_results = _group_results(candidate)
    if set(baseline_results) != set(candidate_results):
        raise ComparisonError("case/target result matrix mismatch")
    _check_throughput(baseline, candidate)

    comparisons: list[dict[str, Any]] = []
    for key in sorted(baseline_results):
        before = baseline_results[key]
        after = candidate_results[key]
        _ensure_compatible(before, after, key)
        before_values = _timings(before, key)
        after_values = _timings(after, key)
        if len(before_values) != len(after_values):
            raise ComparisonError(f"{key} timing counts differ")
        pairs: list[dict[str, Any]] = []
        ratios: list[float] = []
        deltas: list[float] = []
        percents: list[float] = []
        for ordinal, (baseline_value, candidate_value) in enumerate(
            zip(before_values, after_values, strict=True), 1
        ):
            delta = candidate_value - baseline_value
            ratio = candidate_value / baseline_value
            percent = (ratio - 1.0) * 100.0
            ratios.append(ratio)
            deltas.append(delta)
            percents.append(percent)
            pairs.append(
                {
                    "ordinal": ordinal,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "delta": delta,
                    "ratio": ratio,
                    "percent_change": percent,
                }
            )
        comparisons.append(
            {
                "case": key[0],
                "target": key[1],
                "metric": _at(before, "measurement.metric"),
                "pairs": pairs,
                "ratio_statistics": summarize(ratios),
                "delta_statistics": summarize(deltas),
                "percent_change_statistics": summarize(percents),
            }
        )
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "passed",
        "baseline": {
            "path": baseline_path,
            "rocjitsu_sha256": _at(baseline, "provenance.rocjitsu.sha256"),
        },
        "candidate": {
            "path": candidate_path,
            "rocjitsu_sha256": _at(candidate, "provenance.rocjitsu.sha256"),
        },
        "sample_count": _at(baseline, "execution.measurement.samples"),
        "comparisons": comparisons,
    }


def compare_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    baseline_resolved = Path(baseline_path).expanduser().resolve()
    candidate_resolved = Path(candidate_path).expanduser().resolve()
    result = compare_runs(
        load_run(baseline_resolved),
        load_run(candidate_resolved),
        baseline_path=str(baseline_resolved),
        candidate_path=str(candidate_resolved),
    )
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        if output_path.exists():
            raise ComparisonError(f"refusing to overwrite existing comparison: {output_path}")
        write_json(output_path, result)
    return result
