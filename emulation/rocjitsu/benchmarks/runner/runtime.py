# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution, validation, and artifact handling for Rocjitsu benchmarks."""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
import signal
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..catalog import (
    BENCHMARK_ROOT,
    DEPENDENCY_LOCK,
    ROCJITSU_ROOT,
    CaseSpec,
    catalog_sha256,
)
from . import __version__
from .manifest import Suite
from .providers import PreparedCommand, prepare_command, source_paths
from .statistics import summarize

CASE_RESULT_SCHEMA = "rocjitsu.benchmark.case-result.v1"
RUN_SCHEMA = "rocjitsu.benchmark.run.v1"
WORKLOAD_SCHEMA = "rocjitsu.benchmark.workload.v1"
DEPENDENCY_COHORT_SCHEMA = "rocjitsu.benchmark.environment.v1"
DEPENDENCY_COHORT_PACKAGES = frozenset(
    {"rocm_sdk", "torch", "triton", "hipblaslt"}
)
ENVIRONMENT_PROVENANCE_POLICY = "rocjitsu.benchmark.environment-allowlist.v1"
DEFAULT_COLLECTORS = ("time", "perf")
_PERF_EVENTS = (
    "cycles:u",
    "instructions:u",
    "task-clock",
    "context-switches",
    "cpu-migrations",
    "minor-faults",
    "major-faults",
)
_PERFORMANCE_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "AMD_SERIALIZE_KERNEL",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HIPBLASLT_TENSILE_LIBPATH",
        "HIP_LAUNCH_BLOCKING",
        "HIP_VISIBLE_DEVICES",
        "HSA_ENABLE_SDMA",
        "HSA_NO_SCRATCH_RECLAIM",
        "HSA_OVERRIDE_GFX_VERSION",
        "HSA_SCRATCH_SINGLE_LIMIT",
        "HSA_XNACK",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "OMP_DYNAMIC",
        "OMP_NUM_THREADS",
        "OMP_PLACES",
        "OMP_PROC_BIND",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "ROCM_PATH",
        "ROCR_VISIBLE_DEVICES",
        "TRITON_CACHE_DIR",
    }
)
_RUN_RELATIVE_ENVIRONMENT = frozenset({"TRITON_CACHE_DIR"})


class RunnerError(RuntimeError):
    """Raised for a runner-level error shown without a traceback."""


@dataclasses.dataclass(frozen=True)
class RunOptions:
    output: Path
    build_dir: Path | None = None
    python: str = sys.executable
    rocjitsu: str | None = None
    configs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    environment: Mapping[str, str] = dataclasses.field(default_factory=dict)
    targets: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    collectors: tuple[str, ...] | None = None
    warmups: int | None = None
    samples: int | None = None
    fail_fast: bool = False


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write deterministic, standards-compliant JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        stream.write("\n")
    temporary.replace(path)


def _sanitize_json_value(value: Any, path: str = "$") -> tuple[Any, list[str]]:
    if isinstance(value, float) and not math.isfinite(value):
        return None, [path]
    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        invalid: list[str] = []
        for key, item in value.items():
            child, child_invalid = _sanitize_json_value(item, f"{path}.{key}")
            result[key] = child
            invalid.extend(child_invalid)
        return result, invalid
    if isinstance(value, list):
        result_list: list[Any] = []
        invalid = []
        for index, item in enumerate(value):
            child, child_invalid = _sanitize_json_value(item, f"{path}[{index}]")
            result_list.append(child)
            invalid.extend(child_invalid)
        return result_list, invalid
    return value, []


def file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


class _RunFileHashes:
    def __init__(self) -> None:
        self._cache: dict[Path, tuple[tuple[int, int, int, int, int], str]] = {}

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int, int, int] | None:
        try:
            value = path.stat()
        except OSError:
            return None
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def sha256(self, path: Path | None) -> str | None:
        if path is None:
            return None
        signature = self._signature(path)
        if signature is None:
            return None
        cached = self._cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        digest = file_sha256(path)
        if digest is not None and self._signature(path) == signature:
            self._cache[path] = (signature, digest)
        return digest


def _combined_sha256(paths: Sequence[Path], hashes: _RunFileHashes) -> str | None:
    digest = hashlib.sha256()
    for path in paths:
        value = hashes.sha256(path)
        if value is None:
            return None
        relative = path.relative_to(BENCHMARK_ROOT)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _directory_sha256(root: Path, hashes: _RunFileHashes) -> str | None:
    try:
        paths = sorted(path for path in root.rglob("*") if path.is_file())
    except OSError:
        return None
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        value = hashes.sha256(path)
        if value is None:
            return None
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _elapsed_seconds(value: str) -> float:
    parts = value.strip().split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid elapsed time {value!r}") from exc
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        return numbers[0] * 60.0 + numbers[1]
    if len(numbers) == 3:
        return numbers[0] * 3600.0 + numbers[1] * 60.0 + numbers[2]
    raise ValueError(f"invalid elapsed time {value!r}")


def parse_time_verbose(text: str) -> dict[str, float | int]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.strip().partition(": ")
        if separator:
            fields[key] = value.strip()
    names: dict[str, tuple[str, type[float] | type[int]]] = {
        "User time (seconds)": ("user_seconds", float),
        "System time (seconds)": ("system_seconds", float),
        "Maximum resident set size (kbytes)": ("max_rss_kbytes", int),
        "Major (requiring I/O) page faults": ("major_faults", int),
        "Minor (reclaiming a frame) page faults": ("minor_faults", int),
    }
    result: dict[str, float | int] = {}
    missing: list[str] = []
    for source, (destination, conversion) in names.items():
        if source not in fields:
            missing.append(source)
            continue
        try:
            result[destination] = conversion(fields[source])
        except ValueError as exc:
            raise ValueError(
                f"invalid GNU time field {source!r}: {fields[source]!r}"
            ) from exc
    elapsed_key = "Elapsed (wall clock) time (h:mm:ss or m:ss)"
    if elapsed_key not in fields:
        missing.append(elapsed_key)
    else:
        result["elapsed_seconds"] = _elapsed_seconds(fields[elapsed_key])
    if missing:
        raise ValueError(f"GNU time output is missing fields: {missing}")
    return result


def parse_perf_stat(text: str) -> dict[str, float | int]:
    event_names = {
        "cycles:u": "cycles",
        "instructions:u": "instructions",
        "task-clock": "task_clock_milliseconds",
        "context-switches": "context_switches",
        "cpu-migrations": "cpu_migrations",
        "minor-faults": "minor_faults",
        "major-faults": "major_faults",
    }
    result: dict[str, float | int] = {}
    for row in csv.reader(io.StringIO(text), delimiter=","):
        if len(row) < 3:
            continue
        raw_value, _unit, raw_event = (item.strip() for item in row[:3])
        event = raw_event.split("#", 1)[0].strip()
        destination = event_names.get(event)
        if destination is None:
            continue
        if not raw_value or raw_value.startswith("<"):
            raise ValueError(f"perf event {event} was not counted: {raw_value!r}")
        try:
            numeric = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"invalid perf value for {event}: {raw_value!r}") from exc
        result[destination] = (
            numeric if destination == "task_clock_milliseconds" else int(round(numeric))
        )
    missing = sorted(set(event_names.values()) - set(result))
    if missing:
        raise ValueError(f"perf stat output is missing metrics: {missing}")
    return result


def select_suite(
    suite: Suite, options: RunOptions
) -> tuple[tuple[str, ...], tuple[CaseSpec, ...]]:
    targets = options.targets or suite.targets
    unknown_targets = sorted(set(targets) - set(suite.targets))
    if unknown_targets:
        raise RunnerError(f"unknown targets requested: {unknown_targets}")
    if len(set(targets)) != len(targets):
        raise RunnerError("target selection contains duplicates")

    if options.case_ids:
        by_id = {case.id: case for case in suite.cases}
        unknown_cases = sorted(set(options.case_ids) - set(by_id))
        if unknown_cases:
            raise RunnerError(f"unknown cases requested: {unknown_cases}")
        cases = tuple(by_id[case_id] for case_id in options.case_ids)
    else:
        cases = suite.cases
    if len(set(options.case_ids)) != len(options.case_ids):
        raise RunnerError("case selection contains duplicates")
    return tuple(targets), cases


def _resolve_program(
    program: str, cwd: Path, environment: Mapping[str, str]
) -> Path | None:
    if os.sep in program:
        path = Path(program).expanduser()
        if not path.is_absolute():
            path = cwd / path
        return path.absolute()
    found = shutil.which(program, path=environment.get("PATH", os.environ.get("PATH", "")))
    return Path(found).absolute() if found else None


def _cmake_build_type(build_dir: Path) -> str | None:
    try:
        lines = (build_dir.expanduser().resolve() / "CMakeCache.txt").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return None
    prefix = "CMAKE_BUILD_TYPE:STRING="
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
    except ValueError:
        return False
    return True


def _is_executable_file(path: Path | None) -> bool:
    return path is not None and path.is_file() and os.access(path, os.X_OK)


def _validate_timing_config(path: Path) -> tuple[bool, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"{path}: {exc}"
    if not isinstance(value, Mapping):
        return False, f"{path}: top-level value is not an object"
    if value.get("plugins") not in (None, {}):
        return False, f"{path}: official timing config must not enable plugins"
    return True, str(path)


def _effective_counts(suite: Suite, options: RunOptions) -> tuple[int, int]:
    return (
        options.warmups if options.warmups is not None else suite.measurement.warmups,
        options.samples if options.samples is not None else suite.measurement.samples,
    )


def _expected_hipblaslt_version(suite: Suite) -> int | None:
    value = suite.dependencies.get("hipblaslt")
    version = value.get("package_version") if isinstance(value, Mapping) else None
    if not isinstance(version, str):
        return None
    parts = version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        return None
    major, minor, patch = (int(part) for part in parts)
    return major * 100_000 + minor * 100 + patch


def _hipblaslt_target_logic_path(
    library: Path, environment: Mapping[str, str], target: str
) -> Path:
    override = environment.get("HIPBLASLT_TENSILE_LIBPATH")
    if override:
        return Path(override).expanduser().resolve()
    return library.parent / "hipblaslt" / "library" / target


def _hipblaslt_logic_directory_error(logic: Path, target: str) -> str | None:
    try:
        files = tuple(path for path in logic.iterdir() if path.is_file())
    except OSError as exc:
        return f"cannot inspect hipBLASLt device-library directory {logic}: {exc}"
    code_objects = [
        path for path in files if target in path.name and path.name.endswith(".co")
    ]
    lazy_mappings = [
        path
        for path in files
        if path.name
        in {
            f"TensileLibrary_lazy_{target}.dat",
            f"TensileLibrary_lazy_{target}.dat.zlib",
        }
    ]
    if not code_objects or not lazy_mappings:
        return (
            f"hipBLASLt device-library directory {logic} must directly contain "
            f"{target}-named .co files and its TensileLibrary_lazy mapping"
        )
    return None


def _hipblaslt_dependency_provenance(
    payload: Mapping[str, Any],
    environment: Mapping[str, str],
    hashes: _RunFileHashes,
    target: str,
) -> tuple[dict[str, Any] | None, str | None]:
    provenance = payload.get("provenance")
    raw_library = (
        provenance.get("hipblaslt_library_path")
        if isinstance(provenance, Mapping)
        else None
    )
    if not isinstance(raw_library, str) or not raw_library:
        return None, "hipBLASLt workload did not report its loaded library path"
    library = Path(raw_library).expanduser().resolve()
    library_digest = hashes.sha256(library)
    if library_digest is None:
        return None, f"could not hash loaded hipBLASLt library {library}"
    logic = _hipblaslt_target_logic_path(library, environment, target)
    logic_error = _hipblaslt_logic_directory_error(logic, target)
    if logic_error is not None:
        return None, logic_error
    logic_digest = _directory_sha256(logic, hashes)
    if logic_digest is None:
        return None, f"could not hash hipBLASLt Tensile library directory {logic}"
    return (
        {
            "hipblaslt": {"path": str(library), "sha256": library_digest},
            "tensile_library": {"path": str(logic), "sha256": logic_digest},
        },
        None,
    )


def _last_json_object(text: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(value, Mapping):
            return value
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _normalize_dependency_cohort(
    payload: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if payload is None or payload.get("schema") != DEPENDENCY_COHORT_SCHEMA:
        return None, "dependency verifier did not emit the expected report schema"
    expected = payload.get("expected")
    actual = payload.get("actual")
    python = payload.get("python")
    hipblaslt = payload.get("hipblaslt")
    errors = payload.get("errors")
    passed = payload.get("passed")
    if (
        not isinstance(expected, Mapping)
        or not expected
        or any(
            not isinstance(name, str) or not isinstance(version, str)
            for name, version in expected.items()
        )
        or not isinstance(actual, Mapping)
        or set(actual) != set(expected)
        or any(
            not isinstance(name, str)
            or (version is not None and not isinstance(version, str))
            for name, version in actual.items()
        )
        or not isinstance(python, Mapping)
        or not isinstance(python.get("actual"), str)
        or not isinstance(python.get("expected"), str)
        or not isinstance(hipblaslt, Mapping)
        or not isinstance(hipblaslt.get("expected_package_version"), str)
        or not isinstance(passed, bool)
        or not isinstance(errors, list)
        or any(not isinstance(error, str) for error in errors)
    ):
        return None, "dependency verifier emitted a malformed report"
    observed_success = (
        actual == expected
        and python["actual"] == python["expected"]
        and not errors
    )
    if passed is not observed_success:
        return None, "dependency verifier emitted an internally inconsistent report"
    return (
        {
            "schema": DEPENDENCY_COHORT_SCHEMA,
            "expected": dict(sorted(expected.items())),
            "actual": dict(sorted(actual.items())),
            "python": {
                "actual": python["actual"],
                "expected": python["expected"],
            },
            "hipblaslt": {
                "expected_package_version": hipblaslt["expected_package_version"]
            },
            "passed": passed,
            "errors": list(errors),
        },
        None,
    )


def _dependency_cohort_lock_error(
    cohort: Mapping[str, Any], suite: Suite
) -> str | None:
    dependencies = suite.dependencies
    try:
        locked_expected = {
            "rocm_sdk": dependencies["rocm_sdk"]["version"],
            "torch": dependencies["pytorch"]["version"],
            "triton": dependencies["triton"]["version"],
            "hipblaslt": dependencies["hipblaslt"]["distribution_version"],
        }
        locked_python = dependencies["rocm_sdk"]["python"]
        locked_hipblaslt = dependencies["hipblaslt"]["package_version"]
    except (KeyError, TypeError):
        return "dependency lock is missing cohort identity fields"
    if (
        set(locked_expected) != DEPENDENCY_COHORT_PACKAGES
        or any(not isinstance(value, str) for value in locked_expected.values())
        or not isinstance(locked_python, str)
        or not isinstance(locked_hipblaslt, str)
    ):
        return "dependency lock contains invalid cohort identity fields"
    if cohort.get("expected") != locked_expected:
        return "dependency verifier expectations do not match the dependency lock"
    if (
        not isinstance(cohort.get("python"), Mapping)
        or cohort["python"].get("expected") != locked_python
    ):
        return "dependency verifier Python expectation does not match the lock"
    if (
        not isinstance(cohort.get("hipblaslt"), Mapping)
        or cohort["hipblaslt"].get("expected_package_version")
        != locked_hipblaslt
    ):
        return "dependency verifier hipBLASLt expectation does not match the lock"
    return None


def _probe_hipblaslt(
    suite: Suite,
    case: CaseSpec,
    target: str,
    options: RunOptions,
) -> tuple[bool, str]:
    expected_version = _expected_hipblaslt_version(suite)
    if expected_version is None:
        return False, "dependency lock has invalid hipBLASLt identity fields"
    try:
        prepared = prepare_command(
            suite,
            case,
            target,
            options,
            options.output / ".preflight" / case.id / target,
            warmups=0,
            samples=1,
            phase="preflight",
        )
    except ValueError as exc:
        return False, str(exc)
    command = (
        *prepared.argv[:4],
        prepared.payload_program,
        "--probe",
        "--expected-target",
        target,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=prepared.cwd,
            env={**os.environ, **prepared.environment},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"cannot probe hipBLASLt workload: {exc}"
    payload = _last_json_object(completed.stdout)
    version = payload.get("hipblaslt_version") if payload else None
    raw_library = payload.get("hipblaslt_library_path") if payload else None
    library = (
        Path(raw_library).expanduser().resolve()
        if isinstance(raw_library, str) and raw_library
        else None
    )
    logic = (
        _hipblaslt_target_logic_path(
            library, {**os.environ, **prepared.environment}, target
        )
        if library is not None
        else None
    )
    logic_error = (
        _hipblaslt_logic_directory_error(logic, target)
        if logic is not None
        else "hipBLASLt workload did not report a library path"
    )
    passed = (
        completed.returncode == 0
        and version == expected_version
        and library is not None
        and library.is_file()
        and logic is not None
        and logic.is_dir()
        and logic_error is None
    )
    detail = {
        "command": list(command),
        "exit_status": completed.returncode,
        "expected_version": expected_version,
        "reported_version": version,
        "library": str(library) if library else None,
        "tensile_library": str(logic) if logic else None,
        "tensile_library_error": logic_error,
        "output": (completed.stdout + completed.stderr).strip()[-2000:],
    }
    return passed, json.dumps(detail, sort_keys=True)


def preflight(
    suite: Suite, options: RunOptions, *, profile: bool = False
) -> dict[str, Any]:
    targets, cases = select_suite(suite, options)
    checks: list[dict[str, Any]] = []
    dependency_cohort: dict[str, Any] | None = None

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    check(
        "python-version",
        sys.version_info >= (3, 11),
        f"running Python {platform.python_version()}; Python 3.11+ is required",
    )
    python = _resolve_program(options.python, Path.cwd(), options.environment)
    check(
        "python-executable",
        python is not None and python.is_file(),
        str(python) if python else f"not found: {options.python}",
    )
    if python is not None:
        verifier = BENCHMARK_ROOT / "verify_environment.py"
        try:
            probe = subprocess.run(
                [str(python), str(verifier), "--lock", str(DEPENDENCY_LOCK)],
                capture_output=True,
                text=True,
                env={**os.environ, **options.environment},
                timeout=30,
                check=False,
            )
            dependency_cohort, cohort_error = _normalize_dependency_cohort(
                _last_json_object(probe.stdout)
            )
            lock_error = (
                _dependency_cohort_lock_error(dependency_cohort, suite)
                if dependency_cohort is not None
                else None
            )
            cohort_passed = (
                probe.returncode == 0
                and cohort_error is None
                and lock_error is None
                and dependency_cohort is not None
                and dependency_cohort["passed"] is True
            )
            if dependency_cohort is not None:
                report_json = json.dumps(dependency_cohort, sort_keys=True)
                detail = (
                    f"{lock_error}; report={report_json}"
                    if lock_error is not None
                    else report_json
                )
            else:
                detail = (
                    cohort_error
                    or probe.stdout.strip()
                    or probe.stderr.strip()
                    or f"exit status {probe.returncode}"
                )
            check("python-dependency-cohort", cohort_passed, detail)
        except (OSError, subprocess.SubprocessError) as exc:
            check("python-dependency-cohort", False, str(exc))

    expected_build_type = "RelWithDebInfo" if profile else "Release"
    build_type = _cmake_build_type(options.build_dir) if options.build_dir else None
    check(
        "build-type",
        build_type == expected_build_type,
        (
            f"CMAKE_BUILD_TYPE={build_type}"
            if build_type is not None
            else f"--build-dir must identify a {expected_build_type} build"
        ),
    )
    if profile:
        build_dir = options.build_dir
        selected_launcher = (
            Path(options.rocjitsu)
            if options.rocjitsu
            else build_dir / "tools/rocjitsu/rocjitsu"
            if build_dir
            else None
        )
        launcher_matches_build = (
            build_dir is not None
            and selected_launcher is not None
            and _path_is_within(selected_launcher, build_dir)
        )
        check(
            "profile-launcher-build",
            launcher_matches_build,
            (
                f"{selected_launcher.expanduser().resolve()} is inside "
                f"{build_dir.expanduser().resolve()}"
                if launcher_matches_build
                else "the profiled Rocjitsu executable must be inside --build-dir"
            ),
        )

    commands: list[dict[str, Any]] = []
    collector_names = set(
        DEFAULT_COLLECTORS if options.collectors is None else options.collectors
    )
    if profile:
        collector_names = {"time", "perf-record"}
    for case in cases:
        paths = source_paths(case)
        missing_sources = [str(path) for path in paths if not path.is_file()]
        check(
            f"sources:{case.id}",
            not missing_sources,
            ", ".join(str(path) for path in paths)
            if not missing_sources
            else f"missing: {missing_sources}",
        )
        for target in targets:
            warmups, samples = _effective_counts(suite, options)
            if profile and case.provider == "triton":
                command_specs = (("prime", 0, 1), ("profile", 0, 1))
            elif profile:
                command_specs = (("profile", 0, 1),)
            else:
                command_specs = (("timing", warmups, samples),)
            try:
                prepared_commands = tuple(
                    prepare_command(
                        suite,
                        case,
                        target,
                        options,
                        options.output / "cases" / case.id / target / phase,
                        warmups=phase_warmups,
                        samples=phase_samples,
                        phase=phase,
                    )
                    for phase, phase_warmups, phase_samples in command_specs
                )
            except (KeyError, ValueError) as exc:
                check(f"command:{case.id}:{target}", False, str(exc))
                continue
            prepared = prepared_commands[0]
            environment = {**os.environ, **prepared.environment}
            launcher = _resolve_program(prepared.argv[0], prepared.cwd, environment)
            payload = _resolve_program(
                prepared.payload_program, prepared.cwd, environment
            )
            check(
                f"launcher:{case.id}:{target}",
                _is_executable_file(launcher),
                str(launcher) if launcher else f"not found: {prepared.argv[0]}",
            )
            check(
                f"payload:{case.id}:{target}",
                _is_executable_file(payload),
                str(payload) if payload else f"not found: {prepared.payload_program}",
            )
            valid_config, config_detail = _validate_timing_config(prepared.config)
            check(f"config:{target}", valid_config, config_detail)
            for phase_spec, phase_command in zip(
                command_specs, prepared_commands, strict=True
            ):
                commands.append(
                    {
                        "case": case.id,
                        "target": target,
                        "phase": phase_spec[0],
                        "argv": list(phase_command.argv),
                        "cwd": str(phase_command.cwd),
                    }
                )

    if "time" in collector_names:
        check("collector:time", Path("/usr/bin/time").is_file(), "/usr/bin/time")
    if "perf" in collector_names or "perf-record" in collector_names:
        perf = shutil.which("perf")
        check("collector:perf", perf is not None, perf or "perf was not found on PATH")
        if perf:
            try:
                probe = subprocess.run(
                    [perf, "stat", "-e", "cycles:u,instructions:u", "--", "true"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                    check=False,
                )
                check(
                    "collector:perf-permission",
                    probe.returncode == 0,
                    probe.stderr.strip() or f"exit status {probe.returncode}",
                )
            except (OSError, subprocess.SubprocessError) as exc:
                check("collector:perf-permission", False, str(exc))

    hipblaslt_cases = [case for case in cases if case.provider == "hipblaslt"]
    if hipblaslt_cases:
        for target in targets:
            passed, detail = _probe_hipblaslt(
                suite, hipblaslt_cases[0], target, options
            )
            check(f"hipblaslt-package:{target}", passed, detail)

    report_warmups, report_samples = (
        (0, 1) if profile else _effective_counts(suite, options)
    )
    return {
        "passed": all(item["passed"] for item in checks),
        "suite": suite.name,
        "manifest": str(suite.path),
        "targets": list(targets),
        "cases": [case.id for case in cases],
        "measurement": {
            "metric": suite.measurement.metric,
            "warmups": report_warmups,
            "samples": report_samples,
        },
        "dependency_cohort": dependency_cohort,
        "checks": checks,
        "commands": commands,
    }


def _effective_environment_provenance(
    environment: Mapping[str, str],
    prepared_environment: Mapping[str, str],
    run_dir: Path,
) -> dict[str, Any]:
    names = set(_PERFORMANCE_ENVIRONMENT_ALLOWLIST)
    names.update(
        name for name in prepared_environment if not name.startswith("RJ_BENCHMARK_")
    )
    root = str(run_dir)
    values = {
        name: (
            "{run_dir}" + environment[name][len(root) :]
            if name in _RUN_RELATIVE_ENVIRONMENT
            and (environment[name] == root or environment[name].startswith(root + os.sep))
            else environment[name]
        )
        for name in sorted(names)
        if name in environment
    }
    return {"policy": ENVIRONMENT_PROVENANCE_POLICY, "values": values}


def _read_throughput(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, str(exc)
    objects: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, f"invalid throughput JSONL at line {line_number}: {exc}"
        if not isinstance(value, Mapping):
            return None, f"throughput JSONL line {line_number} is not an object"
        sanitized, invalid = _sanitize_json_value(value)
        if invalid:
            return None, "throughput JSONL contains non-finite values at " + ", ".join(invalid)
        objects.append(sanitized)
    summaries = [
        value
        for value in objects
        if value.get("schema") == "rocjitsu.throughput.v2"
        and value.get("record") == "summary"
    ]
    if len(summaries) != 1:
        return None, f"throughput JSONL must contain exactly one summary, found {len(summaries)}"
    if any(
        value.get("schema") != "rocjitsu.throughput.v2"
        or value.get("record") not in {"dispatch", "summary"}
        for value in objects
    ):
        return None, "throughput JSONL contains an unsupported record"
    summary = summaries[0]
    dispatches = summary.get("dispatches")
    if (
        isinstance(dispatches, bool)
        or not isinstance(dispatches, int)
        or dispatches <= 0
    ):
        return None, "throughput summary dispatches must be a positive integer"
    dispatch_records = sum(value.get("record") == "dispatch" for value in objects)
    if dispatch_records != dispatches:
        return None, (
            f"throughput summary reports {dispatches} dispatches, "
            f"but the JSONL contains {dispatch_records} dispatch records"
        )
    wave_instructions = summary.get("wave_instructions")
    if (
        isinstance(wave_instructions, bool)
        or not isinstance(wave_instructions, int)
        or wave_instructions < 0
    ):
        return None, (
            "throughput summary wave_instructions must be a non-negative integer"
        )
    dispatch_seconds = summary.get("dispatch_seconds_sum")
    if (
        isinstance(dispatch_seconds, bool)
        or not isinstance(dispatch_seconds, (int, float))
        or not math.isfinite(float(dispatch_seconds))
        or dispatch_seconds < 0
    ):
        return None, (
            "throughput summary dispatch_seconds_sum must be a finite, "
            "non-negative number"
        )
    return summary, None


def _write_throughput_config(base: Path, result_dir: Path) -> Path:
    try:
        value = json.loads(base.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot derive throughput config from {base}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"base config {base} must contain a JSON object")
    value["plugins"] = {"throughput": {}}
    value["sinks"] = {"types": ["file"], "dir": str(result_dir)}
    output = result_dir / "throughput-config.json"
    write_json(output, value)
    return output


def _application_payload(
    prepared: PreparedCommand, stdout_path: Path
) -> tuple[Mapping[str, Any] | None, str | None]:
    if prepared.output.is_file():
        try:
            value = json.loads(prepared.output.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, f"could not parse workload result: {exc}"
        if not isinstance(value, Mapping):
            return None, "workload result must be a JSON object"
        return value, None
    try:
        value = _last_json_object(stdout_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"could not parse workload stdout: {exc}"
    return (value, None) if value is not None else (None, "workload did not emit JSON")


def _target_matches(reported: str, expected: str) -> bool:
    return reported == expected or reported.startswith(expected + ":")


def _workload_contract(
    payload: Mapping[str, Any] | None,
    suite: Suite,
    case: CaseSpec,
    target: str,
    warmups: int,
    samples: int,
) -> tuple[list[int], str | None]:
    if payload is None:
        return [], "workload did not emit a JSON result object"
    if payload.get("schema") != WORKLOAD_SCHEMA:
        return [], f"workload JSON schema must be {WORKLOAD_SCHEMA}"
    if payload.get("case_id") != case.id:
        return [], f"workload JSON case_id does not match {case.id}"
    if payload.get("workload") != dict(case.parameters):
        return [], "workload JSON parameters do not match the built-in catalog"
    if payload.get("warmups") != warmups or payload.get("samples") != samples:
        return [], "workload JSON sampling counts do not match the requested policy"

    target_value = payload.get("target")
    if not isinstance(target_value, Mapping):
        return [], "workload JSON target must be an object"
    if target_value.get("expected") != target:
        return [], f"workload JSON expected target does not match {target}"
    for name in ("reported", "compiled"):
        value = target_value.get(name)
        if value is not None and (
            not isinstance(value, str) or not _target_matches(value, target)
        ):
            return [], f"workload JSON {name} target does not match {target}"
    reported = target_value.get("reported")
    if not isinstance(reported, str):
        return [], "workload JSON must report the runtime target"

    raw_timings = payload.get("synchronized_dispatch_ns")
    if not isinstance(raw_timings, list) or len(raw_timings) != samples:
        count = len(raw_timings) if isinstance(raw_timings, list) else None
        return [], f"workload JSON must contain exactly {samples} timings, found {count}"
    timings: list[int] = []
    for index, value in enumerate(raw_timings):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return [], f"workload timing {index} must be a positive integer"
        timings.append(value)

    if case.provider == "hipblaslt":
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            return [], "hipBLASLt workload must report provenance"
        version = provenance.get("hipblaslt_version")
        expected_version = _expected_hipblaslt_version(suite)
        if version != expected_version:
            return [], (
                f"hipBLASLt version {version!r} does not match {expected_version!r}"
            )
        dependency = suite.dependencies.get("hipblaslt")
        for field in ("adapter", "algorithm_policy"):
            expected_value = (
                dependency.get(field) if isinstance(dependency, Mapping) else None
            )
            if not isinstance(expected_value, str) or not expected_value:
                return [], f"dependency lock does not define hipblaslt.{field}"
            if provenance.get(field) != expected_value:
                return [], (
                    f"hipBLASLt {field} {provenance.get(field)!r} does not "
                    f"match {expected_value!r}"
                )
    return timings, None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _execute_case(
    suite: Suite,
    case: CaseSpec,
    target: str,
    options: RunOptions,
    result_dir: Path,
    hashes: _RunFileHashes,
    *,
    phase: str,
    warmups: int,
    samples: int,
    collectors: tuple[str, ...],
    profile: bool = False,
    throughput: bool = False,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    config_override = None
    base_config = None
    if throughput:
        base_config = prepare_command(
            suite,
            case,
            target,
            options,
            result_dir,
            warmups=warmups,
            samples=samples,
            phase=phase,
        ).config
        config_override = _write_throughput_config(base_config, result_dir)
    prepared = prepare_command(
        suite,
        case,
        target,
        options,
        result_dir,
        warmups=warmups,
        samples=samples,
        phase=phase,
        config_override=config_override,
    )

    stdout_path = result_dir / "stdout.txt"
    stderr_path = result_dir / "stderr.txt"
    time_path = result_dir / "time.txt"
    perf_stat_path = result_dir / "perf-stat.csv"
    perf_data_path = result_dir / "perf.data"
    throughput_path = result_dir / "throughput.log"
    command = list(prepared.argv)
    if "perf" in collectors:
        command = [
            "perf",
            "stat",
            "-x",
            ",",
            "-o",
            str(perf_stat_path),
            "-e",
            ",".join(_PERF_EVENTS),
            "--",
            *command,
        ]
    else:
        perf_stat_path.write_text("collector disabled\n", encoding="utf-8")
    if profile:
        command = [
            "perf",
            "record",
            "-g",
            "--call-graph",
            "dwarf",
            "-o",
            str(perf_data_path),
            "--",
            *command,
        ]
    if "time" in collectors:
        command = ["/usr/bin/time", "-v", "-o", str(time_path), *command]
    else:
        time_path.write_text("collector disabled\n", encoding="utf-8")

    environment = {**os.environ, **prepared.environment}
    effective_environment = _effective_environment_provenance(
        environment, prepared.environment, options.output
    )
    command_record = {
        "argv": list(prepared.argv),
        "wrapped_argv": command,
        "cwd": str(prepared.cwd),
        "environment": dict(sorted(prepared.environment.items())),
        "effective_environment": effective_environment,
    }
    write_json(result_dir / "command.json", command_record)

    started_at = utc_now()
    started = time.perf_counter_ns()
    return_code: int | None = None
    failure: str | None = None
    timed_out = False
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=prepared.cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=case.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                failure = f"timed out after {case.timeout_seconds:g} seconds"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    return_code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return_code = process.wait()
    except OSError as exc:
        failure = f"could not start command: {exc}"
        stdout_path.touch(exist_ok=True)
        stderr_path.touch(exist_ok=True)
    process_wall_ns = time.perf_counter_ns() - started

    payload, payload_error = _application_payload(prepared, stdout_path)
    if payload_error and failure is None:
        failure = payload_error
    if payload is not None:
        payload, invalid = _sanitize_json_value(payload)
        if invalid and failure is None:
            failure = "workload JSON contains non-finite values at " + ", ".join(invalid)
    timings, contract_error = _workload_contract(
        payload, suite, case, target, warmups, samples
    )
    if contract_error and failure is None:
        failure = contract_error
    dynamic_dependencies = None
    if case.provider == "hipblaslt" and payload is not None:
        dynamic_dependencies, dependency_error = _hipblaslt_dependency_provenance(
            payload, environment, hashes, target
        )
        if dependency_error and failure is None:
            failure = dependency_error
    if return_code not in (0, None) and failure is None:
        failure = f"command exited with status {return_code}"

    throughput_summary = None
    if throughput:
        throughput_summary, throughput_error = _read_throughput(throughput_path)
        if throughput_error and failure is None:
            failure = throughput_error
        if throughput_summary is not None and case.expected_dispatch_count is not None:
            dispatches = throughput_summary.get("dispatches")
            if dispatches != case.expected_dispatch_count and failure is None:
                failure = (
                    f"throughput dispatch count {dispatches!r} does not match "
                    f"{case.expected_dispatch_count}"
                )

    diagnostics: dict[str, Any] = {
        "process_wall_time_ns": process_wall_ns,
        "wall_time_seconds": process_wall_ns / 1_000_000_000.0,
    }
    if "time" in collectors:
        try:
            diagnostics["time"] = parse_time_verbose(time_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            if failure is None:
                failure = f"could not normalize GNU time output: {exc}"
    if "perf" in collectors:
        try:
            diagnostics["perf"] = parse_perf_stat(perf_stat_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            if failure is None:
                failure = f"could not normalize perf stat output: {exc}"

    passed = return_code == 0 and not timed_out and failure is None
    merged_environment = {**os.environ, **prepared.environment}
    launcher = _resolve_program(prepared.argv[0], prepared.cwd, merged_environment)
    payload_program = _resolve_program(
        prepared.payload_program, prepared.cwd, merged_environment
    )
    paths = source_paths(case)
    result: dict[str, Any] = {
        "schema": CASE_RESULT_SCHEMA,
        "suite": suite.name,
        "case": {
            "id": case.id,
            "provider": case.provider,
            "parameters": dict(case.parameters),
            "sources": [str(path.relative_to(BENCHMARK_ROOT)) for path in paths],
            "source_sha256": _combined_sha256(paths, hashes),
            "expected_dispatch_count": case.expected_dispatch_count,
        },
        "target": target,
        "phase": phase,
        "status": "passed" if passed else ("timeout" if timed_out else "failed"),
        "return_code": return_code,
        "timed_out": timed_out,
        "timeout_seconds": case.timeout_seconds,
        "failure": failure,
        "started_at": started_at,
        "ended_at": utc_now(),
        "measurement": {
            "metric": suite.measurement.metric,
            "warmups": warmups,
            "samples": samples,
            "values": timings,
            "statistics": summarize(timings) if len(timings) == samples else None,
        },
        "diagnostics": diagnostics,
        "command": command_record,
        "artifacts": {
            "stdout": _relative(stdout_path, options.output),
            "stderr": _relative(stderr_path, options.output),
            "time": _relative(time_path, options.output),
            "perf_stat": _relative(perf_stat_path, options.output),
            "perf_data": _relative(perf_data_path, options.output) if profile else None,
            "workload": _relative(prepared.output, options.output),
            "throughput": _relative(throughput_path, options.output) if throughput else None,
            "throughput_config": (
                _relative(config_override, options.output) if config_override else None
            ),
        },
        "application_result": dict(payload) if payload is not None else None,
        "throughput_summary": dict(throughput_summary) if throughput_summary else None,
        "provenance": {
            "suite_manifest_sha256": suite.digest,
            "catalog_sha256": catalog_sha256(),
            "config_sha256": hashes.sha256(base_config or prepared.config),
            "derived_config_sha256": (
                hashes.sha256(prepared.config) if config_override else None
            ),
            "dynamic_dependencies": dynamic_dependencies,
            "launcher": {
                "path": str(launcher) if launcher else None,
                "sha256": hashes.sha256(launcher),
            },
            "payload_executable": {
                "path": str(payload_program) if payload_program else None,
                "sha256": hashes.sha256(payload_program),
            },
            "effective_environment": effective_environment,
        },
    }
    write_json(result_dir / "case-result.json", result)
    return result


def _read_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor()


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
                "--",
                ".",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return bool(result.stdout)
    except (OSError, subprocess.SubprocessError):
        return None


def _runner_source_sha256() -> str:
    digest = hashlib.sha256()
    paths = [
        *Path(__file__).resolve().parent.glob("*.py"),
        BENCHMARK_ROOT / "catalog.py",
        BENCHMARK_ROOT / "verify_environment.py",
    ]
    for path in sorted(paths):
        digest.update(path.relative_to(BENCHMARK_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _collect_provenance(
    suite: Suite,
    options: RunOptions,
    targets: Sequence[str],
    hashes: _RunFileHashes,
    dependency_cohort: Mapping[str, Any],
) -> dict[str, Any]:
    config_records: dict[str, Any] = {}
    dummy = suite.cases[0]
    for target in targets:
        prepared = prepare_command(
            suite,
            dummy,
            target,
            options,
            options.output,
            warmups=0,
            samples=1,
            phase="provenance",
        )
        config_records[target] = {
            "path": str(prepared.config),
            "sha256": hashes.sha256(prepared.config),
        }
    rocjitsu = Path(options.rocjitsu).expanduser().resolve() if options.rocjitsu else (
        options.build_dir.expanduser().resolve() / "tools/rocjitsu/rocjitsu"
        if options.build_dir
        else None
    )
    return {
        "runner": {
            "version": __version__,
            "source_sha256": _runner_source_sha256(),
            "python": platform.python_version(),
        },
        "catalog_sha256": catalog_sha256(),
        "suite_manifest_sha256": suite.digest,
        "host": {
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_model": _read_cpu_model(),
            "cpu_count": os.cpu_count(),
        },
        "rocjitsu": {
            "path": str(rocjitsu) if rocjitsu else None,
            "sha256": hashes.sha256(rocjitsu),
        },
        "configs": config_records,
        "source": {
            "path": str(ROCJITSU_ROOT),
            "git_revision": _git_revision(ROCJITSU_ROOT),
            "dirty": _git_dirty(ROCJITSU_ROOT),
        },
        "dependencies": {
            "lock_sha256": hashes.sha256(suite.dependency_lock),
            "verifier_sha256": hashes.sha256(
                BENCHMARK_ROOT / "verify_environment.py"
            ),
            "cohort": dict(dependency_cohort),
        },
        "build_dir": str(options.build_dir.resolve()) if options.build_dir else None,
        "build_type": _cmake_build_type(options.build_dir) if options.build_dir else None,
        "explicit_environment": dict(sorted(options.environment.items())),
    }


def run_suite(
    suite: Suite, options: RunOptions, *, profile: bool = False
) -> dict[str, Any]:
    targets, cases = select_suite(suite, options)
    report = preflight(suite, options, profile=profile)
    failed_checks = [item for item in report["checks"] if not item["passed"]]
    if failed_checks:
        details = "; ".join(
            f"{item['name']}: {item['detail']}" for item in failed_checks
        )
        raise RunnerError(f"preflight failed: {details}")
    dependency_cohort = report.get("dependency_cohort")
    if not isinstance(dependency_cohort, Mapping):
        raise RunnerError("preflight did not produce dependency cohort provenance")

    output = options.output.expanduser().resolve()
    options = dataclasses.replace(options, output=output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RunnerError(f"refusing to overwrite existing output directory: {output}") from exc
    except OSError as exc:
        raise RunnerError(f"cannot create output directory {output}: {exc}") from exc

    started_at = utc_now()
    started = time.perf_counter()
    hashes = _RunFileHashes()
    provenance = _collect_provenance(
        suite, options, targets, hashes, dependency_cohort
    )
    warmups, samples = _effective_counts(suite, options)
    collectors = options.collectors if options.collectors is not None else DEFAULT_COLLECTORS
    results: list[dict[str, Any]] = []
    throughput_results: list[dict[str, Any]] = []
    priming_results: list[dict[str, Any]] = []
    failures: list[str] = []
    budget_exhausted = False

    def budget_is_exhausted() -> bool:
        return time.perf_counter() - started >= suite.budget_seconds

    for case in cases:
        for target in targets:
            if budget_is_exhausted():
                budget_exhausted = True
                break
            root = output / "cases" / case.id / target
            if profile and case.provider == "triton":
                prime = _execute_case(
                    suite,
                    case,
                    target,
                    options,
                    root / "prime",
                    hashes,
                    phase="prime",
                    warmups=0,
                    samples=1,
                    collectors=(),
                )
                priming_results.append(prime)
                if prime["status"] != "passed":
                    failures.append(f"{case.id}/{target}/prime: {prime['failure']}")
                    if options.fail_fast:
                        break
                    continue
                if budget_is_exhausted():
                    budget_exhausted = True
                    break
            if profile:
                result = _execute_case(
                    suite,
                    case,
                    target,
                    options,
                    root / "profile",
                    hashes,
                    phase="profile",
                    warmups=0,
                    samples=1,
                    collectors=("time",),
                    profile=True,
                )
            else:
                result = _execute_case(
                    suite,
                    case,
                    target,
                    options,
                    root / "timing",
                    hashes,
                    phase="timing",
                    warmups=warmups,
                    samples=samples,
                    collectors=collectors,
                )
            results.append(result)
            if result["status"] != "passed":
                failures.append(f"{case.id}/{target}/{result['phase']}: {result['failure']}")
                if options.fail_fast:
                    break
                continue
            if budget_is_exhausted():
                budget_exhausted = True
                break
            if not profile:
                throughput_result = _execute_case(
                    suite,
                    case,
                    target,
                    options,
                    root / "throughput",
                    hashes,
                    phase="throughput",
                    warmups=0,
                    samples=1,
                    collectors=(),
                    throughput=True,
                )
                throughput_results.append(throughput_result)
                if throughput_result["status"] != "passed":
                    failures.append(
                        f"{case.id}/{target}/throughput: {throughput_result['failure']}"
                    )
                    if options.fail_fast:
                        break
                if budget_is_exhausted():
                    budget_exhausted = True
                    break
        if budget_exhausted or (failures and options.fail_fast):
            break

    if budget_is_exhausted():
        budget_exhausted = True
    if budget_exhausted:
        failures.append(f"suite exceeded its {suite.budget_seconds:g}-second budget")

    summaries: list[dict[str, Any]] = []
    for case in cases:
        for target in targets:
            matching = [
                result
                for result in results
                if result["case"]["id"] == case.id and result["target"] == target
            ]
            matching_throughput = [
                result
                for result in throughput_results
                if result["case"]["id"] == case.id and result["target"] == target
            ]
            passed = len(matching) == 1 and matching[0]["status"] == "passed"
            if not profile:
                passed = (
                    passed
                    and len(matching_throughput) == 1
                    and matching_throughput[0]["status"] == "passed"
                )
            summaries.append(
                {
                    "case": case.id,
                    "target": target,
                    "status": "passed" if passed else "failed",
                    "provider": case.provider,
                    "parameters": dict(case.parameters),
                    "metric": suite.measurement.metric,
                    "statistics": (
                        matching[0]["measurement"]["statistics"]
                        if len(matching) == 1
                        else None
                    ),
                    "throughput": (
                        matching_throughput[0].get("throughput_summary")
                        if len(matching_throughput) == 1
                        else None
                    ),
                }
            )

    result = {
        "schema": RUN_SCHEMA,
        "mode": "profile" if profile else "run",
        "status": "passed" if not failures else "failed",
        "started_at": started_at,
        "ended_at": utc_now(),
        "wall_time_seconds": time.perf_counter() - started,
        "suite": {
            "name": suite.name,
            "description": suite.description,
            "manifest": str(suite.path),
            "manifest_sha256": suite.digest,
            "catalog_sha256": catalog_sha256(),
            "targets": list(targets),
            "cases": [case.id for case in cases],
            "budget_seconds": suite.budget_seconds,
        },
        "execution": {
            "measurement": {
                "metric": suite.measurement.metric,
                "warmups": 0 if profile else warmups,
                "samples": 1 if profile else samples,
            },
            "priming": (
                {
                    "provider": "triton",
                    "purpose": "populate the on-disk compilation cache",
                    "warmups": 0,
                    "samples": 1,
                }
                if profile and any(case.provider == "triton" for case in cases)
                else None
            ),
            "warmups_override": options.warmups,
            "samples_override": options.samples,
            "collectors": list(("time", "perf-record") if profile else collectors),
        },
        "provenance": provenance,
        "summaries": summaries,
        "results": results,
        "priming": priming_results,
        "throughput": throughput_results,
        "failures": failures,
    }
    write_json(output / "run.json", result)
    return result
