# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Command construction for built-in benchmark providers."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from ..catalog import BENCHMARK_ROOT, ROCJITSU_ROOT, TARGET_CONFIGS, CaseSpec
from .manifest import Suite


class CommandOptions(Protocol):
    output: Path
    build_dir: Path | None
    python: str
    rocjitsu: str | None
    configs: Mapping[str, str]
    environment: Mapping[str, str]


@dataclasses.dataclass(frozen=True)
class PreparedCommand:
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    config: Path
    output: Path
    payload_program: str


def _build_executable(options: CommandOptions, name: str) -> Path:
    if options.build_dir is None:
        raise ValueError("--build-dir is required for built-in benchmark workloads")
    return options.build_dir.expanduser().resolve() / "benchmarks" / name


def _config(options: CommandOptions, target: str) -> Path:
    override = options.configs.get(target)
    if override is not None:
        return Path(override).expanduser().resolve()
    return TARGET_CONFIGS[target].resolve()


def _rocjitsu(options: CommandOptions) -> Path:
    if options.rocjitsu:
        return Path(options.rocjitsu).expanduser().resolve()
    if options.build_dir is None:
        raise ValueError("--rocjitsu or --build-dir is required")
    return (
        options.build_dir.expanduser().resolve()
        / "tools"
        / "rocjitsu"
        / "rocjitsu"
    )


def prepare_command(
    suite: Suite,
    case: CaseSpec,
    target: str,
    options: CommandOptions,
    result_dir: Path,
    *,
    warmups: int,
    samples: int,
    phase: str,
    config_override: Path | None = None,
) -> PreparedCommand:
    """Build one shell-free command from trusted provider definitions."""

    output = result_dir / "workload.json"
    if case.provider == "native":
        payload = (
            str(
                _build_executable(
                    options, f"rocjitsu-benchmark-native-{target}"
                )
            ),
        )
    elif case.provider == "triton":
        payload = (
            str(Path(options.python).expanduser()),
            "-m",
            "benchmarks.workloads.triton_workloads",
        )
    elif case.provider == "hipblaslt":
        payload = (
            str(_build_executable(options, "rocjitsu-benchmark-hipblaslt")),
        )
    else:  # pragma: no cover - CaseSpec construction prevents this.
        raise ValueError(f"unsupported benchmark provider {case.provider!r}")

    payload += (
        "--case",
        case.id,
        "--expected-target",
        target,
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
        "--output",
        str(output),
    )
    config = config_override.resolve() if config_override else _config(options, target)
    launcher = _rocjitsu(options)
    environment = {
        "PYTHONHASHSEED": "0",
        "TRITON_CACHE_DIR": str(
            options.output.expanduser().resolve() / "cache" / "triton" / target
        ),
        **options.environment,
        "RJ_BENCHMARK_CASE": case.id,
        "RJ_BENCHMARK_TARGET": target,
        "RJ_BENCHMARK_PHASE": phase,
        "RJ_BENCHMARK_RESULT_PATH": str(result_dir / "case-result.json"),
        "RJ_BENCHMARK_THROUGHPUT_PATH": str(result_dir / "throughput.log"),
    }
    return PreparedCommand(
        argv=(str(launcher), "--config", str(config), "--", *payload),
        cwd=ROCJITSU_ROOT,
        environment=environment,
        config=config,
        output=output,
        payload_program=payload[0],
    )


def source_paths(case: CaseSpec) -> tuple[Path, ...]:
    """Resolve catalog-owned source paths independently of suite location."""

    return tuple((BENCHMARK_ROOT / source).resolve() for source in case.sources)
