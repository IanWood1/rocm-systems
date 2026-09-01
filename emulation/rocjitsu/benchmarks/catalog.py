# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical, device-free definitions for built-in Rocjitsu benchmarks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

Provider = Literal["native", "triton", "hipblaslt"]

BENCHMARK_ROOT = Path(__file__).resolve().parent
ROCJITSU_ROOT = BENCHMARK_ROOT.parent
DEPENDENCY_LOCK = BENCHMARK_ROOT / "dependencies.lock.toml"
TARGET_CONFIGS: Mapping[str, Path] = MappingProxyType(
    {
        "gfx950": ROCJITSU_ROOT / "configs/gfx950_mi355x_kmd.json",
        "gfx1250": ROCJITSU_ROOT / "configs/gfx1250_mi455x_kmd.json",
    }
)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[key] = _freeze_mapping(item)
        elif isinstance(item, list):
            result[key] = tuple(item)
        else:
            result[key] = item
    return MappingProxyType(result)


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    """One immutable built-in benchmark definition."""

    id: str
    provider: Provider
    parameters: Mapping[str, Any]
    timeout_seconds: float
    expected_dispatch_count: int | None
    sources: tuple[str, ...]


def _case(
    case_id: str,
    provider: Provider,
    parameters: Mapping[str, Any],
    timeout_seconds: float,
    expected_dispatch_count: int | None,
    *sources: str,
) -> CaseSpec:
    return CaseSpec(
        id=case_id,
        provider=provider,
        parameters=_freeze_mapping(parameters),
        timeout_seconds=timeout_seconds,
        expected_dispatch_count=expected_dispatch_count,
        sources=tuple(sources),
    )


_NATIVE_SOURCES = ("src/native_workloads.hip",)
_TRITON_SOURCES = ("workloads/triton_workloads.py", "catalog.py")
_HIPBLASLT_SOURCES = ("src/hipblaslt_workloads.cpp", "src/json_output.h")

CASES: tuple[CaseSpec, ...] = (
    _case(
        "hip.launch_noop",
        "native",
        {"blocks": 1, "threads_per_block": 64},
        30,
        1,
        *_NATIVE_SOURCES,
    ),
    _case(
        "hip.copy_fp32_32m",
        "native",
        {
            "bytes": 33_554_432,
            "elements": 8_388_608,
            "dtype": "fp32",
            "vector_bytes": 16,
            "grid_size": 256,
            "block_size": 256,
        },
        90,
        1,
        *_NATIVE_SOURCES,
    ),
    _case(
        "hip.vector_add_fp32_tail",
        "native",
        {"elements": 1_048_583, "dtype": "fp32", "block_size": 256},
        90,
        1,
        *_NATIVE_SOURCES,
    ),
    _case(
        "triton.softmax_fp16_aligned",
        "triton",
        {"rows": 64, "columns": 4096, "dtype": "fp16"},
        90,
        1,
        *_TRITON_SOURCES,
    ),
    _case(
        "triton.softmax_fp16_boundary",
        "triton",
        {"rows": 64, "columns": 4097, "dtype": "fp16"},
        90,
        1,
        *_TRITON_SOURCES,
    ),
    _case(
        "triton.rmsnorm_bf16",
        "triton",
        {"rows": 128, "columns": 4096, "dtype": "bf16", "epsilon": 1.0e-5},
        90,
        1,
        *_TRITON_SOURCES,
    ),
    _case(
        "triton.gemm_bf16_aligned",
        "triton",
        {
            "m": 256,
            "n": 256,
            "k": 512,
            "trans_a": "N",
            "trans_b": "N",
            "input_dtype": "bf16",
            "output_dtype": "bf16",
            "accumulator_dtype": "fp32",
        },
        90,
        None,
        *_TRITON_SOURCES,
    ),
    _case(
        "triton.gemm_bf16_ragged",
        "triton",
        {
            "m": 250,
            "n": 250,
            "k": 510,
            "trans_a": "N",
            "trans_b": "N",
            "input_dtype": "bf16",
            "output_dtype": "bf16",
            "accumulator_dtype": "fp32",
        },
        90,
        None,
        *_TRITON_SOURCES,
    ),
    _case(
        "triton.attention_fp16",
        "triton",
        {
            "batch": 1,
            "heads": 8,
            "sequence": 128,
            "head_dimension": 64,
            "causal": False,
            "dtype": "fp16",
        },
        120,
        1,
        *_TRITON_SOURCES,
    ),
    _case(
        "tensile.gemm_fp16",
        "hipblaslt",
        {
            "m": 512,
            "n": 512,
            "k": 512,
            "trans_a": "N",
            "trans_b": "N",
            "batch_count": 1,
            "input_dtype": "fp16",
            "output_dtype": "fp16",
            "accumulator_dtype": "fp32",
            "beta": 0,
        },
        120,
        None,
        *_HIPBLASLT_SOURCES,
    ),
    _case(
        "tensile.gemm_bf16_batched",
        "hipblaslt",
        {
            "m": 256,
            "n": 77,
            "k": 160,
            "trans_a": "N",
            "trans_b": "T",
            "batch_count": 16,
            "input_dtype": "bf16",
            "output_dtype": "bf16",
            "accumulator_dtype": "fp32",
            "beta": 0,
        },
        120,
        None,
        *_HIPBLASLT_SOURCES,
    ),
    _case(
        "tensile.gemm_fp8_scaled",
        "hipblaslt",
        {
            "m": 128,
            "n": 128,
            "k": 128,
            "trans_a": "T",
            "trans_b": "N",
            "batch_count": 1,
            "input_dtype": "fp8",
            "output_dtype": "bf16",
            "accumulator_dtype": "fp32",
            "scalar_scales": True,
            "beta": 0,
        },
        120,
        None,
        *_HIPBLASLT_SOURCES,
    ),
)

CASE_BY_ID: Mapping[str, CaseSpec] = MappingProxyType(
    {case.id: case for case in CASES}
)
TRITON_CASES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {case.id: case.parameters for case in CASES if case.provider == "triton"}
)


def parameters_for(case_id: str) -> dict[str, Any]:
    """Return a mutable JSON-compatible copy of one case's parameters."""

    def thaw(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    return thaw(CASE_BY_ID[case_id].parameters)


def validated_triton_case(
    case_id: str, parameters: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return canonical Triton parameters and reject an inconsistent override."""

    expected = parameters_for(case_id)
    if parameters is not None and dict(parameters) != expected:
        raise ValueError(f"{case_id} parameters do not match the built-in catalog")
    return expected


def catalog_sha256() -> str:
    """Return a path-independent digest of every built-in case definition."""

    value = [
        {
            "id": case.id,
            "provider": case.provider,
            "parameters": parameters_for(case.id),
            "timeout_seconds": case.timeout_seconds,
            "expected_dispatch_count": case.expected_dispatch_count,
            "sources": case.sources,
        }
        for case in CASES
    ]
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
