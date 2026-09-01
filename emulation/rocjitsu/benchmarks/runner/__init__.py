"""Rocjitsu benchmark orchestration and result normalization."""

from ..catalog import CaseSpec
from .manifest import (
    MANIFEST_SCHEMA,
    ManifestError,
    MeasurementPolicy,
    Suite,
    load_suite,
)
from .statistics import summarize

__all__ = [
    "MANIFEST_SCHEMA",
    "CaseSpec",
    "ManifestError",
    "MeasurementPolicy",
    "Suite",
    "load_suite",
    "summarize",
]

__version__ = "0.1.0"
