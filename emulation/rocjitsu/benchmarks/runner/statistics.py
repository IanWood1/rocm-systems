"""Small, dependency-free statistical helpers used by benchmark reports."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any


def summarize(values: Iterable[float]) -> dict[str, Any]:
    """Return stable descriptive statistics for one set of samples.

    MAD is the median absolute deviation from the median.  The function rejects
    non-finite values because serializing NaN or infinity would produce JSON
    that is not portable between consumers.
    """

    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("at least one sample is required")
    if not all(math.isfinite(value) for value in samples):
        raise ValueError("samples must all be finite")

    median = statistics.median(samples)
    minimum = min(samples)
    maximum = max(samples)
    return {
        "count": len(samples),
        "median": median,
        "mad": statistics.median(abs(value - median) for value in samples),
        "minimum": minimum,
        "maximum": maximum,
        "range": maximum - minimum,
        "samples": samples,
    }
