"""Statistical utilities: confidence intervals, hypothesis tests, uniformity tests."""

from __future__ import annotations

import math
import statistics as stdstats

from scipy import stats as scipy_stats


def confidence_interval(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) using Student's t.

    Raises:
        ValueError: if values is empty.
    """
    if not values:
        raise ValueError("Cannot compute CI of empty sample")
    n = len(values)
    mean = stdstats.fmean(values)
    if n < 2:
        return mean, mean, mean
    std = stdstats.stdev(values)
    se = std / math.sqrt(n)
    alpha = 1.0 - confidence
    t_crit = scipy_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    half = t_crit * se
    return mean, mean - half, mean + half
