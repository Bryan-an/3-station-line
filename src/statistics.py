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


def t_test_greater(
    values: list[float], threshold: float
) -> tuple[float, float]:
    """One-sided t-test for H1: mean > threshold.

    Returns (t_statistic, p_value). p_value < 0.05 → reject H0.
    If sample variance is 0, returns (inf or -inf, 0 or 1) depending on sign.
    """
    if len(values) < 2:
        raise ValueError("t-test requires at least 2 samples")
    n = len(values)
    mean = stdstats.fmean(values)
    std = stdstats.stdev(values)
    if std == 0.0:
        if mean > threshold:
            return math.inf, 0.0
        if mean < threshold:
            return -math.inf, 1.0
        return 0.0, 0.5
    se = std / math.sqrt(n)
    t = (mean - threshold) / se
    # P(T_{n-1} > t)
    p = 1.0 - scipy_stats.t.cdf(t, df=n - 1)
    return t, p
