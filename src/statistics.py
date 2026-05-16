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


def chi_square_uniformity(
    uniforms: list[float], k: int = 10, alpha: float = 0.05
) -> tuple[float, float, bool]:
    """Chi-square test for U ~ Uniform(0, 1).

    Returns (chi2_stat, critical_value, passes_at_alpha).
    """
    if not uniforms:
        raise ValueError("uniforms must be non-empty")
    n = len(uniforms)
    expected = n / k
    observed = [0] * k
    for u in uniforms:
        # Clamp to [0, k-1] in case u==1.0 sneaks in
        bin_idx = min(int(u * k), k - 1)
        observed[bin_idx] += 1
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    critical = scipy_stats.chi2.ppf(1.0 - alpha, df=k - 1)
    passes = chi2 < critical
    return chi2, critical, passes


def ks_uniformity(
    uniforms: list[float], alpha: float = 0.05
) -> tuple[float, float, bool]:
    """One-sample Kolmogorov-Smirnov test against Uniform(0, 1).

    Returns (D_statistic, critical_value, passes_at_alpha).
    """
    if not uniforms:
        raise ValueError("uniforms must be non-empty")
    n = len(uniforms)
    sorted_u = sorted(uniforms)
    d_plus = max((i + 1) / n - u for i, u in enumerate(sorted_u))
    d_minus = max(u - i / n for i, u in enumerate(sorted_u))
    d = max(d_plus, d_minus)
    # Asymptotic critical value (Kolmogorov distribution)
    critical = scipy_stats.kstwobign.ppf(1.0 - alpha) / math.sqrt(n)
    passes = d < critical
    return d, critical, passes
