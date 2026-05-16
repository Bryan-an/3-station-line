import random

import pytest

from src.statistics import (
    confidence_interval,
)


def test_ci_zero_variance():
    """Constant data → CI collapses to the mean."""
    mean, lo, hi = confidence_interval([10.0] * 30)
    assert mean == pytest.approx(10.0)
    assert lo == pytest.approx(10.0)
    assert hi == pytest.approx(10.0)


def test_ci_contains_population_mean_for_normal_data():
    random.seed(42)
    values = [random.gauss(10.0, 1.0) for _ in range(100)]
    mean, lo, hi = confidence_interval(values, confidence=0.95)
    assert lo <= 10.0 <= hi
    assert 9.5 < mean < 10.5


def test_ci_returns_none_or_raises_on_empty():
    with pytest.raises(ValueError):
        confidence_interval([])


def test_ci_requires_at_least_two_for_meaningful_interval():
    """Single value: CI not defined (std undefined). Convention: lo=hi=value."""
    mean, lo, hi = confidence_interval([5.0])
    assert mean == 5.0
    assert lo == 5.0
    assert hi == 5.0
