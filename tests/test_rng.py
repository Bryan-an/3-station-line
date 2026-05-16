# tests/test_rng.py
import math
import statistics
import pytest

from src.rng import LCG, exponential


def test_lcg_reproducibility():
    """Same seed → same sequence."""
    rng1 = LCG(state=42)
    rng2 = LCG(state=42)
    seq1 = [rng1.next_uniform() for _ in range(100)]
    seq2 = [rng2.next_uniform() for _ in range(100)]
    assert seq1 == seq2


def test_lcg_known_first_state_with_seed_12345():
    """X₁ for seed=12345 with Numerical Recipes params, verified manually."""
    rng = LCG(state=12345)
    rng.next_uniform()
    assert rng.state == 87_628_868


def test_uniforms_in_unit_interval():
    """All U values must satisfy 0 <= U < 1."""
    rng = LCG(state=1)
    for _ in range(10_000):
        u = rng.next_uniform()
        assert 0.0 <= u < 1.0


def test_exponential_always_positive():
    rng = LCG(state=42)
    for _ in range(10_000):
        t = exponential(rng, rate=0.25)
        assert t > 0.0


def test_exponential_mean_converges_to_inverse_rate():
    """N=50_000 samples; mean within 2% of 4.0 min (rate=0.25)."""
    rng = LCG(state=42)
    samples = [exponential(rng, rate=0.25) for _ in range(50_000)]
    mean = statistics.mean(samples)
    assert abs(mean - 4.0) < 0.08, f"mean={mean}"
