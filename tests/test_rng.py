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
