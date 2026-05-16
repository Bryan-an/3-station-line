# src/rng.py
"""Linear Congruential Generator and inverse-transform exponential sampling.

Parameters: Numerical Recipes (Press et al., 1992).
Hull-Dobell conditions verified — period = 2^32.
"""

import math
from dataclasses import dataclass


@dataclass
class LCG:
    """Linear Congruential Generator with mutable internal state."""

    a: int = 1_664_525
    c: int = 1_013_904_223
    m: int = 2 ** 32
    state: int = 12_345

    def next_uniform(self) -> float:
        """Advance the state and return the next U in [0, 1)."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m


def exponential(rng: LCG, rate: float) -> float:
    """Inverse-transform sampling of Exponential(rate).

    Excludes U == 0 to avoid log(0) → -inf.
    """
    u = rng.next_uniform()
    while u == 0.0:
        u = rng.next_uniform()
    return -math.log(u) / rate
