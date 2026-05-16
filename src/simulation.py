"""Discrete event simulation of a 3-station serial assembly line.

State machine for each station:
  E1: BUSY | BLOCKED          (never FREE — input is infinite)
  E2: BUSY | BLOCKED | FREE
  E3: BUSY | FREE             (never BLOCKED — output is infinite)

Buffer semantics: B1 (between E1 and E2) holds pieces waiting; the piece
inside E2 is NOT counted in B1. Same for B2 / E3.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

from src.rng import LCG, exponential


# Event types
FIN_E1 = "FIN_E1"
FIN_E2 = "FIN_E2"
FIN_E3 = "FIN_E3"

# Priority for simultaneous events: downstream first (frees space upstream)
PRIORITY = {FIN_E3: 0, FIN_E2: 1, FIN_E1: 2}


@dataclass(frozen=True)
class SimulationConfig:
    k1: int
    k2: int
    rate: float = 0.25        # λ (pieces/minute)
    duration: float = 480.0   # minutes
    seed: int = 12_345


@dataclass
class SimulationResult:
    throughput: float                       # pieces / hour
    flow_times: list[float]                 # one per completed piece
    avg_wip: float                          # time-weighted (B1 + B2)
    pct_blocked_e1: float
    pct_blocked_e2: float
    pct_starved_e2: float
    pct_starved_e3: float
    busy_e1: float
    busy_e2: float
    busy_e3: float
    pieces_completed: int
    wip_trace: list[tuple[float, int]]      # (t, B1+B2) after each event
    event_log: list[dict]                   # per-event snapshot


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run one replication. Pure: same config → same result."""
    # Placeholder — replaced in Task 2.2
    return SimulationResult(
        throughput=0.0,
        flow_times=[],
        avg_wip=0.0,
        pct_blocked_e1=0.0,
        pct_blocked_e2=0.0,
        pct_starved_e2=0.0,
        pct_starved_e3=0.0,
        busy_e1=0.0,
        busy_e2=0.0,
        busy_e3=0.0,
        pieces_completed=0,
        wip_trace=[(0.0, 0)],
        event_log=[{"t": 0.0, "event_type": "INIT"}],
    )
