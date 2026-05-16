# Streamlit Simulation App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit web app that simulates a 3-station assembly line with finite buffers, exponential service times, blocking and starvation; with own LCG, discrete event simulation, statistical analysis, deployable to Streamlit Community Cloud, plus a CLI to export the event trace to Excel.

**Architecture:** Modular Python project. `src/` contains pure modules (`rng.py`, `simulation.py`, `statistics.py`, `plotting.py`); `app.py` is a thin Streamlit orchestration layer; `scripts/excel_trace.py` is a CLI reusing `src/`. Tests under `tests/` cover invariants and properties.

**Tech Stack:** Python 3.12 (3.11 not available on dev machine; 3.12 also supported by Streamlit Cloud) · Streamlit 1.32 · matplotlib 3.8 · scipy 1.12 · pandas 2.2 · openpyxl 3.1 · pytest 8.1 · GitHub Actions · Streamlit Community Cloud.

**Reference:** Spec at `docs/superpowers/specs/2026-05-16-app-architecture-design.md`.

---

## File structure (created across tasks)

```
3-station-line/
├── app.py                          # Phase 5
├── src/
│   ├── __init__.py
│   ├── rng.py                      # Phase 1
│   ├── simulation.py               # Phase 2
│   ├── statistics.py               # Phase 3
│   └── plotting.py                 # Phase 4
├── scripts/
│   └── excel_trace.py              # Phase 6
├── tests/
│   ├── __init__.py
│   ├── test_rng.py                 # Phase 1
│   ├── test_simulation.py          # Phase 2
│   └── test_statistics.py          # Phase 3
├── .github/workflows/test.yml      # Phase 7
├── .streamlit/config.toml          # Phase 7
├── requirements.txt                # Phase 0
├── .gitignore                      # Phase 0
└── README.md                       # Phase 7
```

---

## Phase 0 — Project setup

### Task 0.1: Initialize repo and scaffolding

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `src/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `scripts/` directory

- [ ] **Step 1: Initialize git**

```bash
cd /Users/bryan-andagoya/Development/personal/3-station-line-project/3-station-line
git init
git branch -M main
```

Expected: `Initialized empty Git repository...`

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
env/
.pytest_cache/
.coverage
htmlcov/
*.egg-info/

# OS
.DS_Store

# IDE
.vscode/
.idea/

# Generated artifacts
*.xlsx
*.csv
!docs/**/*.csv
!docs/**/*.xlsx

# Streamlit
.streamlit/secrets.toml
```

- [ ] **Step 3: Create `requirements.txt`**

```
streamlit==1.32.0
numpy==1.26.4
scipy==1.12.0
matplotlib==3.8.3
pandas==2.2.1
openpyxl==3.1.2
pytest==8.1.1
```

- [ ] **Step 4: Create empty package files**

```bash
mkdir -p src tests scripts
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 5: Create venv and install**

```bash
python3.12 -m venv .venv      # or python3.11 if available
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Expected: Successful installation, no errors.

- [ ] **Step 6: Verify install**

```bash
python -c "import streamlit, numpy, scipy, matplotlib, pandas, openpyxl, pytest; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt src/__init__.py tests/__init__.py
git commit -m "chore: initial project scaffolding"
```

---

## Phase 1 — Random number generation (`src/rng.py`)

### Task 1.1: LCG reproducibility test

**Files:**
- Create: `tests/test_rng.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_rng.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.rng'` (or similar import error).

- [ ] **Step 3: Implement `src/rng.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_rng.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rng.py tests/test_rng.py
git commit -m "feat(rng): add LCG with Numerical Recipes parameters"
```

### Task 1.2: LCG range + exponential property tests

**Files:**
- Modify: `tests/test_rng.py`

- [ ] **Step 1: Add three more tests**

Append to `tests/test_rng.py`:

```python
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
```

- [ ] **Step 2: Run all rng tests**

```bash
pytest tests/test_rng.py -v
```

Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_rng.py
git commit -m "test(rng): add range and exponential convergence tests"
```

---

## Phase 2 — Discrete event simulation (`src/simulation.py`)

The core. Built in three layers: dataclasses → minimal-skeleton run loop → handlers with cascade logic → invariant tests.

### Task 2.1: Dataclasses and module skeleton

**Files:**
- Create: `src/simulation.py`
- Create: `tests/test_simulation.py`

- [ ] **Step 1: Write a smoke test for the config dataclass**

```python
# tests/test_simulation.py
import pytest

from src.simulation import SimulationConfig, SimulationResult, run_simulation


def test_config_defaults():
    cfg = SimulationConfig(k1=3, k2=3)
    assert cfg.rate == 0.25
    assert cfg.duration == 480.0
    assert cfg.seed == 12_345


def test_run_simulation_returns_result_with_expected_fields():
    cfg = SimulationConfig(k1=3, k2=3, duration=500.0, seed=42)
    result = run_simulation(cfg)
    assert isinstance(result, SimulationResult)
    assert result.pieces_completed >= 0
    assert result.throughput >= 0.0
    assert len(result.event_log) > 0
    assert len(result.wip_trace) > 0
```

- [ ] **Step 2: Run test (expect import failure)**

```bash
pytest tests/test_simulation.py -v
```

Expected: `ImportError: cannot import name 'SimulationConfig'`

- [ ] **Step 3: Create `src/simulation.py` skeleton**

```python
# src/simulation.py
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
```

- [ ] **Step 4: Run smoke tests (expect pass with placeholder)**

```bash
pytest tests/test_simulation.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/simulation.py tests/test_simulation.py
git commit -m "feat(sim): dataclasses and module skeleton"
```

### Task 2.2: Implement the run loop with all handlers

The simulation is too coupled to TDD per-handler; we implement the full loop as one cohesive change and immediately gate it behind invariant tests.

**Files:**
- Modify: `src/simulation.py` (replace `run_simulation`)

- [ ] **Step 1: Replace `run_simulation` with the full implementation**

Replace the placeholder body of `run_simulation` in `src/simulation.py` with:

```python
def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run one replication. Pure: same config → same result."""
    rng = LCG(state=config.seed)

    def sample_duration() -> tuple[float, float]:
        """Returns (u_used, duration)."""
        u = rng.next_uniform()
        while u == 0.0:
            u = rng.next_uniform()
        return u, -math.log(u) / config.rate

    # --- State ---
    t = 0.0
    fel: list[tuple[float, int, int, str, int]] = []
    tiebreaker = 0  # ensures stable order for equal (t, priority)

    e1_state, e1_piece = "BUSY", None
    e2_state, e2_piece = "FREE", None
    e3_state, e3_piece = "FREE", None
    b1: deque[int] = deque()
    b2: deque[int] = deque()

    next_piece_id = 1
    pieces_completed = 0

    t_busy_e1 = t_busy_e2 = t_busy_e3 = 0.0
    t_blocked_e1 = t_blocked_e2 = 0.0
    t_starved_e2 = t_starved_e3 = 0.0
    area_wip = 0.0

    piece_entry: dict[int, float] = {}
    piece_exit: dict[int, float] = {}

    wip_trace: list[tuple[float, int]] = [(0.0, 0)]
    event_log: list[dict] = []

    def push_event(when: float, event_type: str, piece: int) -> None:
        nonlocal tiebreaker
        heapq.heappush(
            fel, (when, PRIORITY[event_type], tiebreaker, event_type, piece)
        )
        tiebreaker += 1

    def accumulate_until(t_next: float) -> None:
        """Add (t_next - t) to the relevant accumulators based on current state."""
        nonlocal area_wip, t_busy_e1, t_busy_e2, t_busy_e3
        nonlocal t_blocked_e1, t_blocked_e2, t_starved_e2, t_starved_e3
        delta = t_next - t
        if delta <= 0.0:
            return
        if e1_state == "BUSY":
            t_busy_e1 += delta
        elif e1_state == "BLOCKED":
            t_blocked_e1 += delta
        if e2_state == "BUSY":
            t_busy_e2 += delta
        elif e2_state == "BLOCKED":
            t_blocked_e2 += delta
        elif e2_state == "FREE":
            t_starved_e2 += delta
        if e3_state == "BUSY":
            t_busy_e3 += delta
        elif e3_state == "FREE":
            t_starved_e3 += delta
        area_wip += (len(b1) + len(b2)) * delta

    def snapshot(event_type: str, piece_id, u_used, duration_used, note: str) -> None:
        event_log.append({
            "t": t,
            "event_type": event_type,
            "piece_id": piece_id,
            "u": u_used,
            "duration": duration_used,
            "e1_state": e1_state,
            "e2_state": e2_state,
            "e3_state": e3_state,
            "e1_piece": e1_piece,
            "e2_piece": e2_piece,
            "e3_piece": e3_piece,
            "b1_count": len(b1),
            "b2_count": len(b2),
            "wip": len(b1) + len(b2),
            "pieces_completed": pieces_completed,
            "next_piece_id": next_piece_id,
            "note": note,
        })

    # --- Initialize: E1 takes piece 1 immediately ---
    e1_piece = next_piece_id
    piece_entry[next_piece_id] = 0.0
    next_piece_id += 1
    u0, d0 = sample_duration()
    push_event(d0, FIN_E1, e1_piece)
    snapshot("INIT", e1_piece, u0, d0, "E1 starts piece 1")

    # --- Main loop ---
    while fel:
        t_next, _prio, _tb, event_type, piece = heapq.heappop(fel)
        if t_next >= config.duration:
            accumulate_until(config.duration)
            t = config.duration
            break

        accumulate_until(t_next)
        t = t_next

        if event_type == FIN_E1:
            if len(b1) < config.k1:
                b1.append(e1_piece)
                e1_piece = next_piece_id
                piece_entry[next_piece_id] = t
                next_piece_id += 1
                u, d = sample_duration()
                push_event(t + d, FIN_E1, e1_piece)
                # E1 remains BUSY
                snapshot(FIN_E1, piece, u, d, "E1 → B1; started next piece")
                if e2_state == "FREE":
                    e2_piece = b1.popleft()
                    e2_state = "BUSY"
                    u2, d2 = sample_duration()
                    push_event(t + d2, FIN_E2, e2_piece)
                    snapshot(FIN_E1, piece, u2, d2, "E2 took from B1")
            else:
                e1_state = "BLOCKED"
                snapshot(FIN_E1, piece, None, None, "B1 full → E1 BLOCKED")

        elif event_type == FIN_E2:
            if len(b2) < config.k2:
                b2.append(e2_piece)
                # Cascade: E1 was blocked, B1 now has space
                if e1_state == "BLOCKED":
                    b1.append(e1_piece)
                    e1_piece = next_piece_id
                    piece_entry[next_piece_id] = t
                    next_piece_id += 1
                    u_e1, d_e1 = sample_duration()
                    push_event(t + d_e1, FIN_E1, e1_piece)
                    e1_state = "BUSY"
                    snapshot(FIN_E2, piece, u_e1, d_e1, "E1 unblocked")
                # E2 tries to take next
                if len(b1) > 0:
                    e2_piece = b1.popleft()
                    u_e2, d_e2 = sample_duration()
                    push_event(t + d_e2, FIN_E2, e2_piece)
                    e2_state = "BUSY"
                    snapshot(FIN_E2, piece, u_e2, d_e2, "E2 took next from B1")
                else:
                    e2_state = "FREE"
                    e2_piece = None
                    snapshot(FIN_E2, piece, None, None, "E2 FREE (starved)")
                if e3_state == "FREE":
                    e3_piece = b2.popleft()
                    e3_state = "BUSY"
                    u_e3, d_e3 = sample_duration()
                    push_event(t + d_e3, FIN_E3, e3_piece)
                    snapshot(FIN_E2, piece, u_e3, d_e3, "E3 took from B2")
            else:
                e2_state = "BLOCKED"
                snapshot(FIN_E2, piece, None, None, "B2 full → E2 BLOCKED")

        elif event_type == FIN_E3:
            piece_exit[e3_piece] = t
            pieces_completed += 1
            completed_piece = e3_piece
            e3_piece = None
            # Cascade: E2 was blocked
            if e2_state == "BLOCKED":
                b2.append(e2_piece)
                if len(b1) > 0:
                    e2_piece = b1.popleft()
                    u_e2, d_e2 = sample_duration()
                    push_event(t + d_e2, FIN_E2, e2_piece)
                    e2_state = "BUSY"
                    snapshot(FIN_E3, completed_piece, u_e2, d_e2, "E2 unblocked, took from B1")
                    if e1_state == "BLOCKED":
                        b1.append(e1_piece)
                        e1_piece = next_piece_id
                        piece_entry[next_piece_id] = t
                        next_piece_id += 1
                        u_e1, d_e1 = sample_duration()
                        push_event(t + d_e1, FIN_E1, e1_piece)
                        e1_state = "BUSY"
                        snapshot(FIN_E3, completed_piece, u_e1, d_e1, "E1 unblocked (cascade)")
                else:
                    e2_state = "FREE"
                    e2_piece = None
                    snapshot(FIN_E3, completed_piece, None, None, "E2 unblocked but B1 empty")
            # E3 tries to take next
            if len(b2) > 0:
                e3_piece = b2.popleft()
                e3_state = "BUSY"
                u_e3, d_e3 = sample_duration()
                push_event(t + d_e3, FIN_E3, e3_piece)
                snapshot(FIN_E3, completed_piece, u_e3, d_e3, "E3 took next from B2")
            else:
                e3_state = "FREE"
                snapshot(FIN_E3, completed_piece, None, None, "E3 FREE (starved)")

        wip_trace.append((t, len(b1) + len(b2)))

    # --- Final stats ---
    duration = config.duration
    flow_times = [piece_exit[pid] - piece_entry[pid] for pid in piece_exit]
    throughput = pieces_completed / (duration / 60.0)
    avg_wip = area_wip / duration

    return SimulationResult(
        throughput=throughput,
        flow_times=flow_times,
        avg_wip=avg_wip,
        pct_blocked_e1=t_blocked_e1 / duration,
        pct_blocked_e2=t_blocked_e2 / duration,
        pct_starved_e2=t_starved_e2 / duration,
        pct_starved_e3=t_starved_e3 / duration,
        busy_e1=t_busy_e1 / duration,
        busy_e2=t_busy_e2 / duration,
        busy_e3=t_busy_e3 / duration,
        pieces_completed=pieces_completed,
        wip_trace=wip_trace,
        event_log=event_log,
    )
```

- [ ] **Step 2: Run existing smoke tests**

```bash
pytest tests/test_simulation.py -v
```

Expected: 2 passed; `pieces_completed > 0` and `len(event_log) > 1`.

- [ ] **Step 3: Manual smoke check**

Run:
```bash
python -c "from src.simulation import SimulationConfig, run_simulation; r = run_simulation(SimulationConfig(k1=3, k2=3, duration=480, seed=42)); print(f'throughput={r.throughput:.2f}/h pieces={r.pieces_completed} avg_wip={r.avg_wip:.2f}')"
```

Expected: throughput between ~10 and ~14, pieces ~80-110, avg_wip between ~0.5 and ~5.

- [ ] **Step 4: Commit**

```bash
git add src/simulation.py
git commit -m "feat(sim): implement DES run loop with blocking/starvation cascade"
```

### Task 2.3: Invariant tests (capacity, conservation, fractions)

**Files:**
- Modify: `tests/test_simulation.py`

- [ ] **Step 1: Add invariant tests**

Append to `tests/test_simulation.py`:

```python
@pytest.mark.parametrize("k1,k2,seed", [
    (1, 1, 42), (3, 3, 42), (10, 10, 42),
    (1, 10, 99), (10, 1, 99), (2, 5, 777),
])
def test_buffer_never_exceeds_capacity(k1, k2, seed):
    cfg = SimulationConfig(k1=k1, k2=k2, seed=seed, duration=500.0)
    result = run_simulation(cfg)
    for event in result.event_log:
        assert event["b1_count"] <= k1, f"B1 overflow at t={event['t']}"
        assert event["b2_count"] <= k2, f"B2 overflow at t={event['t']}"


@pytest.mark.parametrize("seed", [1, 42, 999, 12_345])
def test_piece_conservation(seed):
    """started = completed + still_in_system."""
    cfg = SimulationConfig(k1=3, k2=3, seed=seed, duration=500.0)
    result = run_simulation(cfg)
    final = result.event_log[-1]
    pieces_started = final["next_piece_id"] - 1
    in_stations = sum(
        1 for s in (final["e1_state"], final["e2_state"], final["e3_state"])
        if s in ("BUSY", "BLOCKED")
    )
    in_buffers = final["b1_count"] + final["b2_count"]
    assert pieces_started == result.pieces_completed + in_stations + in_buffers


def test_e1_never_starves():
    """E1 has infinite input."""
    result = run_simulation(SimulationConfig(k1=3, k2=3, seed=42, duration=500.0))
    assert all(e["e1_state"] != "FREE" for e in result.event_log)


def test_e3_never_blocks():
    """E3 has infinite output."""
    result = run_simulation(SimulationConfig(k1=3, k2=3, seed=42, duration=500.0))
    assert all(e["e3_state"] != "BLOCKED" for e in result.event_log)


def test_state_fractions_sum_to_at_most_one():
    result = run_simulation(SimulationConfig(k1=3, k2=3, seed=42, duration=500.0))
    eps = 1e-9
    assert result.busy_e1 + result.pct_blocked_e1 <= 1.0 + eps
    assert (
        result.busy_e2 + result.pct_blocked_e2 + result.pct_starved_e2
        <= 1.0 + eps
    )
    assert result.busy_e3 + result.pct_starved_e3 <= 1.0 + eps
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_simulation.py -v
```

Expected: all parametrized cases pass (11 new test cases + 2 smoke = 13).

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulation.py
git commit -m "test(sim): invariants for buffer capacity, conservation, state fractions"
```

### Task 2.4: Reproducibility and theoretical-limit tests

**Files:**
- Modify: `tests/test_simulation.py`

- [ ] **Step 1: Add tests**

Append to `tests/test_simulation.py`:

```python
import statistics as stdstats


def test_reproducibility_bit_for_bit():
    cfg = SimulationConfig(k1=3, k2=3, seed=42, duration=500.0)
    r1 = run_simulation(cfg)
    r2 = run_simulation(cfg)
    assert r1.throughput == r2.throughput
    assert r1.flow_times == r2.flow_times
    assert r1.pieces_completed == r2.pieces_completed


def test_throughput_below_theoretical_max():
    """Theoretical max = 60/4 = 15 pieces/h; allow small finite-sample margin."""
    for seed in [1, 42, 999]:
        result = run_simulation(SimulationConfig(k1=10, k2=10, seed=seed, duration=960.0))
        assert result.throughput < 15.5


def test_larger_buffers_give_higher_average_throughput():
    """K=10,10 should outperform K=1,1 averaged across 20 replications."""
    n = 20
    tp_small = [
        run_simulation(SimulationConfig(k1=1, k2=1, seed=s, duration=960.0)).throughput
        for s in range(1, n + 1)
    ]
    tp_large = [
        run_simulation(SimulationConfig(k1=10, k2=10, seed=s, duration=960.0)).throughput
        for s in range(1, n + 1)
    ]
    assert stdstats.mean(tp_large) > stdstats.mean(tp_small)


def test_degenerate_k1_k2_equal_one_does_not_crash():
    result = run_simulation(SimulationConfig(k1=1, k2=1, seed=42, duration=500.0))
    assert result.throughput > 0.0
    assert result.pct_blocked_e1 > 0.0   # blocking expected with K=1
```

- [ ] **Step 2: Run full simulation suite**

```bash
pytest tests/test_simulation.py -v
```

Expected: all green. The `larger_buffers` test takes ~30 seconds (20×2 replications of 960 min).

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulation.py
git commit -m "test(sim): reproducibility and theoretical-limit checks"
```

---

## Phase 3 — Statistics (`src/statistics.py`)

### Task 3.1: Confidence interval

**Files:**
- Create: `src/statistics.py`
- Create: `tests/test_statistics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_statistics.py
import random

import pytest

from src.statistics import (
    confidence_interval,
    t_test_greater,
    chi_square_uniformity,
    ks_uniformity,
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
```

- [ ] **Step 2: Run (expect import error)**

```bash
pytest tests/test_statistics.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `src/statistics.py` with `confidence_interval`**

```python
# src/statistics.py
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_statistics.py::test_ci_zero_variance tests/test_statistics.py::test_ci_contains_population_mean_for_normal_data tests/test_statistics.py::test_ci_returns_none_or_raises_on_empty tests/test_statistics.py::test_ci_requires_at_least_two_for_meaningful_interval -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/statistics.py tests/test_statistics.py
git commit -m "feat(stats): confidence_interval using Student's t"
```

### Task 3.2: One-sided t-test

**Files:**
- Modify: `src/statistics.py`
- Modify: `tests/test_statistics.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_statistics.py`:

```python
def test_t_test_rejects_when_clearly_greater():
    t, p = t_test_greater([15.0] * 10, threshold=12.0)
    assert p < 0.001


def test_t_test_does_not_reject_when_at_threshold():
    """Constant values at threshold → t = 0, p ≈ 0.5."""
    t, p = t_test_greater([12.0] * 10, threshold=12.0)
    assert p == pytest.approx(0.5, abs=0.01) or math.isnan(t)


def test_t_test_p_value_close_to_one_when_clearly_less():
    t, p = t_test_greater([5.0] * 10, threshold=12.0)
    assert p > 0.99
```

- [ ] **Step 2: Run (expect failure on import / definition)**

```bash
pytest tests/test_statistics.py -k "t_test" -v
```

Expected: FAIL (either ImportError if not added yet, or NameError).

- [ ] **Step 3: Add `t_test_greater` to `src/statistics.py`**

Append:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_statistics.py -k "t_test" -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/statistics.py tests/test_statistics.py
git commit -m "feat(stats): one-sided t_test_greater for hypothesis testing"
```

### Task 3.3: Chi-square uniformity test

**Files:**
- Modify: `src/statistics.py`
- Modify: `tests/test_statistics.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_statistics.py`:

```python
from src.rng import LCG


def test_chi_square_passes_for_good_lcg():
    rng = LCG(state=42)
    uniforms = [rng.next_uniform() for _ in range(10_000)]
    chi2, crit, passes = chi_square_uniformity(uniforms, k=10)
    assert passes
    assert chi2 < crit


def test_chi_square_fails_for_obviously_non_uniform_data():
    """All values in [0, 0.1) → massive deviation, should fail."""
    uniforms = [0.05] * 1_000
    chi2, crit, passes = chi_square_uniformity(uniforms, k=10)
    assert not passes
    assert chi2 > crit
```

- [ ] **Step 2: Run (expect failure)**

```bash
pytest tests/test_statistics.py -k "chi_square" -v
```

Expected: FAIL.

- [ ] **Step 3: Add `chi_square_uniformity` to `src/statistics.py`**

Append:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_statistics.py -k "chi_square" -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/statistics.py tests/test_statistics.py
git commit -m "feat(stats): chi_square_uniformity test"
```

### Task 3.4: Kolmogorov-Smirnov uniformity test

**Files:**
- Modify: `src/statistics.py`
- Modify: `tests/test_statistics.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_statistics.py`:

```python
def test_ks_passes_for_good_lcg():
    rng = LCG(state=42)
    uniforms = [rng.next_uniform() for _ in range(5_000)]
    d, crit, passes = ks_uniformity(uniforms)
    assert passes
    assert d < crit


def test_ks_fails_for_clustered_data():
    uniforms = [0.5] * 100
    d, crit, passes = ks_uniformity(uniforms)
    assert not passes
```

- [ ] **Step 2: Run (expect failure)**

```bash
pytest tests/test_statistics.py -k "ks_" -v
```

Expected: FAIL.

- [ ] **Step 3: Add `ks_uniformity` to `src/statistics.py`**

Append:

```python
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
```

- [ ] **Step 4: Run all statistics tests**

```bash
pytest tests/test_statistics.py -v
```

Expected: all green (~11 tests).

- [ ] **Step 5: Commit**

```bash
git add src/statistics.py tests/test_statistics.py
git commit -m "feat(stats): KS uniformity test"
```

---

## Phase 4 — Plotting (`src/plotting.py`)

No unit tests for plotting — verified visually when the app runs.

### Task 4.1: Implement all five plotting functions

**Files:**
- Create: `src/plotting.py`

- [ ] **Step 1: Write `src/plotting.py`**

```python
# src/plotting.py
"""Matplotlib figure builders. Each function returns a Figure; no side effects."""

from __future__ import annotations

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def plot_wip_evolution(wip_trace: list[tuple[float, int]], title: str = "WIP en el tiempo") -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if wip_trace:
        ts, wips = zip(*wip_trace)
        ax.step(ts, wips, where="post", linewidth=1.2)
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("WIP (B1 + B2)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_flow_time_histogram(flow_times: list[float], title: str = "Distribución de tiempo de flujo") -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if flow_times:
        ax.hist(flow_times, bins=30, edgecolor="black", alpha=0.75)
        mean = float(np.mean(flow_times))
        ax.axvline(mean, color="red", linestyle="--", label=f"Media={mean:.2f} min")
        ax.legend()
    ax.set_xlabel("Tiempo de flujo (min)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_throughput_bar_with_ci(
    throughputs: list[float], ci: tuple[float, float, float] | None = None
) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    indices = np.arange(1, len(throughputs) + 1)
    ax.bar(indices, throughputs, alpha=0.7, edgecolor="black")
    if ci is not None:
        mean, lo, hi = ci
        ax.axhline(mean, color="red", linestyle="-", linewidth=1.5, label=f"Media={mean:.2f}")
        ax.axhspan(lo, hi, color="red", alpha=0.15, label=f"IC 95% [{lo:.2f}, {hi:.2f}]")
        ax.legend()
    ax.set_xlabel("Réplica #")
    ax.set_ylabel("Throughput (piezas/h)")
    ax.set_title("Throughput por réplica")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_heatmap(grid: dict[tuple[int, int], float], metric_name: str = "Throughput") -> Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    k1s = sorted({k for k, _ in grid})
    k2s = sorted({k for _, k in grid})
    matrix = np.array([[grid[(k1, k2)] for k1 in k1s] for k2 in k2s])
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(k1s)), k1s)
    ax.set_yticks(range(len(k2s)), k2s)
    ax.set_xlabel("K1")
    ax.set_ylabel("K2")
    ax.set_title(f"{metric_name} por configuración (K1, K2)")
    for i, k2 in enumerate(k2s):
        for j, k1 in enumerate(k1s):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white", fontsize=7)
    # Mark max
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    ax.plot(max_idx[1], max_idx[0], "o", markersize=18, markerfacecolor="none", markeredgecolor="red", markeredgewidth=2)
    fig.colorbar(im, ax=ax, label=metric_name)
    fig.tight_layout()
    return fig


def plot_comparison_wip(
    trace_a: list[tuple[float, int]],
    trace_b: list[tuple[float, int]],
    label_a: str = "Config A",
    label_b: str = "Config B",
) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if trace_a:
        ta, wa = zip(*trace_a)
        ax.step(ta, wa, where="post", label=label_a, alpha=0.8)
    if trace_b:
        tb, wb = zip(*trace_b)
        ax.step(tb, wb, where="post", label=label_b, alpha=0.8)
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("WIP (B1 + B2)")
    ax.set_title("Comparación de WIP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Smoke import**

```bash
python -c "from src.plotting import plot_wip_evolution, plot_flow_time_histogram, plot_throughput_bar_with_ci, plot_heatmap, plot_comparison_wip; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/plotting.py
git commit -m "feat(plot): matplotlib figure builders for all charts"
```

---

## Phase 5 — Streamlit app (`app.py`)

### Task 5.1: Bootstrap with sidebar and run button

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write `app.py` skeleton**

```python
# app.py
"""Streamlit UI for the 3-station line simulation."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.simulation import SimulationConfig, run_simulation
from src.statistics import (
    confidence_interval,
    t_test_greater,
    chi_square_uniformity,
    ks_uniformity,
)
from src.plotting import (
    plot_wip_evolution,
    plot_flow_time_histogram,
    plot_throughput_bar_with_ci,
    plot_heatmap,
    plot_comparison_wip,
)
from src.rng import LCG, exponential


SEED_PRIME = 7919  # decorrelates replication seeds


st.set_page_config(
    page_title="Línea de 3 Estaciones",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def validate_inputs(k1: int, k2: int, duration: float, n_replicas: int, seed: int) -> list[str]:
    errors = []
    if not (1 <= k1 <= 10):
        errors.append("K1 debe estar entre 1 y 10")
    if not (1 <= k2 <= 10):
        errors.append("K2 debe estar entre 1 y 10")
    if duration < 500:
        errors.append("Duración mínima: 500 minutos")
    if duration > 50_000:
        errors.append("Duración > 50_000 min puede congelar el navegador")
    if not (1 <= n_replicas <= 20):
        errors.append("Réplicas entre 1 y 20")
    if seed < 0 or seed >= 2 ** 31:
        errors.append("Semilla fuera de rango [0, 2³¹)")
    return errors


def run_replications(k1: int, k2: int, duration: float, n_replicas: int, base_seed: int):
    results = []
    progress = st.progress(0.0, text="Simulando...")
    for i in range(n_replicas):
        cfg = SimulationConfig(
            k1=k1, k2=k2, duration=duration, seed=base_seed + i * SEED_PRIME
        )
        results.append(run_simulation(cfg))
        progress.progress((i + 1) / n_replicas, text=f"Réplica {i+1}/{n_replicas}")
    progress.empty()
    return results


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Parámetros")
    k1 = st.slider("K1 (buffer entre E1 y E2)", 1, 10, 3)
    k2 = st.slider("K2 (buffer entre E2 y E3)", 1, 10, 3)
    seed = st.number_input("Semilla GLC", min_value=0, max_value=2**31 - 1, value=12_345, step=1)
    duration = st.number_input("Duración (min)", min_value=500, max_value=50_000, value=960, step=60)
    n_replicas = st.slider("Número de réplicas", 1, 20, 10)
    run_button = st.button("▶ Simular", type="primary", use_container_width=True)

    with st.expander("ℹ️ Sobre el modelo"):
        st.markdown(
            """
            **Línea de 3 estaciones en serie** con tiempos exponenciales (media 4 min),
            buffers finitos K1 y K2, bloqueo y hambre.

            - **E1** nunca pasa hambre (input infinito).
            - **E3** nunca se bloquea (output infinito).
            - **Bloqueo:** estación termina pero buffer siguiente lleno → espera.
            - **Hambre:** estación libre pero buffer anterior vacío → espera.

            GLC: Numerical Recipes (a=1_664_525, c=1_013_904_223, m=2³²).
            """
        )

# --- Main area ---
st.title("🏭 Simulación: Línea de 3 Estaciones")

tab_analysis, tab_compare, tab_heatmap = st.tabs(
    ["📊 Análisis", "🔀 Comparación", "🔥 Heatmap K1×K2"]
)

# Persist results across reruns
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.config_used = None

if run_button:
    errors = validate_inputs(k1, k2, duration, n_replicas, seed)
    if errors:
        for e in errors:
            st.error(e)
    else:
        with st.spinner("Corriendo réplicas..."):
            st.session_state.results = run_replications(k1, k2, duration, n_replicas, seed)
            st.session_state.config_used = dict(
                k1=k1, k2=k2, duration=duration, n_replicas=n_replicas, seed=seed
            )

with tab_analysis:
    if st.session_state.results is None:
        st.info("Configura los parámetros y pulsa **Simular** en la barra lateral.")
    else:
        st.write("Análisis a poblar en Task 5.2-5.5")

with tab_compare:
    st.write("Comparación a poblar en Task 5.6")

with tab_heatmap:
    st.write("Heatmap a poblar en Task 5.7")
```

- [ ] **Step 2: Run locally**

```bash
streamlit run app.py
```

Expected: browser opens; sidebar inputs visible; clicking Simular shows progress and "Análisis a poblar".

Press Ctrl+C to stop.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): Streamlit bootstrap with sidebar and replication runner"
```

### Task 5.2: Análisis tab — metrics + hypothesis test

**Files:**
- Modify: `app.py` (replace the `with tab_analysis:` block)

- [ ] **Step 1: Replace the `tab_analysis` block**

Find:
```python
with tab_analysis:
    if st.session_state.results is None:
        st.info("Configura los parámetros y pulsa **Simular** en la barra lateral.")
    else:
        st.write("Análisis a poblar en Task 5.2-5.5")
```

Replace with:

```python
def render_metrics_and_hypothesis(results):
    throughputs = [r.throughput for r in results]
    flow_means = [
        sum(r.flow_times) / len(r.flow_times) if r.flow_times else 0.0
        for r in results
    ]
    wips = [r.avg_wip for r in results]
    blocked_e1 = [r.pct_blocked_e1 for r in results]
    blocked_e2 = [r.pct_blocked_e2 for r in results]
    starved_e2 = [r.pct_starved_e2 for r in results]
    starved_e3 = [r.pct_starved_e3 for r in results]

    n = len(throughputs)
    if n >= 2:
        tp_mean, tp_lo, tp_hi = confidence_interval(throughputs)
        ft_mean, ft_lo, ft_hi = confidence_interval(flow_means)
    else:
        tp_mean, tp_lo, tp_hi = throughputs[0], throughputs[0], throughputs[0]
        ft_mean, ft_lo, ft_hi = flow_means[0], flow_means[0], flow_means[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Throughput (piezas/h)",
        f"{tp_mean:.2f}",
        delta=f"IC95 [{tp_lo:.2f}, {tp_hi:.2f}]",
        delta_color="off",
    )
    c2.metric(
        "Tiempo de flujo (min)",
        f"{ft_mean:.2f}",
        delta=f"IC95 [{ft_lo:.2f}, {ft_hi:.2f}]",
        delta_color="off",
    )
    c3.metric("WIP promedio", f"{sum(wips)/n:.2f}")
    total_block_starve = (
        sum(blocked_e1) + sum(blocked_e2) + sum(starved_e2) + sum(starved_e3)
    ) / n
    c4.metric("Bloqueo + Hambre", f"{total_block_starve * 100:.1f}%")

    st.subheader("Prueba de hipótesis  ·  H₁: μ throughput > 12 piezas/h")
    if n >= 2:
        t_stat, p_val = t_test_greater(throughputs, threshold=12.0)
        decision = "🟢 Rechazo H₀" if p_val < 0.05 else "🔴 No rechazo H₀"
        st.write(
            f"t = **{t_stat:.3f}**  ·  p-value = **{p_val:.4g}**  ·  {decision}  (α = 0.05)"
        )
    else:
        st.info("Se necesitan ≥ 2 réplicas para la prueba t.")

    return throughputs, flow_means, wips


with tab_analysis:
    if st.session_state.results is None:
        st.info("Configura los parámetros y pulsa **Simular** en la barra lateral.")
    else:
        results = st.session_state.results
        throughputs, flow_means, wips = render_metrics_and_hypothesis(results)
        st.session_state._cached_throughputs = throughputs
        st.session_state._cached_flow_means = flow_means
```

- [ ] **Step 2: Run locally and click Simular**

```bash
streamlit run app.py
```

Expected: after clicking, you see 4 metric cards and the hypothesis test row.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): metrics and hypothesis test in Análisis tab"
```

### Task 5.3: Análisis tab — WIP evolution + flow time histogram

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Extend the `tab_analysis` block to render plots**

After the `render_metrics_and_hypothesis(results)` call inside `with tab_analysis:`, add:

```python
        # --- Plots ---
        st.subheader("📈 Evolución del WIP")
        rep_idx = st.selectbox(
            "Réplica a visualizar",
            options=list(range(1, len(results) + 1)),
            index=0,
            key="wip_rep_idx",
        )
        st.pyplot(plot_wip_evolution(results[rep_idx - 1].wip_trace))

        st.subheader("📊 Histograma de tiempos de flujo")
        st.pyplot(plot_flow_time_histogram(results[rep_idx - 1].flow_times))

        st.subheader("📐 Throughput por réplica")
        if len(throughputs) >= 2:
            ci = confidence_interval(throughputs)
        else:
            ci = (throughputs[0], throughputs[0], throughputs[0])
        st.pyplot(plot_throughput_bar_with_ci(throughputs, ci=ci))
```

- [ ] **Step 2: Run and verify**

```bash
streamlit run app.py
```

Expected: three new plots appear; changing "Réplica a visualizar" updates the WIP and histogram.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): WIP evolution, flow time histogram, throughput bar"
```

### Task 5.4: Análisis tab — replication table + CSV download

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add table and CSV section**

Append to `with tab_analysis:` (after the throughput bar):

```python
        st.subheader("📋 Resultados por réplica")
        df = pd.DataFrame([
            {
                "Réplica": i + 1,
                "Throughput (piezas/h)": round(r.throughput, 3),
                "Flow time prom. (min)": round(
                    sum(r.flow_times) / len(r.flow_times) if r.flow_times else 0.0, 3
                ),
                "WIP prom.": round(r.avg_wip, 3),
                "% Bloqueo E1": round(r.pct_blocked_e1 * 100, 2),
                "% Bloqueo E2": round(r.pct_blocked_e2 * 100, 2),
                "% Hambre E2": round(r.pct_starved_e2 * 100, 2),
                "% Hambre E3": round(r.pct_starved_e3 * 100, 2),
                "Piezas completadas": r.pieces_completed,
            }
            for i, r in enumerate(results)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Descargar CSV",
            data=csv_bytes,
            file_name=f"resultados_K1{k1}_K2{k2}_seed{seed}.csv",
            mime="text/csv",
        )
```

- [ ] **Step 2: Run and verify table + download**

```bash
streamlit run app.py
```

Expected: table renders, CSV download triggers a file save.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): per-replica table with CSV download"
```

### Task 5.5: Análisis tab — GLC validation expander

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add expander at the bottom of `tab_analysis`**

Append:

```python
        with st.expander("🎲 Validación del GLC"):
            n_u = 1_000
            val_seed = st.session_state.config_used["seed"]
            rng_val = LCG(state=val_seed)
            uniforms = [rng_val.next_uniform() for _ in range(n_u)]
            rng_exp = LCG(state=val_seed)
            ts = [exponential(rng_exp, 0.25) for _ in range(n_u)]

            chi2, crit_chi, pass_chi = chi_square_uniformity(uniforms, k=10)
            d, crit_ks, pass_ks = ks_uniformity(uniforms)
            mean_t = sum(ts) / n_u

            colu1, colu2, colu3 = st.columns(3)
            colu1.metric("χ²", f"{chi2:.2f}", delta=f"crítico {crit_chi:.2f}",
                         delta_color="normal" if pass_chi else "inverse")
            colu2.metric("K-S D", f"{d:.4f}", delta=f"crítico {crit_ks:.4f}",
                         delta_color="normal" if pass_ks else "inverse")
            colu3.metric("Media T exp.", f"{mean_t:.3f} min", delta="esperado 4.00", delta_color="off")

            st.write("Primeros 30 valores Uᵢ generados:")
            st.dataframe(
                pd.DataFrame({"i": range(1, 31), "Uᵢ": uniforms[:30]}),
                use_container_width=True,
                hide_index=True,
            )

            st.write("Histograma de 1000 Uᵢ (debe verse plano):")
            import matplotlib.pyplot as plt
            fig_u, ax_u = plt.subplots(figsize=(9, 2.5))
            ax_u.hist(uniforms, bins=20, edgecolor="black", alpha=0.7)
            ax_u.set_xlabel("U")
            ax_u.set_ylabel("Frecuencia")
            fig_u.tight_layout()
            st.pyplot(fig_u)
```

- [ ] **Step 2: Verify expander**

```bash
streamlit run app.py
```

Expected: expander present, three metrics and a histogram visible when expanded.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): GLC validation expander with chi-square, KS, exp mean"
```

### Task 5.6: Comparación tab

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace `with tab_compare:` block**

Find:
```python
with tab_compare:
    st.write("Comparación a poblar en Task 5.6")
```

Replace with:

```python
with tab_compare:
    st.subheader("🔀 Comparar dos configuraciones (K1, K2)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Configuración A**")
        ka1 = st.slider("K1 (A)", 1, 10, 2, key="ka1")
        ka2 = st.slider("K2 (A)", 1, 10, 2, key="ka2")
    with col_b:
        st.markdown("**Configuración B**")
        kb1 = st.slider("K1 (B)", 1, 10, 5, key="kb1")
        kb2 = st.slider("K2 (B)", 1, 10, 5, key="kb2")

    n_cmp = st.slider("Réplicas por configuración", 1, 20, 5, key="n_cmp")
    cmp_seed = st.number_input(
        "Semilla base", min_value=0, max_value=2**31 - 1, value=42, key="cmp_seed"
    )
    run_compare = st.button("▶ Comparar", key="run_compare")

    if run_compare:
        with st.spinner("Simulando A..."):
            res_a = run_replications(ka1, ka2, 960.0, n_cmp, cmp_seed)
        with st.spinner("Simulando B..."):
            res_b = run_replications(kb1, kb2, 960.0, n_cmp, cmp_seed)
        st.session_state.cmp = (res_a, res_b, ka1, ka2, kb1, kb2)

    if "cmp" in st.session_state:
        res_a, res_b, ka1_, ka2_, kb1_, kb2_ = st.session_state.cmp
        tp_a = [r.throughput for r in res_a]
        tp_b = [r.throughput for r in res_b]
        wip_a = sum(r.avg_wip for r in res_a) / len(res_a)
        wip_b = sum(r.avg_wip for r in res_b) / len(res_b)
        flow_a = [sum(r.flow_times) / len(r.flow_times) if r.flow_times else 0.0 for r in res_a]
        flow_b = [sum(r.flow_times) / len(r.flow_times) if r.flow_times else 0.0 for r in res_b]

        ca, cb = st.columns(2)
        with ca:
            st.markdown(f"**A: K1={ka1_}, K2={ka2_}**")
            st.metric("Throughput", f"{sum(tp_a)/len(tp_a):.2f}")
            st.metric("Flow time", f"{sum(flow_a)/len(flow_a):.2f} min")
            st.metric("WIP", f"{wip_a:.2f}")
        with cb:
            st.markdown(f"**B: K1={kb1_}, K2={kb2_}**")
            st.metric("Throughput", f"{sum(tp_b)/len(tp_b):.2f}")
            st.metric("Flow time", f"{sum(flow_b)/len(flow_b):.2f} min")
            st.metric("WIP", f"{wip_b:.2f}")

        st.subheader("Evolución del WIP — primera réplica de cada config")
        st.pyplot(plot_comparison_wip(
            res_a[0].wip_trace, res_b[0].wip_trace,
            label_a=f"A (K1={ka1_}, K2={ka2_})",
            label_b=f"B (K1={kb1_}, K2={kb2_})",
        ))
```

- [ ] **Step 2: Run and verify**

```bash
streamlit run app.py
```

Expected: switch to Comparación tab, set configs, click Comparar, see two metric columns and the WIP comparison plot.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): comparison tab for two configurations"
```

### Task 5.7: Heatmap tab

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace `with tab_heatmap:` block**

Find:
```python
with tab_heatmap:
    st.write("Heatmap a poblar en Task 5.7")
```

Replace with:

```python
@st.cache_data(show_spinner=False)
def compute_heatmap(seed: int, duration: float, n_per_cell: int) -> dict[tuple[int, int], float]:
    grid: dict[tuple[int, int], float] = {}
    for k1_ in range(1, 11):
        for k2_ in range(1, 11):
            tps = []
            for i in range(n_per_cell):
                cfg = SimulationConfig(
                    k1=k1_, k2=k2_, duration=duration, seed=seed + i * SEED_PRIME
                )
                tps.append(run_simulation(cfg).throughput)
            grid[(k1_, k2_)] = sum(tps) / n_per_cell
    return grid


with tab_heatmap:
    st.subheader("🔥 Throughput promedio por (K1, K2)")
    st.warning(
        "Esto corre 100 combinaciones × N réplicas × duración. "
        "Con N=3 y duración=960 min tarda ~2-4 min en Streamlit Cloud."
    )

    n_per_cell = st.slider("Réplicas por celda", 1, 5, 3, key="hm_n")
    hm_duration = st.number_input(
        "Duración por simulación (min)", min_value=300, max_value=2000, value=960, step=60, key="hm_dur"
    )
    hm_seed = st.number_input(
        "Semilla base", min_value=0, max_value=2**31 - 1, value=42, key="hm_seed"
    )
    run_heatmap = st.button("▶ Calcular heatmap", key="run_heatmap")

    if run_heatmap:
        with st.spinner("Calculando 100 combinaciones..."):
            grid = compute_heatmap(hm_seed, float(hm_duration), n_per_cell)
        st.session_state.heatmap_grid = grid

    if "heatmap_grid" in st.session_state:
        grid = st.session_state.heatmap_grid
        st.pyplot(plot_heatmap(grid, metric_name="Throughput (piezas/h)"))
        best = max(grid.items(), key=lambda kv: kv[1])
        (k1_best, k2_best), val_best = best
        st.success(f"🏆 Óptimo: K1={k1_best}, K2={k2_best} → {val_best:.2f} piezas/h")
```

- [ ] **Step 2: Run and verify (use small N to keep test fast)**

```bash
streamlit run app.py
```

Expected: switch to Heatmap tab, set N=1, click Calcular; ~30-60s later the heatmap appears with the optimum circled.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): heatmap tab over K1×K2 grid with optimum marker"
```

---

## Phase 6 — Excel CLI (`scripts/excel_trace.py`)

### Task 6.1: Implement the CLI

**Files:**
- Create: `scripts/excel_trace.py`

- [ ] **Step 1: Write the script**

```python
# scripts/excel_trace.py
"""CLI: run one replication and export the event trace to .xlsx.

Used to produce the deliverable 2.2 spreadsheet manually.

Example:
    python scripts/excel_trace.py --k1 3 --k2 3 --seed 12345 --duration 480 --out trace.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/excel_trace.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

from src.simulation import SimulationConfig, run_simulation


def export_trace(result, config: SimulationConfig, out: Path) -> None:
    wb = Workbook()

    # --- Sheet 1: Trace ---
    ws = wb.active
    ws.title = "Trace"
    headers = [
        "t (min)", "Tipo", "Pieza", "U usado", "T generado (min)",
        "Estado E1", "Estado E2", "Estado E3",
        "Pieza E1", "Pieza E2", "Pieza E3",
        "B1 cont.", "B2 cont.", "WIP",
        "Piezas terminadas", "Próx. pieza ID", "Comentario",
    ]
    ws.append(headers)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2C3E50")
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill

    for ev in result.event_log:
        ws.append([
            round(ev["t"], 4),
            ev["event_type"],
            ev["piece_id"],
            round(ev["u"], 6) if ev["u"] is not None else "",
            round(ev["duration"], 4) if ev["duration"] is not None else "",
            ev["e1_state"], ev["e2_state"], ev["e3_state"],
            ev["e1_piece"], ev["e2_piece"], ev["e3_piece"],
            ev["b1_count"], ev["b2_count"], ev["wip"],
            ev["pieces_completed"], ev["next_piece_id"], ev["note"],
        ])
    for col in ws.columns:
        max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = max(10, min(max_len + 2, 40))

    # --- Sheet 2: Métricas ---
    ms = wb.create_sheet("Métricas")
    ms.append(["Métrica", "Valor"])
    for cell in ms[1]:
        cell.font = header_font
        cell.fill = header_fill
    rows = [
        ("Piezas completadas", result.pieces_completed),
        ("Throughput (piezas/h)", round(result.throughput, 4)),
        ("Tiempo de flujo promedio (min)",
         round(sum(result.flow_times) / len(result.flow_times), 4) if result.flow_times else 0.0),
        ("WIP promedio", round(result.avg_wip, 4)),
        ("% Bloqueo E1", round(result.pct_blocked_e1 * 100, 3)),
        ("% Bloqueo E2", round(result.pct_blocked_e2 * 100, 3)),
        ("% Hambre E2", round(result.pct_starved_e2 * 100, 3)),
        ("% Hambre E3", round(result.pct_starved_e3 * 100, 3)),
        ("% Ocupado E1", round(result.busy_e1 * 100, 3)),
        ("% Ocupado E2", round(result.busy_e2 * 100, 3)),
        ("% Ocupado E3", round(result.busy_e3 * 100, 3)),
    ]
    for r in rows:
        ms.append(r)
    ms.column_dimensions["A"].width = 35
    ms.column_dimensions["B"].width = 15

    # --- Sheet 3: Inputs ---
    ins = wb.create_sheet("Inputs")
    ins.append(["Parámetro", "Valor"])
    for cell in ins[1]:
        cell.font = header_font
        cell.fill = header_fill
    for k, v in {
        "K1": config.k1,
        "K2": config.k2,
        "Tasa λ (1/min)": config.rate,
        "Media exponencial (min)": 1.0 / config.rate,
        "Duración (min)": config.duration,
        "Semilla GLC": config.seed,
        "GLC a": 1_664_525,
        "GLC c": 1_013_904_223,
        "GLC m": 2 ** 32,
    }.items():
        ins.append([k, v])
    ins.column_dimensions["A"].width = 25
    ins.column_dimensions["B"].width = 20

    wb.save(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export 3-station-line trace to .xlsx")
    parser.add_argument("--k1", type=int, required=True)
    parser.add_argument("--k2", type=int, required=True)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--duration", type=float, default=480.0)
    parser.add_argument("--rate", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("trace.xlsx"))
    args = parser.parse_args()

    cfg = SimulationConfig(
        k1=args.k1, k2=args.k2, rate=args.rate, duration=args.duration, seed=args.seed
    )
    result = run_simulation(cfg)
    export_trace(result, cfg, args.out)
    print(f"Wrote {args.out} — {result.pieces_completed} pieces, "
          f"{result.throughput:.2f} piezas/h, {len(result.event_log)} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the CLI**

```bash
python scripts/excel_trace.py --k1 3 --k2 3 --seed 12345 --duration 480 --out /tmp/trace_test.xlsx
```

Expected: `Wrote /tmp/trace_test.xlsx — NN pieces, X.XX piezas/h, MMM events`.

- [ ] **Step 3: Open and visually verify**

```bash
open /tmp/trace_test.xlsx
```

Expected: 3 sheets (Trace, Métricas, Inputs), populated correctly.

- [ ] **Step 4: Commit**

```bash
git add scripts/excel_trace.py
git commit -m "feat(cli): export event trace to .xlsx for deliverable 2.2"
```

---

## Phase 7 — CI, deployment, README

### Task 7.1: GitHub Actions CI

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/test.yml
name: tests
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run pytest
        run: pytest tests/ -v
```

- [ ] **Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/test.yml
git commit -m "ci: pytest on push and PR"
```

### Task 7.2: Streamlit theme config

**Files:**
- Create: `.streamlit/config.toml`

- [ ] **Step 1: Write config**

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 10
enableCORS = false

[browser]
gatherUsageStats = false
```

- [ ] **Step 2: Commit**

```bash
mkdir -p .streamlit
git add .streamlit/config.toml
git commit -m "chore: streamlit theme config"
```

### Task 7.3: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# Línea de 3 Estaciones — Simulación

Mini Reto III · Simulación, Análisis y Diseño · EPN.

App web interactiva que simula una línea de ensamblaje de 3 estaciones en serie
con buffers finitos, tiempos exponenciales, bloqueo y hambre.

## App desplegada

→ **https://3-station-line.streamlit.app**

> Si la app duerme tras inactividad, el primer load toma ~30-60s.

## Cómo correr local

```bash
git clone <repo-url>
cd 3-station-line
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## Generar el Excel del entregable 2.2

```bash
python scripts/excel_trace.py --k1 3 --k2 3 --seed 12345 --duration 480 --out trace.xlsx
```

## Estructura

```
app.py                 # Streamlit UI
src/rng.py             # Linear Congruential Generator + exponencial
src/simulation.py      # Discrete event simulation (FEL + manejadores)
src/statistics.py      # IC, t-test, χ², K-S
src/plotting.py        # Figuras matplotlib
scripts/excel_trace.py # CLI para entregable 2.2
tests/                 # pytest
```

## Decisiones técnicas

- **GLC**: parámetros Numerical Recipes (a=1_664_525, c=1_013_904_223, m=2³²),
  verificados con Hull-Dobell (período = 2³²).
- **Tiempos exponenciales**: transformada inversa T = -ln(U) / λ con λ = 0.25.
- **DES**: Future Event List con `heapq`. Prioridad downstream-first
  (FIN_E3 > FIN_E2 > FIN_E1) para eventos simultáneos.
- **Bloqueo/hambre**: programados a mano (sin SimPy) para defenderlos en
  sustentación.

## Uso de IA

Esta sección documenta el uso de herramientas de IA durante el desarrollo:

- [ ] Listar herramientas (ChatGPT, Claude, Copilot, etc.)
- [ ] Listar tareas concretas en las que se usó
- [ ] Reflexión: qué funcionó, qué no, ejemplos de outputs corregidos

## Autores

- [Nombre 1]
- [Nombre 2]
- ...
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with run instructions and design rationale"
```

### Task 7.4: Push to GitHub and deploy

**Files:** none (manual steps)

- [ ] **Step 1: Create public GitHub repo**

Go to https://github.com/new
- Name: `3-station-line`
- Visibility: **Public** (required by Streamlit Community Cloud)
- Do NOT initialize with README (we have one)

- [ ] **Step 2: Push**

```bash
git remote add origin https://github.com/<your-username>/3-station-line.git
git push -u origin main
```

Expected: branch tracks remote.

- [ ] **Step 3: Verify CI green**

Visit `https://github.com/<your-username>/3-station-line/actions`.

Expected: the `tests` workflow runs and finishes green within ~3 min.

- [ ] **Step 4: Deploy to Streamlit Community Cloud**

1. Visit https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Repository: `<your-username>/3-station-line`, branch: `main`, file: `app.py`
5. (Optional) Custom URL: `3-station-line`
6. Click "Deploy"

Expected: ~2-3 min to first deploy. App becomes available at `https://3-station-line.streamlit.app` (or your chosen URL).

- [ ] **Step 5: Smoke test the deployed app**

Open the URL in a private/incognito window:
- [ ] Sidebar inputs visible
- [ ] Click Simular: progress bar runs, metrics appear
- [ ] Switch to Comparación tab: works
- [ ] Switch to Heatmap tab: works with small N

- [ ] **Step 6: Update README with the actual URL**

If your URL differs from `3-station-line.streamlit.app`, edit `README.md` and push:

```bash
git add README.md
git commit -m "docs: update deployed app URL"
git push
```

---

## Final verification checklist

- [ ] `pytest tests/ -v` all green locally
- [ ] CI badge green on GitHub
- [ ] `streamlit run app.py` works without errors
- [ ] All 3 tabs functional (Análisis, Comparación, Heatmap)
- [ ] GLC validation expander shows χ² and K-S passing
- [ ] `python scripts/excel_trace.py --k1 3 --k2 3` produces valid xlsx
- [ ] Deployed app URL works in a clean browser
- [ ] README references the actual deployed URL
- [ ] "Uso de IA" section in README filled out
