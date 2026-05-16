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
