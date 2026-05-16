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
