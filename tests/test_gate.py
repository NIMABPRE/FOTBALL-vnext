from __future__ import annotations

import pytest

from football_vnext.domain.backtest.clv import CLVSummary
from football_vnext.domain.backtest.gate import BacktestGate, BacktestGateConfig, GateError
from football_vnext.domain.backtest.metrics import BacktestMetrics


def _metrics(n_bets=500, roi=0.05, max_drawdown=0.15, p_value=0.01) -> BacktestMetrics:
    return BacktestMetrics(
        n_bets=n_bets, starting_bankroll=1.0, ending_bankroll=1.0 + roi,
        total_pnl=roi, roi=roi, max_drawdown=max_drawdown, sharpe_like_ratio=0.1,
        win_rate=0.5, t_statistic=3.0, p_value=p_value, returns_significantly_positive=(p_value < 0.05),
    )


def _clv(mean_clv=0.03, p_value=0.01, significant=True) -> CLVSummary:
    return CLVSummary(
        n_bets=500, mean_clv=mean_clv, pct_positive_clv=0.6,
        t_statistic=3.0, p_value=p_value, clv_significantly_positive=significant,
    )


def test_passes_when_all_criteria_met():
    gate = BacktestGate(BacktestGateConfig(min_bets=300))
    result = gate.evaluate(_metrics(), _clv())
    assert result.passed is True
    assert result.reasons_for_failure == []


def test_fails_on_too_few_bets():
    gate = BacktestGate(BacktestGateConfig(min_bets=300))
    result = gate.evaluate(_metrics(n_bets=50), _clv())
    assert result.passed is False
    assert any("bets" in r for r in result.reasons_for_failure)


def test_fails_on_insignificant_clv():
    gate = BacktestGate(BacktestGateConfig(min_bets=300))
    result = gate.evaluate(_metrics(), _clv(significant=False))
    assert result.passed is False
    assert any("CLV" in r for r in result.reasons_for_failure)


def test_fails_on_negative_roi():
    gate = BacktestGate(BacktestGateConfig(min_bets=300))
    result = gate.evaluate(_metrics(roi=-0.02), _clv())
    assert result.passed is False
    assert any("ROI" in r for r in result.reasons_for_failure)


def test_fails_on_excessive_drawdown():
    gate = BacktestGate(BacktestGateConfig(min_bets=300, max_acceptable_drawdown=0.20))
    result = gate.evaluate(_metrics(max_drawdown=0.35), _clv())
    assert result.passed is False
    assert any("drawdown" in r for r in result.reasons_for_failure)


def test_multiple_failures_all_reported():
    gate = BacktestGate(BacktestGateConfig(min_bets=300))
    result = gate.evaluate(_metrics(n_bets=10, roi=-0.1), _clv(significant=False))
    assert result.passed is False
    assert len(result.reasons_for_failure) >= 3


def test_rejects_invalid_config():
    with pytest.raises(GateError):
        BacktestGateConfig(min_bets=0)
    with pytest.raises(GateError):
        BacktestGateConfig(max_acceptable_drawdown=0.0)
