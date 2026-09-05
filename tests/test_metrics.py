from __future__ import annotations

import pytest

from football_vnext.domain.backtest.clv import BetRecord
from football_vnext.domain.backtest.metrics import BacktestMetricsCalculator, MetricsError
from football_vnext.domain.value.edge import Outcome


def _bet(won: bool, stake: float = 1.0, odds: float = 2.0) -> BetRecord:
    pnl = stake * (odds - 1.0) if won else -stake
    return BetRecord(
        match_id="M", outcome=Outcome.HOME, odds_taken=odds, closing_odds=odds,
        stake_fraction=0.02, stake_amount=stake, won=won, pnl=pnl,
    )


def test_compute_raises_on_empty_bets():
    with pytest.raises(MetricsError):
        BacktestMetricsCalculator().compute([], [1.0, 1.0], 1.0)


def test_compute_raises_on_short_equity_curve():
    with pytest.raises(MetricsError):
        BacktestMetricsCalculator().compute([_bet(True)], [1.0], 1.0)


def test_roi_known_value():
    # 2 wins at odds 2.0 (pnl=+1 each), 1 loss (pnl=-1) => total_pnl=1, staked=3 => roi=1/3
    bets = [_bet(True), _bet(True), _bet(False)]
    metrics = BacktestMetricsCalculator().compute(bets, [1.0, 2.0, 3.0, 2.0], 1.0)
    assert metrics.roi == pytest.approx(1 / 3)
    assert metrics.win_rate == pytest.approx(2 / 3)


def test_max_drawdown_known_value():
    calc = BacktestMetricsCalculator()
    # equity goes 1.0 -> 2.0 (peak) -> 1.0 (50% drawdown from peak) -> 1.5
    metrics = calc.compute([_bet(True)], [1.0, 2.0, 1.0, 1.5], 1.0)
    assert metrics.max_drawdown == pytest.approx(0.5)


def test_all_wins_gives_significant_positive_returns():
    bets = [_bet(True) for _ in range(30)]
    equity = [1.0] + [1.0 + i for i in range(1, 31)]
    metrics = BacktestMetricsCalculator().compute(bets, equity, 1.0)
    assert metrics.returns_significantly_positive is True


def test_rejects_invalid_significance_level():
    with pytest.raises(MetricsError):
        BacktestMetricsCalculator(significance_level=0.0)
