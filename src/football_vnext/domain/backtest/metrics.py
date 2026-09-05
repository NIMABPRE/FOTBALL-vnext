"""
Backtest performance metrics.

Deliberately separate from CLV (clv.py) -- ROI/drawdown/Sharpe describe how
THIS PARTICULAR backtest run went, which is heavily influenced by variance
over a finite sample. CLV is the more trustworthy signal of genuine edge.
Both are needed for the final gate (gate.py): good ROI with bad/insignificant
CLV is a yellow flag (could be luck); good CLV with mediocre ROI in a short
backtest is still meaningful evidence of edge that hasn't fully paid off yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from football_vnext.domain.backtest.clv import BetRecord


class MetricsError(Exception):
    """Raised on invalid metrics inputs."""


@dataclass(frozen=True)
class BacktestMetrics:
    n_bets: int
    starting_bankroll: float
    ending_bankroll: float
    total_pnl: float
    roi: float  # total_pnl / total_amount_staked
    max_drawdown: float  # as a fraction of peak bankroll, e.g. 0.25 = 25% drawdown
    sharpe_like_ratio: float  # mean(per-bet return) / std(per-bet return)
    win_rate: float
    t_statistic: float
    p_value: float
    returns_significantly_positive: bool

    def __str__(self) -> str:
        return (
            f"BacktestMetrics(n={self.n_bets}, roi={self.roi:+.2%}, "
            f"max_drawdown={self.max_drawdown:.2%}, sharpe={self.sharpe_like_ratio:.3f}, "
            f"win_rate={self.win_rate:.1%}, p_value={self.p_value:.4f}, "
            f"significant={self.returns_significantly_positive})"
        )


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(curve)
    drawdowns = (running_max - curve) / running_max
    return float(np.max(drawdowns)) if len(drawdowns) else 0.0


class BacktestMetricsCalculator:
    def __init__(self, significance_level: float = 0.05) -> None:
        if not (0.0 < significance_level < 1.0):
            raise MetricsError("significance_level must be in (0, 1)")
        self.significance_level = significance_level

    def compute(
        self, bets: Sequence[BetRecord], equity_curve: Sequence[float], starting_bankroll: float,
    ) -> BacktestMetrics:
        if not bets:
            raise MetricsError("Cannot compute metrics for an empty set of bets.")
        if len(equity_curve) < 2:
            raise MetricsError("equity_curve must have at least a starting and ending value.")
        if starting_bankroll <= 0:
            raise MetricsError("starting_bankroll must be > 0")

        total_staked = sum(b.stake_amount for b in bets)
        total_pnl = sum(b.pnl for b in bets)
        roi = total_pnl / total_staked if total_staked > 0 else 0.0

        per_bet_returns = np.array(
            [b.pnl / b.stake_amount if b.stake_amount > 0 else 0.0 for b in bets]
        )
        mean_return = float(per_bet_returns.mean())
        std_return = float(per_bet_returns.std())
        sharpe_like = mean_return / std_return if std_return > 0 else 0.0

        if len(bets) < 2:
            t_stat, p_value = 0.0, 1.0
        elif std_return == 0:
            t_stat, p_value = (float("inf"), 0.0) if mean_return > 0 else (0.0, 1.0)
        else:
            t_stat, p_value = stats.ttest_1samp(per_bet_returns, popmean=0.0, alternative="greater")
            t_stat, p_value = float(t_stat), float(p_value)

        win_rate = float(np.mean([b.won for b in bets]))

        return BacktestMetrics(
            n_bets=len(bets),
            starting_bankroll=starting_bankroll,
            ending_bankroll=float(equity_curve[-1]),
            total_pnl=total_pnl,
            roi=roi,
            max_drawdown=_max_drawdown(equity_curve),
            sharpe_like_ratio=sharpe_like,
            win_rate=win_rate,
            t_statistic=t_stat,
            p_value=p_value,
            returns_significantly_positive=(p_value < self.significance_level and mean_return > 0),
        )
