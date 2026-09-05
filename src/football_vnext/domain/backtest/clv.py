"""
Closing Line Value (CLV) tracking.

CLV is the single most important number this whole system produces. Raw
backtest ROI can be — and very often is, given football's variance — mostly
noise even over a few hundred bets. CLV asks a sharper, more falsifiable
question: at the moment you placed each bet, did you consistently get a
better price than the market eventually settled on? If yes, over enough bets,
that is strong evidence of genuine, exploitable edge, independent of whether
variance happened to go your way this particular backtest run.

    CLV = (odds_taken / closing_odds) - 1

Positive CLV means you bet at odds better than the closing line implies you
"should" have gotten -- i.e., the market moved in the direction your bet
predicted, after you had already bet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from football_vnext.domain.value.edge import Outcome


class CLVError(Exception):
    """Raised on invalid CLV inputs."""


@dataclass(frozen=True)
class BetRecord:
    """One simulated bet placed during a backtest, with its eventual result."""

    match_id: str
    outcome: Outcome
    odds_taken: float
    closing_odds: float
    stake_fraction: float  # fraction of bankroll staked at the time
    stake_amount: float  # absolute currency amount staked (post-compounding)
    won: bool
    pnl: float  # profit (positive) or loss (negative), in the same currency unit as stake_amount

    def __post_init__(self) -> None:
        if self.odds_taken <= 1.0 or self.closing_odds <= 1.0:
            raise CLVError("odds must be > 1.0")
        if self.stake_fraction < 0 or self.stake_amount < 0:
            raise CLVError("stake values cannot be negative")

    @property
    def clv(self) -> float:
        return (self.odds_taken / self.closing_odds) - 1.0


@dataclass(frozen=True)
class CLVSummary:
    n_bets: int
    mean_clv: float
    pct_positive_clv: float
    t_statistic: float
    p_value: float
    clv_significantly_positive: bool

    def __str__(self) -> str:
        return (
            f"CLVSummary(n={self.n_bets}, mean_clv={self.mean_clv:+.2%}, "
            f"pct_positive={self.pct_positive_clv:.1%}, p_value={self.p_value:.4f}, "
            f"significant={self.clv_significantly_positive})"
        )


class CLVAnalyzer:
    def __init__(self, significance_level: float = 0.05) -> None:
        if not (0.0 < significance_level < 1.0):
            raise CLVError("significance_level must be in (0, 1)")
        self.significance_level = significance_level

    def summarize(self, bets: Sequence[BetRecord]) -> CLVSummary:
        if not bets:
            raise CLVError("Cannot summarize CLV for an empty set of bets.")

        clv_values = np.array([b.clv for b in bets])
        mean_clv = float(clv_values.mean())
        pct_positive = float(np.mean(clv_values > 0))

        if len(bets) < 2:
            t_stat, p_value = 0.0, 1.0
        elif np.isclose(clv_values.std(), 0.0):
            # Zero variance: every bet had identical CLV. If that shared
            # value is positive, this is maximally (not zero) significant;
            # if it's non-positive, there's no evidence of edge at all.
            t_stat, p_value = (float("inf"), 0.0) if mean_clv > 0 else (0.0, 1.0)
        else:
            t_stat, p_value = stats.ttest_1samp(clv_values, popmean=0.0, alternative="greater")
            t_stat, p_value = float(t_stat), float(p_value)

        return CLVSummary(
            n_bets=len(bets),
            mean_clv=mean_clv,
            pct_positive_clv=pct_positive,
            t_statistic=t_stat,
            p_value=p_value,
            clv_significantly_positive=(p_value < self.significance_level and mean_clv > 0),
        )
