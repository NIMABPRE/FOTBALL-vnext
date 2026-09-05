"""
Kelly staking.

Single-bet Kelly
-----------------
For a bet at decimal odds `d` (net odds b = d - 1) with true win probability
`p`, the Kelly criterion maximizing long-run log growth is:

    f* = (b*p - q) / b,   q = 1 - p

Full Kelly is provably growth-optimal but has brutal variance; in practice
everyone uses Fractional Kelly (f = kelly_fraction * f*, typically 0.25-0.5)
to trade some growth rate for a much shallower drawdown profile, which also
buys a margin of safety against probability-estimation error.

Portfolio-level adjustment for simultaneous bets
--------------------------------------------------
Single-bet Kelly assumes each bet's outcome is independent of every other
bet you place at the same time. That assumption breaks down for bets on the
same match (e.g. match result + total goals) or across matches that share a
common factor (e.g. the same team appearing in two markets, or systematic
model error correlated across a full slate). Treating simultaneously-placed
correlated bets as independent understates the portfolio's true risk and
leads to overexposure.

This module implements a practical (not textbook-exact) correlation
adjustment: it scales the whole vector of naive per-bet Kelly stakes down by
a single factor so that the portfolio's variance proxy (f^T Sigma f, with
Sigma built from a correlation matrix and each bet's Bernoulli variance)
does not exceed the variance of a single average-sized independent bet. This
is a conservative heuristic, not a full joint-distribution multivariate
Kelly solve (which requires modeling the joint outcome distribution of all
bets together) — but it directly fixes the overexposure problem of naive
independent Kelly on correlated bets, and is transparent about being an
approximation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from football_vnext.domain.value.edge import ValueBetSignal

logger = logging.getLogger(__name__)


class KellyError(Exception):
    """Raised on invalid Kelly staking inputs."""


@dataclass(frozen=True)
class StakeRecommendation:
    """
    A final staking recommendation for one value-bet signal, in "units"
    (fraction of bankroll), both before and after the portfolio-level
    correlation adjustment.
    """

    signal: ValueBetSignal
    naive_kelly_fraction: float
    adjusted_kelly_fraction: float
    stake_fraction_of_bankroll: float


def single_bet_kelly_fraction(probability: float, net_odds: float) -> float:
    """
    Full-Kelly fraction for one bet. Returns 0.0 (never negative) when the
    bet has no edge at these odds — Kelly should never recommend staking on
    a bet with non-positive expected value.
    """
    if not (0.0 <= probability <= 1.0):
        raise KellyError(f"probability must be in [0,1], got {probability}")
    if net_odds <= 0:
        raise KellyError(f"net_odds must be > 0, got {net_odds}")

    q = 1.0 - probability
    f_star = (net_odds * probability - q) / net_odds
    return max(f_star, 0.0)


class PortfolioKellyStaker:
    """
    Computes fractional Kelly stakes for a batch of simultaneous value-bet
    signals, applying a correlation-aware scale-down when multiple signals
    are correlated (by default: signals sharing the same match_id are
    treated as fully correlated; everything else is treated as independent
    unless an explicit correlation matrix is supplied).
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_stake_per_bet: float = 0.05,
        max_total_exposure: float = 0.25,
    ) -> None:
        if not (0.0 < kelly_fraction <= 1.0):
            raise KellyError("kelly_fraction must be in (0, 1]")
        if not (0.0 < max_stake_per_bet <= 1.0):
            raise KellyError("max_stake_per_bet must be in (0, 1]")
        if not (0.0 < max_total_exposure <= 1.0):
            raise KellyError("max_total_exposure must be in (0, 1]")
        self.kelly_fraction = kelly_fraction
        self.max_stake_per_bet = max_stake_per_bet
        self.max_total_exposure = max_total_exposure

    def _build_correlation_matrix(self, signals: Sequence[ValueBetSignal]) -> np.ndarray:
        n = len(signals)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                same_match = signals[i].candidate.match_id == signals[j].candidate.match_id
                rho = 1.0 if same_match else 0.0
                corr[i, j] = corr[j, i] = rho
        return corr

    def compute_stakes(
        self,
        signals: Sequence[ValueBetSignal],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> List[StakeRecommendation]:
        if not signals:
            return []

        n = len(signals)
        if correlation_matrix is None:
            correlation_matrix = self._build_correlation_matrix(signals)
        elif correlation_matrix.shape != (n, n):
            raise KellyError(
                f"correlation_matrix shape {correlation_matrix.shape} does not match "
                f"{n} signals."
            )

        naive_fractions = np.array(
            [
                single_bet_kelly_fraction(s.candidate.calibrated_prob, s.candidate.net_odds)
                * self.kelly_fraction
                for s in signals
            ]
        )
        # Cap individual stakes before the portfolio adjustment, so one huge
        # single-bet Kelly number can't dominate the correlation scaling.
        naive_fractions = np.minimum(naive_fractions, self.max_stake_per_bet)

        # Bernoulli variance of each bet's *probability* (not its payout) as
        # a simple proxy for how much each stake contributes to portfolio risk.
        variances = np.array(
            [s.candidate.calibrated_prob * (1 - s.candidate.calibrated_prob) for s in signals]
        )
        sigma = np.sqrt(variances)

        covariance = correlation_matrix * np.outer(sigma, sigma)
        naive_portfolio_variance = float(naive_fractions @ covariance @ naive_fractions)

        # Reference risk budget: the variance of a single bet at the average
        # stake size and average variance, i.e. "as risky as placing one
        # typical independent bet".
        avg_fraction = float(naive_fractions.mean())
        avg_variance = float(variances.mean())
        reference_variance = (avg_fraction**2) * avg_variance

        if naive_portfolio_variance <= reference_variance or naive_portfolio_variance <= 0:
            scale = 1.0
        else:
            scale = float(np.sqrt(reference_variance / naive_portfolio_variance))

        adjusted_fractions = naive_fractions * scale

        total = float(adjusted_fractions.sum())
        if total > self.max_total_exposure and total > 0:
            adjusted_fractions = adjusted_fractions * (self.max_total_exposure / total)

        logger.info(
            "Portfolio Kelly: %d signals, correlation scale=%.4f, total exposure=%.4f",
            n, scale, float(adjusted_fractions.sum()),
        )

        return [
            StakeRecommendation(
                signal=signals[i],
                naive_kelly_fraction=float(naive_fractions[i]),
                adjusted_kelly_fraction=float(adjusted_fractions[i] / scale) if scale > 0 else 0.0,
                stake_fraction_of_bankroll=float(adjusted_fractions[i]),
            )
            for i in range(n)
        ]
