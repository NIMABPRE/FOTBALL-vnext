"""
Market Efficiency Filter.

Bookmaker markets are not equally efficient. Top European leagues attract
enormous liquidity and sharp bettors, so their closing lines are very hard
to beat — a detected "edge" there is more likely to be model noise, stale
data, or a mispriced probability than a genuine, exploitable inefficiency.
Lower-tier / less-covered leagues get far less scrutiny from bookmakers and
sharp money, so genuine mispricing is more plausible there — though (handled
separately, in risk.py) they usually also come with worse data quality.

This module does not reject bets outright; it returns a multiplier that the
Edge/EV filtering stage should apply to its thresholds, i.e. requiring MORE
edge/EV to accept a bet in a highly efficient market, and allowing a bet to
clear the bar more easily in a less efficient one.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet


class MarketEfficiencyTier(str, Enum):
    HIGH = "high_efficiency"
    MEDIUM = "medium_efficiency"
    LOW = "low_efficiency"


# NOTE: illustrative starter registry, not exhaustive. Extend as needed once
# real competition data is wired in (Phase 1.B / 2.B).
_HIGH_EFFICIENCY_LEAGUES: FrozenSet[str] = frozenset(
    {
        "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
        "champions league", "europa league",
    }
)
_LOW_EFFICIENCY_LEAGUES: FrozenSet[str] = frozenset(
    {
        "sample league",  # matches the synthetic sample_data.py competition name
    }
)

_REQUIRED_EDGE_MULTIPLIER: Dict[MarketEfficiencyTier, float] = {
    MarketEfficiencyTier.HIGH: 1.5,
    MarketEfficiencyTier.MEDIUM: 1.0,
    MarketEfficiencyTier.LOW: 0.8,
}

_RISK_CONTRIBUTION: Dict[MarketEfficiencyTier, float] = {
    # Higher = more likely that a detected edge in this tier is spurious.
    MarketEfficiencyTier.HIGH: 0.8,
    MarketEfficiencyTier.MEDIUM: 0.4,
    MarketEfficiencyTier.LOW: 0.2,
}


class MarketEfficiencyClassifier:
    """
    Classifies a competition name into an efficiency tier. Unknown
    competitions default to MEDIUM (neither penalized nor favored) rather
    than silently assumed inefficient, which would make the filter useless.
    """

    def classify(self, competition: str) -> MarketEfficiencyTier:
        key = competition.strip().lower()
        if key in _HIGH_EFFICIENCY_LEAGUES:
            return MarketEfficiencyTier.HIGH
        if key in _LOW_EFFICIENCY_LEAGUES:
            return MarketEfficiencyTier.LOW
        return MarketEfficiencyTier.MEDIUM

    def required_edge_multiplier(self, tier: MarketEfficiencyTier) -> float:
        """
        Multiply your base min_edge / min_ev thresholds by this value for
        the given tier before running EdgeDetector — e.g. a 1.5x multiplier
        on a 3% base min_edge means 4.5% is actually required in HIGH
        efficiency markets.
        """
        return _REQUIRED_EDGE_MULTIPLIER[tier]

    def risk_contribution(self, tier: MarketEfficiencyTier) -> float:
        """A value in [0,1] feeding into the overall Risk Score (risk.py)."""
        return _RISK_CONTRIBUTION[tier]
