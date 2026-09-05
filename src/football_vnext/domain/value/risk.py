"""
Risk Score.

Combines three independent sources of doubt about a value-bet signal into a
single score in [0, 1] (higher = riskier / less trustworthy), each capturing
a different failure mode that Edge/EV alone cannot see:

1. Bookmaker disagreement — if multiple bookmakers quote very different odds
   for the same outcome, at least one of them disagrees sharply with the
   emerging consensus, meaning the "fair" price used to compute Edge/EV is
   itself uncertain. Measured as the coefficient of variation (std/mean) of
   the raw implied probabilities across bookmakers.

2. Data quality — Dixon-Coles attack/defence parameters fit on very few
   historical matches for either team are unreliable estimates. Measured
   against a configurable minimum reliable sample size.

3. Market efficiency — see market_efficiency.py. Highly efficient markets
   make a detected edge more likely to be spurious.

This score is used two ways downstream: (a) as a hard filter (reject above
a threshold) and (b) as a stake dampener applied on top of Kelly sizing
(risk-adjusted stake = kelly_stake * (1 - risk_score)), composed at the
application level rather than inside kelly.py, to keep Kelly's math pure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from football_vnext.domain.odds.models import BookmakerOdds
from football_vnext.domain.value.edge import Outcome
from football_vnext.domain.value.market_efficiency import (
    MarketEfficiencyClassifier,
    MarketEfficiencyTier,
)

logger = logging.getLogger(__name__)


class RiskScoreError(Exception):
    """Raised on invalid risk-score inputs."""


@dataclass(frozen=True)
class RiskAssessment:
    bookmaker_disagreement: float
    data_quality_risk: float
    market_efficiency_risk: float
    market_efficiency_tier: MarketEfficiencyTier
    risk_score: float
    is_acceptable: bool


class RiskScoreCalculator:
    """
    :param min_reliable_sample_size: number of historical matches a team
        needs to have played (in the fitting window) for its attack/defence
        estimate to be considered fully reliable (data_quality_risk -> 0).
    :param max_acceptable_risk_score: signals with a computed risk_score
        above this are flagged as not acceptable.
    :param weights: relative weight of (bookmaker_disagreement,
        data_quality_risk, market_efficiency_risk) in the combined score.
        Must sum to 1.0.
    """

    def __init__(
        self,
        min_reliable_sample_size: int = 20,
        max_acceptable_risk_score: float = 0.6,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        efficiency_classifier: Optional[MarketEfficiencyClassifier] = None,
    ) -> None:
        if min_reliable_sample_size < 1:
            raise RiskScoreError("min_reliable_sample_size must be >= 1")
        if not (0.0 < max_acceptable_risk_score <= 1.0):
            raise RiskScoreError("max_acceptable_risk_score must be in (0, 1]")
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise RiskScoreError(f"weights must sum to 1.0, got {sum(weights)}")
        self.min_reliable_sample_size = min_reliable_sample_size
        self.max_acceptable_risk_score = max_acceptable_risk_score
        self.weights = weights
        self.efficiency_classifier = efficiency_classifier or MarketEfficiencyClassifier()

    def _bookmaker_disagreement(self, quotes: Sequence[BookmakerOdds], outcome: Outcome) -> float:
        if len(quotes) < 2:
            # Cannot cross-check a single quote -- treat as moderate,
            # non-zero risk rather than falsely reporting perfect agreement.
            return 0.5

        implied = np.array([1.0 / getattr(q, outcome.value) for q in quotes])
        mean = float(implied.mean())
        if mean <= 0:
            raise RiskScoreError("Mean implied probability was non-positive.")
        cv = float(implied.std() / mean)
        # A coefficient of variation of 0.1 (10%) or more across bookmakers
        # on the same outcome is already a lot of disagreement in a liquid
        # market; treat that as maximal risk contribution from this factor.
        return float(np.clip(cv / 0.10, 0.0, 1.0))

    def _data_quality_risk(self, home_team_matches: int, away_team_matches: int) -> float:
        if home_team_matches < 0 or away_team_matches < 0:
            raise RiskScoreError("Team match counts cannot be negative.")
        weakest = min(home_team_matches, away_team_matches)
        return float(np.clip(1.0 - weakest / self.min_reliable_sample_size, 0.0, 1.0))

    def compute(
        self,
        bookmaker_quotes: Sequence[BookmakerOdds],
        outcome: Outcome,
        home_team_matches: int,
        away_team_matches: int,
        competition: str,
    ) -> RiskAssessment:
        disagreement = self._bookmaker_disagreement(bookmaker_quotes, outcome)
        data_quality_risk = self._data_quality_risk(home_team_matches, away_team_matches)

        tier = self.efficiency_classifier.classify(competition)
        efficiency_risk = self.efficiency_classifier.risk_contribution(tier)

        w_disagree, w_quality, w_efficiency = self.weights
        risk_score = (
            w_disagree * disagreement + w_quality * data_quality_risk + w_efficiency * efficiency_risk
        )
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        assessment = RiskAssessment(
            bookmaker_disagreement=disagreement,
            data_quality_risk=data_quality_risk,
            market_efficiency_risk=efficiency_risk,
            market_efficiency_tier=tier,
            risk_score=risk_score,
            is_acceptable=risk_score <= self.max_acceptable_risk_score,
        )
        logger.info(
            "Risk assessment: score=%.3f (disagreement=%.2f, data_quality=%.2f, "
            "efficiency=%.2f/%s) -> acceptable=%s",
            risk_score, disagreement, data_quality_risk, efficiency_risk, tier.value,
            assessment.is_acceptable,
        )
        return assessment
