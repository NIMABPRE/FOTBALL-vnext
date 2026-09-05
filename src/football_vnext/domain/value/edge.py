"""
Edge / Expected Value detection.

Two related but distinct quantities, kept separate deliberately (conflating
them is a common mistake):

    Edge = calibrated_model_probability - fair_market_probability
        -> how much your model disagrees with the (de-vigged) market.
           This can be nonzero even at odds that pay out at breakeven or worse.

    EV   = calibrated_model_probability * decimal_odds - 1
        -> the actual expected return per unit staked AT THE ODDS ON OFFER.
           This is what your money actually cares about — a bet can have a
           positive Edge but a small or negative EV if the offered odds are
           worse than the fair odds implied by the market itself (e.g. a
           different, worse-priced bookmaker).

Both are required as filters: Edge alone can be a statistical artifact of
calibration noise; EV alone doesn't tell you whether the "edge" behind it is
believable or just bookmaker-to-bookmaker price variance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

from football_vnext.domain.models.probability import OutcomeProbabilities

logger = logging.getLogger(__name__)


class Outcome(str, Enum):
    HOME = "home"
    DRAW = "draw"
    AWAY = "away"


class ValueDetectionError(Exception):
    """Raised when a value-bet candidate is malformed."""


@dataclass(frozen=True)
class ValueBetCandidate:
    """
    One specific betting opportunity: a single outcome of a single match, at
    a specific price, with the model's calibrated probability and the fair
    (de-vigged) market probability for that same outcome already computed.
    """

    match_id: str
    outcome: Outcome
    decimal_odds: float
    calibrated_prob: float
    fair_market_prob: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.decimal_odds <= 1.0:
            raise ValueDetectionError(f"decimal_odds must be > 1.0, got {self.decimal_odds}")
        if not (0.0 <= self.calibrated_prob <= 1.0):
            raise ValueDetectionError(
                f"calibrated_prob out of [0,1]: {self.calibrated_prob}"
            )
        if not (0.0 <= self.fair_market_prob <= 1.0):
            raise ValueDetectionError(
                f"fair_market_prob out of [0,1]: {self.fair_market_prob}"
            )

    @property
    def edge(self) -> float:
        return self.calibrated_prob - self.fair_market_prob

    @property
    def expected_value(self) -> float:
        return self.calibrated_prob * self.decimal_odds - 1.0

    @property
    def net_odds(self) -> float:
        """Decimal odds minus 1 (the 'b' in Kelly's f = (bp - q) / b)."""
        return self.decimal_odds - 1.0


@dataclass(frozen=True)
class ValueBetSignal:
    """A candidate that has passed the Edge and EV filters, ready for staking."""

    candidate: ValueBetCandidate
    edge: float
    expected_value: float


def build_candidates_from_predictions(
    match_id: str,
    calibrated_probs: OutcomeProbabilities,
    fair_market_probs: OutcomeProbabilities,
    *,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> List[ValueBetCandidate]:
    """
    Convenience constructor: builds one ValueBetCandidate per outcome (home,
    draw, away) for a single match from already-computed calibrated and
    fair-market probability vectors, plus the actual prices on offer.
    """
    return [
        ValueBetCandidate(
            match_id=match_id, outcome=Outcome.HOME, decimal_odds=home_odds,
            calibrated_prob=calibrated_probs.home, fair_market_prob=fair_market_probs.home,
            label=f"{match_id}:home",
        ),
        ValueBetCandidate(
            match_id=match_id, outcome=Outcome.DRAW, decimal_odds=draw_odds,
            calibrated_prob=calibrated_probs.draw, fair_market_prob=fair_market_probs.draw,
            label=f"{match_id}:draw",
        ),
        ValueBetCandidate(
            match_id=match_id, outcome=Outcome.AWAY, decimal_odds=away_odds,
            calibrated_prob=calibrated_probs.away, fair_market_prob=fair_market_probs.away,
            label=f"{match_id}:away",
        ),
    ]


class EdgeDetector:
    """
    Filters a pool of ValueBetCandidate objects down to genuine value-bet
    signals, requiring BOTH a minimum Edge and a minimum EV to pass.
    """

    def __init__(self, min_edge: float = 0.03, min_ev: float = 0.03) -> None:
        if min_edge < 0:
            raise ValueError("min_edge must be >= 0")
        if min_ev < 0:
            raise ValueError("min_ev must be >= 0")
        self.min_edge = min_edge
        self.min_ev = min_ev

    def detect(self, candidates: Sequence[ValueBetCandidate]) -> List[ValueBetSignal]:
        signals: List[ValueBetSignal] = []
        for c in candidates:
            edge = c.edge
            ev = c.expected_value
            if edge >= self.min_edge and ev >= self.min_ev:
                signals.append(ValueBetSignal(candidate=c, edge=edge, expected_value=ev))
            else:
                logger.debug(
                    "Rejected %s: edge=%.4f (min %.4f), ev=%.4f (min %.4f)",
                    c.label or c.match_id, edge, self.min_edge, ev, self.min_ev,
                )

        signals.sort(key=lambda s: s.expected_value, reverse=True)
        logger.info(
            "Edge detection: %d/%d candidates passed (min_edge=%.3f, min_ev=%.3f)",
            len(signals), len(candidates), self.min_edge, self.min_ev,
        )
        return signals
