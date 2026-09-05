from __future__ import annotations

import logging
from enum import Enum
from typing import Sequence

import numpy as np

from football_vnext.domain.odds.models import BookmakerOdds

logger = logging.getLogger(__name__)


class AggregationError(Exception):
    """Raised when odds aggregation cannot be performed."""


class AggregationMethod(str, Enum):
    MEDIAN = "median"
    BEST_PRICE = "best_price"


class OddsAggregator:
    """
    Combines quotes from multiple bookmakers into a single representative
    odds set before de-vigging.

    MEDIAN is the sane default for edge detection: it is robust to a single
    mispriced or stale bookmaker skewing the result. BEST_PRICE (the highest
    available decimal odds per outcome) is what you would actually place the
    bet at operationally, but it overstates the "true" market view because it
    cherry-picks the most generous quote per outcome across possibly
    different bookmakers (which is not a coherent single bookmaker's book).
    """

    def __init__(self, method: AggregationMethod = AggregationMethod.MEDIAN) -> None:
        self.method = method

    def aggregate(self, quotes: Sequence[BookmakerOdds]) -> BookmakerOdds:
        if not quotes:
            raise AggregationError("Cannot aggregate an empty list of bookmaker quotes.")

        home_odds = np.array([q.home for q in quotes])
        draw_odds = np.array([q.draw for q in quotes])
        away_odds = np.array([q.away for q in quotes])

        if self.method == AggregationMethod.MEDIAN:
            agg_home, agg_draw, agg_away = (
                float(np.median(home_odds)),
                float(np.median(draw_odds)),
                float(np.median(away_odds)),
            )
            label = "aggregated_median"
        elif self.method == AggregationMethod.BEST_PRICE:
            agg_home, agg_draw, agg_away = (
                float(home_odds.max()),
                float(draw_odds.max()),
                float(away_odds.max()),
            )
            label = "aggregated_best_price"
        else:
            raise AggregationError(f"Unknown aggregation method: {self.method}")

        logger.info(
            "Aggregated %d bookmaker quotes via %s: home=%.3f draw=%.3f away=%.3f",
            len(quotes), self.method.value, agg_home, agg_draw, agg_away,
        )
        return BookmakerOdds(bookmaker=label, home=agg_home, draw=agg_draw, away=agg_away)
