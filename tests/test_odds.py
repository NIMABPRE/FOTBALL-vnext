from __future__ import annotations

import pytest

from football_vnext.domain.odds.aggregation import AggregationError, AggregationMethod, OddsAggregator
from football_vnext.domain.odds.models import BookmakerOdds


def test_bookmaker_odds_rejects_odds_at_or_below_one():
    with pytest.raises(ValueError):
        BookmakerOdds(bookmaker="X", home=1.0, draw=3.0, away=3.0)


def test_bookmaker_odds_rejects_empty_name():
    with pytest.raises(ValueError):
        BookmakerOdds(bookmaker="   ", home=2.0, draw=3.0, away=3.5)


def test_implied_probabilities_and_overround():
    odds = BookmakerOdds(bookmaker="Bet365", home=2.0, draw=3.0, away=4.0)
    implied = odds.implied_probabilities()
    assert implied == pytest.approx([0.5, 1 / 3, 0.25], abs=1e-9)
    # sum = 0.5 + 0.3333 + 0.25 = 1.0833 -> overround ~8.33%
    assert odds.overround() == pytest.approx(0.0833, abs=1e-3)


def test_aggregator_rejects_empty_list():
    aggregator = OddsAggregator()
    with pytest.raises(AggregationError):
        aggregator.aggregate([])


def test_aggregator_median():
    quotes = [
        BookmakerOdds(bookmaker="A", home=2.0, draw=3.0, away=4.0),
        BookmakerOdds(bookmaker="B", home=2.1, draw=3.2, away=3.8),
        BookmakerOdds(bookmaker="C", home=1.9, draw=3.1, away=4.2),
    ]
    result = OddsAggregator(AggregationMethod.MEDIAN).aggregate(quotes)
    assert result.home == pytest.approx(2.0)
    assert result.draw == pytest.approx(3.1)
    assert result.away == pytest.approx(4.0)


def test_aggregator_best_price_takes_max_per_outcome():
    quotes = [
        BookmakerOdds(bookmaker="A", home=2.0, draw=3.0, away=4.0),
        BookmakerOdds(bookmaker="B", home=2.1, draw=3.2, away=3.8),
    ]
    result = OddsAggregator(AggregationMethod.BEST_PRICE).aggregate(quotes)
    assert result.home == pytest.approx(2.1)
    assert result.draw == pytest.approx(3.2)
    assert result.away == pytest.approx(4.0)
