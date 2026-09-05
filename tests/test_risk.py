from __future__ import annotations

import pytest

from football_vnext.domain.odds.models import BookmakerOdds
from football_vnext.domain.value.edge import Outcome
from football_vnext.domain.value.risk import RiskScoreCalculator, RiskScoreError


def test_rejects_bad_weights():
    with pytest.raises(RiskScoreError):
        RiskScoreCalculator(weights=(0.5, 0.5, 0.5))


def test_rejects_invalid_sample_size_or_threshold():
    with pytest.raises(RiskScoreError):
        RiskScoreCalculator(min_reliable_sample_size=0)
    with pytest.raises(RiskScoreError):
        RiskScoreCalculator(max_acceptable_risk_score=0.0)


def test_single_bookmaker_gets_moderate_disagreement_risk():
    calc = RiskScoreCalculator()
    quotes = [BookmakerOdds(bookmaker="A", home=2.0, draw=3.2, away=4.0)]
    assessment = calc.compute(
        quotes, Outcome.HOME, home_team_matches=30, away_team_matches=30, competition="Sample League"
    )
    assert assessment.bookmaker_disagreement == pytest.approx(0.5)


def test_high_bookmaker_disagreement_scores_higher_risk_than_low():
    calc = RiskScoreCalculator()
    low_disagreement_quotes = [
        BookmakerOdds(bookmaker="A", home=2.00, draw=3.2, away=4.0),
        BookmakerOdds(bookmaker="B", home=2.02, draw=3.2, away=4.0),
        BookmakerOdds(bookmaker="C", home=1.98, draw=3.2, away=4.0),
    ]
    high_disagreement_quotes = [
        BookmakerOdds(bookmaker="A", home=1.50, draw=3.2, away=4.0),
        BookmakerOdds(bookmaker="B", home=2.50, draw=3.2, away=4.0),
        BookmakerOdds(bookmaker="C", home=3.50, draw=3.2, away=4.0),
    ]
    low_assessment = calc.compute(
        low_disagreement_quotes, Outcome.HOME, 30, 30, competition="Sample League"
    )
    high_assessment = calc.compute(
        high_disagreement_quotes, Outcome.HOME, 30, 30, competition="Sample League"
    )
    assert high_assessment.bookmaker_disagreement > low_assessment.bookmaker_disagreement
    assert high_assessment.risk_score > low_assessment.risk_score


def test_low_sample_size_scores_higher_data_quality_risk():
    calc = RiskScoreCalculator(min_reliable_sample_size=20)
    quotes = [BookmakerOdds(bookmaker="A", home=2.0, draw=3.2, away=4.0)]
    low_data_assessment = calc.compute(quotes, Outcome.HOME, 2, 30, competition="Sample League")
    high_data_assessment = calc.compute(quotes, Outcome.HOME, 30, 30, competition="Sample League")
    assert low_data_assessment.data_quality_risk > high_data_assessment.data_quality_risk
    assert low_data_assessment.risk_score > high_data_assessment.risk_score


def test_high_efficiency_league_scores_higher_efficiency_risk():
    calc = RiskScoreCalculator()
    quotes = [BookmakerOdds(bookmaker="A", home=2.0, draw=3.2, away=4.0)]
    high_eff = calc.compute(quotes, Outcome.HOME, 30, 30, competition="Premier League")
    low_eff = calc.compute(quotes, Outcome.HOME, 30, 30, competition="Sample League")
    assert high_eff.market_efficiency_risk > low_eff.market_efficiency_risk
    assert high_eff.risk_score > low_eff.risk_score


def test_is_acceptable_flag_respects_threshold():
    calc = RiskScoreCalculator(max_acceptable_risk_score=0.3)
    # Deliberately risky: single bookmaker, low sample size, high-efficiency league
    quotes = [BookmakerOdds(bookmaker="A", home=2.0, draw=3.2, away=4.0)]
    risky_assessment = calc.compute(quotes, Outcome.HOME, 2, 2, competition="Premier League")
    assert risky_assessment.is_acceptable is False

    lenient_calc = RiskScoreCalculator(max_acceptable_risk_score=1.0)
    lenient_assessment = lenient_calc.compute(quotes, Outcome.HOME, 2, 2, competition="Premier League")
    assert lenient_assessment.is_acceptable is True


def test_negative_team_matches_raises():
    calc = RiskScoreCalculator()
    quotes = [BookmakerOdds(bookmaker="A", home=2.0, draw=3.2, away=4.0)]
    with pytest.raises(RiskScoreError):
        calc.compute(quotes, Outcome.HOME, -1, 30, competition="Sample League")
