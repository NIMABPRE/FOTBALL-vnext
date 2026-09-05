from __future__ import annotations

import pytest

from football_vnext.domain.backtest.clv import BetRecord, CLVAnalyzer, CLVError
from football_vnext.domain.value.edge import Outcome


def _bet(odds_taken: float, closing_odds: float, won: bool = True) -> BetRecord:
    return BetRecord(
        match_id="M1", outcome=Outcome.HOME, odds_taken=odds_taken, closing_odds=closing_odds,
        stake_fraction=0.02, stake_amount=2.0, won=won, pnl=1.0 if won else -2.0,
    )


def test_clv_known_value():
    bet = _bet(odds_taken=2.2, closing_odds=2.0)
    assert bet.clv == pytest.approx(0.10)


def test_negative_clv_when_odds_taken_worse_than_closing():
    bet = _bet(odds_taken=1.8, closing_odds=2.0)
    assert bet.clv < 0


def test_bet_record_rejects_bad_odds():
    with pytest.raises(CLVError):
        BetRecord(
            match_id="M1", outcome=Outcome.HOME, odds_taken=1.0, closing_odds=2.0,
            stake_fraction=0.02, stake_amount=2.0, won=True, pnl=1.0,
        )


def test_summarize_empty_raises():
    with pytest.raises(CLVError):
        CLVAnalyzer().summarize([])


def test_summarize_consistently_positive_clv_is_significant():
    bets = [_bet(odds_taken=2.2, closing_odds=2.0) for _ in range(50)]
    summary = CLVAnalyzer().summarize(bets)
    assert summary.mean_clv == pytest.approx(0.10)
    assert summary.pct_positive_clv == 1.0
    assert summary.clv_significantly_positive is True


def test_summarize_mixed_clv_around_zero_is_not_significant():
    bets = [_bet(odds_taken=2.0, closing_odds=2.0) for _ in range(50)]
    summary = CLVAnalyzer().summarize(bets)
    assert summary.mean_clv == pytest.approx(0.0)
    assert summary.clv_significantly_positive is False


def test_rejects_invalid_significance_level():
    with pytest.raises(CLVError):
        CLVAnalyzer(significance_level=0.0)
    with pytest.raises(CLVError):
        CLVAnalyzer(significance_level=1.0)
