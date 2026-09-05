from datetime import datetime, timedelta, timezone

import pytest

from football_vnext.domain.features.xg import XGFeatureEngine, XGTeamSignal
from football_vnext.domain.models.match import Match, MatchResult


def _match(i, home, away, hg, ag, hxg, axg):
    return Match(
        match_id=str(i), competition="Test", season="2526",
        kickoff=datetime(2025, 8, 1, tzinfo=timezone.utc) + timedelta(days=i),
        home_team_id=home, away_team_id=away,
        home_team_name=home, away_team_name=away,
        status="finished", result=MatchResult(home_goals=hg, away_goals=ag),
        home_xg=hxg, away_xg=axg,
    )


def test_worse_defense_increases_opponent_lambda():
    """
    Hand-constructed signals, matching this ratio's OWN convention (higher
    defence_ratio = worse true defence -- see fit()'s docstring/comment).
    Kept alongside the end-to-end test below, which is what actually proves
    the convention is applied consistently with what fit() produces.
    """
    engine = XGFeatureEngine()
    engine.signals = {
        "H": XGTeamSignal(1.0, 1.0, 20),
        "A": XGTeamSignal(1.0, 1.2, 20),  # A's defence_ratio > 1 -> worse true defence
    }
    home, away = engine.multipliers("H", "A")
    assert home > 1.0  # H benefits from A's worse defence
    assert away == pytest.approx(1.0)


def test_stronger_defense_reduces_opponent_lambda():
    engine = XGFeatureEngine()
    engine.signals = {
        "H": XGTeamSignal(1.0, 0.85, 20),  # H's defence_ratio < 1 -> better true defence
        "A": XGTeamSignal(1.0, 1.0, 20),
    }
    home, away = engine.multipliers("H", "A")
    assert home == pytest.approx(1.0)
    assert away < 1.0  # A is held back by H's stronger defence


def test_fit_counts_only_settled_matches_with_xg():
    rows = [_match(i, "H", "A", 1, 0, 1.2, 0.8) for i in range(4)]
    engine = XGFeatureEngine()
    engine.fit(rows)
    assert engine.signals["H"].samples == 4
    assert engine.signals["A"].samples == 4


def test_end_to_end_lucky_team_correctly_increases_opponent_lambda():
    """
    Regression test for a real bug: this exercises the FULL fit() ->
    multipliers() pipeline with realistic data, rather than hand-constructing
    an XGTeamSignal that could itself embed a wrong assumption about the
    ratio's direction (which is exactly how the original bug shipped
    undetected). "LuckyTeam" concedes far fewer actual goals than the xG
    they faced suggests they should -- their TRUE underlying defence is
    WORSE than recent results show, so an opponent should be projected to
    score MORE against them, not less.
    """
    engine = XGFeatureEngine(prior_weight=0.5)
    rows = []
    for i in range(20):
        # LuckyTeam concedes 0 actual goals despite facing high-quality
        # chances (xG against = 2.5) in every match, home or away.
        rows.append(_match(i, "LuckyTeam", "Opponent", 1, 0, 1.0, 2.5))
        rows.append(_match(i + 100, "Opponent", "LuckyTeam", 0, 1, 2.5, 1.0))

    engine.fit(rows)
    assert engine.signals["LuckyTeam"].defence_ratio > 1.0  # confirm the fixture behaves as intended

    home_mult, _ = engine.multipliers("Opponent", "LuckyTeam")
    assert home_mult > 1.0, (
        "Opponent should be projected to score MORE against LuckyTeam's "
        "worse true defence, not less."
    )
