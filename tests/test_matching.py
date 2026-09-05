from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from football_vnext.domain.models.match import Match
from football_vnext.domain.odds.matching import TeamNameMatcher, _normalize_team_name
from football_vnext.domain.odds.models import BookmakerOdds, MatchOddsQuote


def _match(home_name: str, away_name: str, kickoff: datetime) -> Match:
    return Match(
        match_id="M1", competition="Premier League", season="2025",
        kickoff=kickoff, home_team_id="57", away_team_id="61",
        home_team_name=home_name, away_team_name=away_name,
    )


def _quote(home_name: str, away_name: str, commence_time: datetime) -> MatchOddsQuote:
    return MatchOddsQuote(
        home_team_name=home_name, away_team_name=away_name, commence_time=commence_time,
        bookmaker_quotes=[BookmakerOdds(bookmaker="X", home=2.0, draw=3.0, away=4.0)],
    )


def test_normalize_strips_suffixes_and_case():
    assert _normalize_team_name("Arsenal FC") == "arsenal"
    assert _normalize_team_name("ARSENAL") == "arsenal"
    assert _normalize_team_name("Athletic Club") == "athletic club"


def test_matches_when_names_and_time_align():
    kickoff = datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc)
    match = _match("Arsenal FC", "Chelsea FC", kickoff)
    quote = _quote("Arsenal", "Chelsea", kickoff + timedelta(minutes=5))

    matcher = TeamNameMatcher()
    result = matcher.find_match(match, [quote])

    assert result is not None
    assert result.home_team_name == "Arsenal"


def test_no_match_when_team_names_differ():
    kickoff = datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc)
    match = _match("Arsenal FC", "Chelsea FC", kickoff)
    quote = _quote("Liverpool", "Everton", kickoff)

    matcher = TeamNameMatcher()
    assert matcher.find_match(match, [quote]) is None


def test_no_match_when_kickoff_too_far_apart():
    kickoff = datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc)
    match = _match("Arsenal FC", "Chelsea FC", kickoff)
    quote = _quote("Arsenal", "Chelsea", kickoff + timedelta(days=3))

    matcher = TeamNameMatcher(kickoff_tolerance=timedelta(hours=12))
    assert matcher.find_match(match, [quote]) is None


def test_picks_correct_quote_among_several():
    kickoff = datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc)
    match = _match("Arsenal FC", "Chelsea FC", kickoff)
    quotes = [
        _quote("Liverpool", "Everton", kickoff),
        _quote("Arsenal", "Chelsea", kickoff),
        _quote("Manchester United", "Manchester City", kickoff),
    ]

    matcher = TeamNameMatcher()
    result = matcher.find_match(match, quotes)
    assert result is not None
    assert result.home_team_name == "Arsenal"


def test_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        TeamNameMatcher(kickoff_tolerance=timedelta(hours=-1))
