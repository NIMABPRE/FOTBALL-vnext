from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from football_vnext.application.odds_loading import load_match_odds
from football_vnext.config import Settings
from football_vnext.domain.models.match import Match
from football_vnext.domain.odds.models import BookmakerOdds, MatchOddsQuote
from football_vnext.infrastructure.data_sources.exceptions import DataSourceError

EXAMPLE_ODDS = BookmakerOdds(bookmaker="example", home=1.72, draw=3.80, away=5.25)


def _match() -> Match:
    return Match(
        match_id="M1", competition="Premier League", season="2025",
        kickoff=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc),
        home_team_id="57", away_team_id="61",
        home_team_name="Arsenal FC", away_team_name="Chelsea FC",
    )


def test_uses_example_when_no_sources_configured():
    settings = Settings()
    result = load_match_odds(_match(), EXAMPLE_ODDS, settings=settings)
    assert result.source == "example"
    assert result.odds == EXAMPLE_ODDS


@patch("football_vnext.application.odds_loading.TheOddsApiAdapter")
def test_uses_the_odds_api_when_match_found(mock_adapter_cls):
    quote = MatchOddsQuote(
        home_team_name="Arsenal", away_team_name="Chelsea",
        commence_time=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc),
        bookmaker_quotes=[
            BookmakerOdds(bookmaker="Bet365", home=1.80, draw=3.70, away=5.00),
            BookmakerOdds(bookmaker="WilliamHill", home=1.85, draw=3.75, away=4.90),
        ],
    )
    mock_adapter_cls.return_value.fetch_odds.return_value = [quote]
    settings = Settings(odds_api_key="test-key")

    result = load_match_odds(_match(), EXAMPLE_ODDS, settings=settings)

    assert result.source == "the-odds-api"
    assert result.fallback_reason is None
    assert result.odds.home == pytest.approx(1.825)  # median of [1.80, 1.85]


@patch("football_vnext.application.odds_loading.ApiFootballAdapter")
@patch("football_vnext.application.odds_loading.TheOddsApiAdapter")
def test_falls_back_to_api_football_when_odds_api_has_no_match(mock_odds_api_cls, mock_af_cls):
    unrelated_quote = MatchOddsQuote(
        home_team_name="Liverpool", away_team_name="Everton",
        commence_time=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc),
        bookmaker_quotes=[BookmakerOdds(bookmaker="X", home=2.0, draw=3.0, away=4.0)],
    )
    mock_odds_api_cls.return_value.fetch_odds.return_value = [unrelated_quote]

    af_quote = MatchOddsQuote(
        home_team_name="(fixture-id-keyed)", away_team_name="(fixture-id-keyed)",
        commence_time=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc),
        bookmaker_quotes=[BookmakerOdds(bookmaker="Bet365", home=1.75, draw=3.75, away=5.10)],
    )
    mock_af_cls.return_value.fetch_odds_for_fixture.return_value = af_quote

    settings = Settings(odds_api_key="odds-key", api_football_key="af-key")
    result = load_match_odds(_match(), EXAMPLE_ODDS, api_football_fixture_id=12345, settings=settings)

    assert result.source == "api-football"
    assert result.odds.home == pytest.approx(1.75)


@patch("football_vnext.application.odds_loading.ApiFootballAdapter")
def test_api_football_skipped_without_fixture_id(mock_af_cls):
    settings = Settings(api_football_key="af-key")
    result = load_match_odds(_match(), EXAMPLE_ODDS, api_football_fixture_id=None, settings=settings)

    mock_af_cls.return_value.fetch_odds_for_fixture.assert_not_called()
    assert result.source == "example"


@patch("football_vnext.application.odds_loading.ApiFootballAdapter")
@patch("football_vnext.application.odds_loading.TheOddsApiAdapter")
def test_falls_back_to_example_when_all_real_sources_fail(mock_odds_api_cls, mock_af_cls):
    mock_odds_api_cls.return_value.fetch_odds.side_effect = DataSourceError("boom")
    mock_af_cls.return_value.fetch_odds_for_fixture.return_value = None
    settings = Settings(odds_api_key="odds-key", api_football_key="af-key")

    result = load_match_odds(_match(), EXAMPLE_ODDS, api_football_fixture_id=1, settings=settings)

    assert result.source == "example"
    assert result.fallback_reason is not None
