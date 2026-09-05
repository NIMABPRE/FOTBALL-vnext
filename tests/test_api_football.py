from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from football_vnext.domain.models.match import MatchStatus
from football_vnext.infrastructure.data_sources.api_football import ApiFootballAdapter
from football_vnext.infrastructure.data_sources.exceptions import (
    AuthenticationError,
    DataSourceUnavailableError,
    RateLimitExceededError,
)

SAMPLE_FIXTURES_RESPONSE = {
    "response": [
        {
            "fixture": {
                "id": 1035001,
                "date": "2026-01-10T15:00:00+00:00",
                "status": {"short": "FT"},
            },
            "league": {"id": 39, "name": "Premier League", "season": 2025},
            "teams": {
                "home": {"id": 42, "name": "Arsenal", "winner": True},
                "away": {"id": 49, "name": "Chelsea", "winner": False},
            },
            "goals": {"home": 2, "away": 1},
        },
        {
            "fixture": {
                "id": 1035002,
                "date": "2026-01-17T15:00:00+00:00",
                "status": {"short": "NS"},
            },
            "league": {"id": 39, "name": "Premier League", "season": 2025},
            "teams": {
                "home": {"id": 49, "name": "Chelsea", "winner": None},
                "away": {"id": 42, "name": "Arsenal", "winner": None},
            },
            "goals": {"home": None, "away": None},
        },
    ]
}

SAMPLE_ODDS_RESPONSE = {
    "response": [
        {
            "fixture": {"id": 1035001, "date": "2026-01-10T15:00:00+00:00"},
            "bookmakers": [
                {
                    "id": 8,
                    "name": "Bet365",
                    "bets": [
                        {
                            "id": 1,
                            "name": "Match Winner",
                            "values": [
                                {"value": "Home", "odd": "1.72"},
                                {"value": "Draw", "odd": "3.80"},
                                {"value": "Away", "odd": "5.25"},
                            ],
                        }
                    ],
                },
                {
                    "id": 11,
                    "name": "Marathonbet",
                    "bets": [
                        {
                            "id": 1,
                            "name": "Match Winner",
                            "values": [
                                {"value": "Home", "odd": "1.75"},
                                {"value": "Draw", "odd": "3.75"},
                                {"value": "Away", "odd": "5.10"},
                            ],
                        }
                    ],
                },
            ],
        }
    ]
}


def _make_response(status_code: int, json_data=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def test_rejects_empty_api_key():
    with pytest.raises(AuthenticationError):
        ApiFootballAdapter(api_key="")


def test_rejects_invalid_retry_params():
    with pytest.raises(ValueError):
        ApiFootballAdapter(api_key="x", max_retries=-1)


@patch("requests.Session.get")
def test_fetch_fixtures_parses_successful_response(mock_get):
    mock_get.return_value = _make_response(200, SAMPLE_FIXTURES_RESPONSE)
    adapter = ApiFootballAdapter(api_key="test-key")

    matches = adapter.fetch_fixtures(league_id=39, season=2025)

    assert len(matches) == 2
    finished, scheduled = matches
    assert finished.match_id == "1035001"
    assert finished.status == MatchStatus.FINISHED
    assert finished.result.home_goals == 2
    assert finished.result.away_goals == 1
    assert finished.home_team_id == "42"
    assert finished.home_team_name == "Arsenal"

    assert scheduled.status == MatchStatus.SCHEDULED
    assert scheduled.result is None


@patch("requests.Session.get")
def test_fetch_fixtures_skips_malformed_record(mock_get):
    broken = {"response": [SAMPLE_FIXTURES_RESPONSE["response"][0], {"fixture": {"id": 999}}]}
    mock_get.return_value = _make_response(200, broken)
    adapter = ApiFootballAdapter(api_key="test-key")

    matches = adapter.fetch_fixtures(league_id=39, season=2025)
    assert len(matches) == 1


@patch("requests.Session.get")
def test_fetch_odds_for_fixture_parses_multiple_bookmakers(mock_get):
    mock_get.return_value = _make_response(200, SAMPLE_ODDS_RESPONSE)
    adapter = ApiFootballAdapter(api_key="test-key")

    quote = adapter.fetch_odds_for_fixture(1035001)

    assert quote is not None
    assert len(quote.bookmaker_quotes) == 2
    bet365 = next(b for b in quote.bookmaker_quotes if b.bookmaker == "Bet365")
    assert bet365.home == 1.72


@patch("requests.Session.get")
def test_fetch_odds_returns_none_when_no_bookmakers_have_market(mock_get):
    empty = {"response": [{"fixture": {"id": 1, "date": "2026-01-10T15:00:00+00:00"}, "bookmakers": []}]}
    mock_get.return_value = _make_response(200, empty)
    adapter = ApiFootballAdapter(api_key="test-key")

    quote = adapter.fetch_odds_for_fixture(1)
    assert quote is None


@patch("requests.Session.get")
def test_fetch_odds_returns_none_when_response_empty(mock_get):
    mock_get.return_value = _make_response(200, {"response": []})
    adapter = ApiFootballAdapter(api_key="test-key")

    quote = adapter.fetch_odds_for_fixture(999)
    assert quote is None


@patch("requests.Session.get")
def test_raises_authentication_error_on_403(mock_get):
    mock_get.return_value = _make_response(403)
    adapter = ApiFootballAdapter(api_key="bad-key")
    with pytest.raises(AuthenticationError):
        adapter.fetch_fixtures(league_id=39, season=2025)


@patch("requests.Session.get")
def test_raises_authentication_error_on_token_error_in_200_response(mock_get):
    mock_get.return_value = _make_response(200, {"errors": {"token": "Invalid API key"}, "response": []})
    adapter = ApiFootballAdapter(api_key="bad-key")
    with pytest.raises(AuthenticationError):
        adapter.fetch_fixtures(league_id=39, season=2025)


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_raises_rate_limit_error_after_exhausting_retries(mock_get, mock_sleep):
    mock_get.return_value = _make_response(429)
    adapter = ApiFootballAdapter(api_key="test-key", max_retries=1)
    with pytest.raises(RateLimitExceededError):
        adapter.fetch_fixtures(league_id=39, season=2025)


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_raises_after_persistent_server_errors(mock_get, mock_sleep):
    mock_get.return_value = _make_response(500)
    adapter = ApiFootballAdapter(api_key="test-key", max_retries=1)
    with pytest.raises(DataSourceUnavailableError):
        adapter.fetch_fixtures(league_id=39, season=2025)
