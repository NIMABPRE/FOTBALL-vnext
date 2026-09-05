"""
Tests for TheOddsApiAdapter — mocked HTTP, same rationale as
test_football_data_org.py (this dev sandbox cannot reach api.the-odds-api.com).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from football_vnext.infrastructure.data_sources.exceptions import (
    AuthenticationError,
    DataSourceUnavailableError,
    RateLimitExceededError,
)
from football_vnext.infrastructure.data_sources.odds_api import TheOddsApiAdapter

SAMPLE_RESPONSE = [
    {
        "id": "abc123",
        "sport_key": "soccer_epl",
        "commence_time": "2026-01-10T15:00:00Z",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {
                "key": "bet365",
                "title": "Bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 1.72},
                            {"name": "Draw", "price": 3.80},
                            {"name": "Chelsea", "price": 5.25},
                        ],
                    }
                ],
            },
            {
                "key": "williamhill",
                "title": "William Hill",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 1.70},
                            {"name": "Draw", "price": 3.75},
                            {"name": "Chelsea", "price": 5.30},
                        ],
                    }
                ],
            },
        ],
    }
]


def _make_response(status_code: int, json_data=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or []
    resp.text = text
    return resp


def test_rejects_empty_api_key():
    with pytest.raises(AuthenticationError):
        TheOddsApiAdapter(api_key="")


def test_rejects_invalid_retry_params():
    with pytest.raises(ValueError):
        TheOddsApiAdapter(api_key="x", max_retries=-1)


@patch("requests.Session.get")
def test_fetch_odds_parses_successful_response(mock_get):
    mock_get.return_value = _make_response(200, SAMPLE_RESPONSE)
    adapter = TheOddsApiAdapter(api_key="test-key")

    quotes = adapter.fetch_odds()

    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.home_team_name == "Arsenal"
    assert quote.away_team_name == "Chelsea"
    assert len(quote.bookmaker_quotes) == 2
    bet365 = next(b for b in quote.bookmaker_quotes if b.bookmaker == "Bet365")
    assert bet365.home == 1.72
    assert bet365.draw == 3.80
    assert bet365.away == 5.25


@patch("requests.Session.get")
def test_skips_bookmaker_with_incomplete_market(mock_get):
    broken = [
        {
            "id": "abc123",
            "commence_time": "2026-01-10T15:00:00Z",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "key": "bad",
                    "title": "BadBookmaker",
                    "markets": [
                        {"key": "h2h", "outcomes": [{"name": "Arsenal", "price": 1.72}]}
                    ],
                },
                SAMPLE_RESPONSE[0]["bookmakers"][0],
            ],
        }
    ]
    mock_get.return_value = _make_response(200, broken)
    adapter = TheOddsApiAdapter(api_key="test-key")

    quotes = adapter.fetch_odds()

    assert len(quotes) == 1
    assert len(quotes[0].bookmaker_quotes) == 1  # only the good bookmaker survived
    assert quotes[0].bookmaker_quotes[0].bookmaker == "Bet365"


@patch("requests.Session.get")
def test_skips_event_with_no_usable_bookmakers(mock_get):
    no_h2h = [
        {
            "id": "xyz",
            "commence_time": "2026-01-10T15:00:00Z",
            "home_team": "A",
            "away_team": "B",
            "bookmakers": [{"key": "x", "title": "X", "markets": []}],
        }
    ]
    mock_get.return_value = _make_response(200, no_h2h)
    adapter = TheOddsApiAdapter(api_key="test-key")

    quotes = adapter.fetch_odds()
    assert quotes == []


@patch("requests.Session.get")
def test_raises_authentication_error_on_401(mock_get):
    mock_get.return_value = _make_response(401)
    adapter = TheOddsApiAdapter(api_key="bad-key")
    with pytest.raises(AuthenticationError):
        adapter.fetch_odds()


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_raises_rate_limit_error_after_exhausting_retries(mock_get, mock_sleep):
    mock_get.return_value = _make_response(429)
    adapter = TheOddsApiAdapter(api_key="test-key", max_retries=1)
    with pytest.raises(RateLimitExceededError):
        adapter.fetch_odds()
    assert mock_get.call_count == 2


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_raises_after_persistent_server_errors(mock_get, mock_sleep):
    mock_get.return_value = _make_response(503, text="down")
    adapter = TheOddsApiAdapter(api_key="test-key", max_retries=1)
    with pytest.raises(DataSourceUnavailableError):
        adapter.fetch_odds()
