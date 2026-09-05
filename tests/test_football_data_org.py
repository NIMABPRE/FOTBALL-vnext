"""
Tests for FootballDataOrgAdapter.

These tests mock `requests.Session.get` rather than hitting the real API.
This is deliberate, not a shortcut: this development environment's network
egress does not include api.football-data.org, so live-API tests are not
possible here. What IS tested and verified for real: JSON parsing, status
code handling, retry/backoff behavior, and rate-limit handling. What is NOT
verified here: that a real API key against the real API actually returns
data in the shape we expect — do that once, manually, before production use.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from football_vnext.domain.models.match import MatchStatus
from football_vnext.infrastructure.data_sources.exceptions import (
    AuthenticationError,
    DataSourceUnavailableError,
    RateLimitExceededError,
)
from football_vnext.infrastructure.data_sources.football_data_org import FootballDataOrgAdapter

SAMPLE_RESPONSE = {
    "matches": [
        {
            "id": 12345,
            "utcDate": "2026-01-10T15:00:00Z",
            "status": "FINISHED",
            "matchday": 20,
            "season": {"startDate": "2025-08-01"},
            "competition": {"name": "Premier League"},
            "homeTeam": {"id": 57, "name": "Arsenal FC"},
            "awayTeam": {"id": 61, "name": "Chelsea FC"},
            "score": {"fullTime": {"home": 2, "away": 1}},
        },
        {
            "id": 12346,
            "utcDate": "2026-01-17T15:00:00Z",
            "status": "SCHEDULED",
            "matchday": 21,
            "season": {"startDate": "2025-08-01"},
            "competition": {"name": "Premier League"},
            "homeTeam": {"id": 61, "name": "Chelsea FC"},
            "awayTeam": {"id": 57, "name": "Arsenal FC"},
            "score": {"fullTime": {"home": None, "away": None}},
        },
    ]
}


def _make_response(status_code: int, json_data=None, headers=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.text = text
    return resp


def test_rejects_empty_api_key():
    with pytest.raises(AuthenticationError):
        FootballDataOrgAdapter(api_key="")


def test_rejects_invalid_retry_params():
    with pytest.raises(ValueError):
        FootballDataOrgAdapter(api_key="x", max_retries=-1)
    with pytest.raises(ValueError):
        FootballDataOrgAdapter(api_key="x", backoff_factor=-1)


def test_rejects_empty_competition_code():
    adapter = FootballDataOrgAdapter(api_key="test-key")
    with pytest.raises(ValueError):
        adapter.fetch_matches("")


@patch("requests.Session.get")
def test_fetch_matches_parses_successful_response(mock_get):
    mock_get.return_value = _make_response(200, SAMPLE_RESPONSE)
    adapter = FootballDataOrgAdapter(api_key="test-key")

    matches = adapter.fetch_matches("PL", season=2025)

    assert len(matches) == 2
    finished, scheduled = matches
    assert finished.match_id == "12345"
    assert finished.status == MatchStatus.FINISHED
    assert finished.result is not None
    assert finished.result.home_goals == 2
    assert finished.result.away_goals == 1
    assert finished.home_team_id == "57"
    assert finished.home_team_name == "Arsenal FC"

    assert scheduled.status == MatchStatus.SCHEDULED
    assert scheduled.result is None


@patch("requests.Session.get")
def test_fetch_matches_skips_malformed_record_and_continues(mock_get):
    broken_response = {
        "matches": [
            SAMPLE_RESPONSE["matches"][0],
            {"id": 999, "utcDate": "not-a-date", "homeTeam": {}, "awayTeam": {}},  # malformed
        ]
    }
    mock_get.return_value = _make_response(200, broken_response)
    adapter = FootballDataOrgAdapter(api_key="test-key")

    matches = adapter.fetch_matches("PL")

    assert len(matches) == 1
    assert matches[0].match_id == "12345"


@patch("requests.Session.get")
def test_raises_authentication_error_on_403(mock_get):
    mock_get.return_value = _make_response(403)
    adapter = FootballDataOrgAdapter(api_key="bad-key")
    with pytest.raises(AuthenticationError):
        adapter.fetch_matches("PL")


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_retries_on_429_then_succeeds(mock_get, mock_sleep):
    mock_get.side_effect = [
        _make_response(429, headers={"Retry-After": "1"}),
        _make_response(200, SAMPLE_RESPONSE),
    ]
    adapter = FootballDataOrgAdapter(api_key="test-key", max_retries=2)

    matches = adapter.fetch_matches("PL")

    assert len(matches) == 2
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once()


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_raises_rate_limit_error_after_exhausting_retries(mock_get, mock_sleep):
    mock_get.return_value = _make_response(429, headers={"Retry-After": "1"})
    adapter = FootballDataOrgAdapter(api_key="test-key", max_retries=2)

    with pytest.raises(RateLimitExceededError):
        adapter.fetch_matches("PL")

    assert mock_get.call_count == 3  # initial + 2 retries


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_retries_on_server_error_then_succeeds(mock_get, mock_sleep):
    mock_get.side_effect = [
        _make_response(503, text="Service Unavailable"),
        _make_response(200, SAMPLE_RESPONSE),
    ]
    adapter = FootballDataOrgAdapter(api_key="test-key", max_retries=2)

    matches = adapter.fetch_matches("PL")

    assert len(matches) == 2


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_raises_after_persistent_server_errors(mock_get, mock_sleep):
    mock_get.return_value = _make_response(500, text="Internal Server Error")
    adapter = FootballDataOrgAdapter(api_key="test-key", max_retries=1)

    with pytest.raises(DataSourceUnavailableError):
        adapter.fetch_matches("PL")


@patch("requests.Session.get")
def test_raises_on_unexpected_status_code(mock_get):
    mock_get.return_value = _make_response(418, text="I'm a teapot")
    adapter = FootballDataOrgAdapter(api_key="test-key")

    with pytest.raises(DataSourceUnavailableError):
        adapter.fetch_matches("PL")


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_retries_on_network_exception(mock_get, mock_sleep):
    import requests as requests_module

    mock_get.side_effect = [
        requests_module.exceptions.ConnectionError("boom"),
        _make_response(200, SAMPLE_RESPONSE),
    ]
    adapter = FootballDataOrgAdapter(api_key="test-key", max_retries=2)

    matches = adapter.fetch_matches("PL")

    assert len(matches) == 2
