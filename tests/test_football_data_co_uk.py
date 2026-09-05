"""
Tests for FootballDataCoUkAdapter.

Parsing logic (the CSV column handling, fallback chains, malformed-row
skipping) is tested for real against embedded sample CSV text matching the
site's actual column format. Only the HTTP download itself is mocked (this
dev sandbox's network egress does not include football-data.co.uk) — see
scripts/verify_football_data_co_uk.py to confirm live download works.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from football_vnext.domain.models.match import MatchStatus
from football_vnext.infrastructure.data_sources.exceptions import DataSourceUnavailableError
from football_vnext.infrastructure.data_sources.football_data_co_uk import FootballDataCoUkAdapter

# Mirrors real football-data.co.uk column layout (subset of real columns).
SAMPLE_CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,AvgH,AvgD,AvgA,AvgCH,AvgCD,AvgCA
E0,15/08/25,Arsenal,Chelsea,2,1,H,1.72,3.80,5.25,1.75,3.70,5.10,1.68,3.85,5.40
E0,16/08/25,Liverpool,Everton,3,0,H,1.30,5.50,9.00,1.32,5.40,8.80,1.28,5.60,9.50
E0,17/08/25,Fulham,Brighton,1,1,D,2.40,3.20,3.00,2.35,3.25,3.05,,,
"""

MALFORMED_ROW_CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A
E0,15/08/25,Arsenal,Chelsea,2,1,H,1.72,3.80,5.25
E0,not-a-date,Liverpool,Everton,3,0,H,1.30,5.50,9.00
"""


def _make_response(status_code: int, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


def test_rejects_invalid_retry_params():
    with pytest.raises(ValueError):
        FootballDataCoUkAdapter(max_retries=-1)
    with pytest.raises(ValueError):
        FootballDataCoUkAdapter(backoff_factor=-1)


def test_parse_csv_extracts_matches_and_odds():
    adapter = FootballDataCoUkAdapter()
    records = adapter.parse_csv(SAMPLE_CSV, competition_name="Premier League")

    assert len(records) == 3
    first = records[0]
    assert first.match.home_team_name == "Arsenal"
    assert first.match.season == "unknown"  # parse_csv() without a season remains backward-compatible
    assert first.match.away_team_name == "Chelsea"
    assert first.match.status == MatchStatus.FINISHED
    assert first.match.result.home_goals == 2
    assert first.match.result.away_goals == 1

    # AvgH/AvgD/AvgA preferred over B365 for opening odds
    assert first.opening_odds is not None
    assert first.opening_odds.home == pytest.approx(1.75)
    assert first.opening_odds.bookmaker == "Avg"

    assert first.closing_odds is not None
    assert first.closing_odds.home == pytest.approx(1.68)


def test_fetch_historical_matches_propagates_season_into_match_model():
    adapter = FootballDataCoUkAdapter()
    records = adapter.parse_csv(SAMPLE_CSV, competition_name="Premier League", season="2526")
    assert all(r.match.season == "2526" for r in records)


def test_missing_closing_odds_columns_result_in_none():
    adapter = FootballDataCoUkAdapter()
    records = adapter.parse_csv(SAMPLE_CSV, competition_name="Premier League")

    third = records[2]  # Fulham vs Brighton row has empty AvgC* columns
    assert third.closing_odds is None
    assert third.opening_odds is not None  # opening odds still present


def test_falls_back_to_b365_when_avg_columns_absent():
    csv_without_avg = """Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A
15/08/25,Arsenal,Chelsea,2,1,1.72,3.80,5.25
"""
    adapter = FootballDataCoUkAdapter()
    records = adapter.parse_csv(csv_without_avg, competition_name="Premier League")

    assert len(records) == 1
    assert records[0].opening_odds is not None
    assert records[0].opening_odds.bookmaker == "B365"
    assert records[0].opening_odds.home == pytest.approx(1.72)
    assert records[0].closing_odds is None


def test_skips_malformed_rows_and_continues():
    adapter = FootballDataCoUkAdapter()
    records = adapter.parse_csv(MALFORMED_ROW_CSV, competition_name="Premier League")

    assert len(records) == 1
    assert records[0].match.home_team_name == "Arsenal"


def test_scheduled_match_with_no_score_has_no_result():
    csv_upcoming = """Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A
20/08/26,Arsenal,Chelsea,,,1.72,3.80,5.25
"""
    adapter = FootballDataCoUkAdapter()
    records = adapter.parse_csv(csv_upcoming, competition_name="Premier League")

    assert len(records) == 1
    assert records[0].match.result is None
    assert records[0].match.status == MatchStatus.SCHEDULED


def test_rejects_empty_league_code_or_season():
    adapter = FootballDataCoUkAdapter()
    with pytest.raises(ValueError):
        adapter.fetch_historical_matches("", "2324")
    with pytest.raises(ValueError):
        adapter.fetch_historical_matches("E0", "")


@patch("requests.Session.get")
def test_fetch_historical_matches_end_to_end(mock_get):
    mock_get.return_value = _make_response(200, SAMPLE_CSV)
    adapter = FootballDataCoUkAdapter()

    records = adapter.fetch_historical_matches("E0", "2526", competition_name="Premier League")

    assert len(records) == 3
    called_url = mock_get.call_args[0][0]
    assert "mmz4281/2526/E0.csv" in called_url


@patch("requests.Session.get")
def test_raises_on_404(mock_get):
    mock_get.return_value = _make_response(404)
    adapter = FootballDataCoUkAdapter()
    with pytest.raises(DataSourceUnavailableError):
        adapter.fetch_historical_matches("E0", "9999")


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_retries_on_server_error_then_succeeds(mock_get, mock_sleep):
    mock_get.side_effect = [_make_response(503), _make_response(200, SAMPLE_CSV)]
    adapter = FootballDataCoUkAdapter(max_retries=2)

    records = adapter.fetch_historical_matches("E0", "2526")
    assert len(records) == 3


@patch("time.sleep", return_value=None)
@patch("requests.Session.get")
def test_raises_after_persistent_server_errors(mock_get, mock_sleep):
    mock_get.return_value = _make_response(500)
    adapter = FootballDataCoUkAdapter(max_retries=1)
    with pytest.raises(DataSourceUnavailableError):
        adapter.fetch_historical_matches("E0", "2526")
