from datetime import date
from unittest.mock import patch

import pytest

from football_vnext.application.daily_pipeline import _season_start_year, enrich_xg, run_daily
from football_vnext.infrastructure.data_sources.exceptions import DataSourceUnavailableError


def test_season_labels_convert_to_understat_start_year():
    assert _season_start_year("2526") == 2025
    assert _season_start_year("2425") == 2024
    assert _season_start_year("2025/26") == 2025
    assert _season_start_year("unknown") is None


def test_enrich_xg_uses_season_instead_of_unknown(monkeypatch):
    class FakeUA:
        def fetch_league_matches(self, code, year):
            assert code == "E0"
            assert year == 2025
            return [{"date": date(2025, 8, 2), "home_team": "Arsenal", "away_team": "Chelsea", "home_xg": 1.7, "away_xg": .6}]
    from football_vnext.application import daily_pipeline
    monkeypatch.setattr(daily_pipeline, "UnderstatAdapter", FakeUA)
    class R:
        def __init__(self):
            from football_vnext.domain.models.match import Match, MatchResult
            from datetime import datetime, timezone
            self.match=Match(match_id="1",competition="EPL",season="2526",kickoff=datetime(2025,8,2,tzinfo=timezone.utc),home_team_id="A",away_team_id="C",home_team_name="Arsenal",away_team_name="Chelsea",status="finished",result=MatchResult(home_goals=1,away_goals=0))
    records, count = enrich_xg([R()], "E0")
    assert count == 1
    assert records[0].match.home_xg == 1.7


def test_run_daily_reports_source_failure_instead_of_generic_zero_fit():
    class FakeHist:
        def __init__(self, *args, **kwargs): pass
        def fetch_historical_matches(self, *args, **kwargs): raise DataSourceUnavailableError("network unavailable")
    from football_vnext.application import daily_pipeline
    monkeypatch = __import__('pytest').MonkeyPatch()
    monkeypatch.setattr(daily_pipeline, "FootballDataCoUkAdapter", FakeHist)
    try:
        from football_vnext.config import Settings
        with pytest.raises(Exception, match="historical source returned no usable settled matches|network unavailable"):
            run_daily(Settings(odds_api_key="key"), "Premier League", date(2026, 9, 4), ("2526",))
    finally:
        monkeypatch.undo()
