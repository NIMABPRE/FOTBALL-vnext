from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from football_vnext.application.data_loading import load_training_matches
from football_vnext.config import Settings
from football_vnext.domain.models.match import Match, MatchResult
from football_vnext.infrastructure.data_sources.exceptions import DataSourceError


def _fake_finished_matches(n: int, competition: str = "Premier League") -> list[Match]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    matches = []
    for i in range(n):
        matches.append(
            Match(
                match_id=f"REAL-{i}",
                competition=competition,
                season="2025",
                matchday=i + 1,
                kickoff=start + timedelta(days=i),
                home_team_id="57",
                away_team_id="61",
                home_team_name="Arsenal FC",
                away_team_name="Chelsea FC",
                status="finished",
                result=MatchResult(home_goals=1, away_goals=1),
            )
        )
    return matches


def test_uses_synthetic_when_no_api_keys():
    settings = Settings()
    result = load_training_matches(settings=settings, n_synthetic_rounds=15)
    assert result.source == "synthetic"
    assert result.fallback_reason is None
    assert len(result.matches) > 0


@patch("football_vnext.application.data_loading.FootballDataOrgAdapter")
def test_uses_football_data_org_when_it_succeeds(mock_adapter_cls):
    mock_adapter_cls.return_value.fetch_matches.return_value = _fake_finished_matches(25)
    settings = Settings(football_data_api_key="test-key")

    result = load_training_matches(competition_code="PL", settings=settings)

    assert result.source == "football-data.org"
    assert result.competition_label == "Premier League"
    assert len(result.matches) == 25
    assert result.fallback_reason is None


@patch("football_vnext.application.data_loading.ApiFootballAdapter")
@patch("football_vnext.application.data_loading.FootballDataOrgAdapter")
def test_falls_back_to_api_football_when_football_data_org_fails(mock_fdo_cls, mock_af_cls):
    mock_fdo_cls.return_value.fetch_matches.side_effect = DataSourceError("boom")
    mock_af_cls.return_value.fetch_fixtures.return_value = _fake_finished_matches(25)
    settings = Settings(football_data_api_key="fd-key", api_football_key="af-key")

    result = load_training_matches(competition_code="PL", settings=settings)

    assert result.source == "api-football"
    assert len(result.matches) == 25
    assert result.fallback_reason is None


@patch("football_vnext.application.data_loading.ApiFootballAdapter")
@patch("football_vnext.application.data_loading.FootballDataOrgAdapter")
def test_falls_back_to_synthetic_when_all_real_sources_fail(mock_fdo_cls, mock_af_cls):
    mock_fdo_cls.return_value.fetch_matches.side_effect = DataSourceError("boom")
    mock_af_cls.return_value.fetch_fixtures.side_effect = DataSourceError("also boom")
    settings = Settings(football_data_api_key="fd-key", api_football_key="af-key")

    result = load_training_matches(competition_code="PL", settings=settings, n_synthetic_rounds=15)

    assert result.source == "synthetic"
    assert result.fallback_reason is not None
    assert len(result.matches) > 0


@patch("football_vnext.application.data_loading.FootballDataOrgAdapter")
def test_falls_back_to_synthetic_on_insufficient_finished_matches_with_only_fdo_configured(mock_adapter_cls):
    mock_adapter_cls.return_value.fetch_matches.return_value = _fake_finished_matches(5)
    settings = Settings(football_data_api_key="test-key")

    result = load_training_matches(settings=settings, n_synthetic_rounds=15)

    assert result.source == "synthetic"
    assert result.fallback_reason is not None


@patch("football_vnext.application.data_loading.ApiFootballAdapter")
def test_api_football_skipped_for_unmapped_competition_code(mock_af_cls):
    settings = Settings(api_football_key="af-key")
    result = load_training_matches(competition_code="UNKNOWN_LEAGUE", settings=settings, n_synthetic_rounds=15)

    mock_af_cls.return_value.fetch_fixtures.assert_not_called()
    assert result.source == "synthetic"
