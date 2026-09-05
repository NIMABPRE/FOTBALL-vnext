from __future__ import annotations

from football_vnext.config import Settings


def test_from_env_reads_all_three_keys(monkeypatch):
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "  fd-key  ")
    monkeypatch.setenv("ODDS_API_KEY", "odds-key")
    monkeypatch.setenv("API_FOOTBALL_KEY", "")
    settings = Settings.from_env()

    assert settings.football_data_api_key == "fd-key"
    assert settings.odds_api_key == "odds-key"
    assert settings.api_football_key is None
    assert settings.has_football_data_api_key is True
    assert settings.has_odds_api_key is True
    assert settings.has_api_football_key is False


def test_from_env_defaults_to_none_when_unset(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    settings = Settings.from_env()

    assert settings.football_data_api_key is None
    assert settings.has_football_data_api_key is False


def test_manual_construction_defaults_are_none():
    settings = Settings()
    assert settings.football_data_api_key is None
    assert settings.odds_api_key is None
    assert settings.api_football_key is None
