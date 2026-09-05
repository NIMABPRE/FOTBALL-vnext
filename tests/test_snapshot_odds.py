import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from scripts import snapshot_odds


def test_ligue_one_uses_real_odds_api_sport_key():
    assert snapshot_odds.SPORTS["Ligue 1"] == "soccer_france_ligue_one"


def test_snapshot_job_persists_each_quote():
    quote = MagicMock()
    quote.home_team_name = "A"
    quote.away_team_name = "B"
    quote.commence_time = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
    q = MagicMock(bookmaker="Book", home=2.0, draw=3.2, away=4.0)
    quote.bookmaker_quotes = [q]
    fake_adapter = MagicMock()
    fake_adapter.fetch_odds.return_value = [quote]
    fake_store = MagicMock()
    with patch.object(snapshot_odds, "TheOddsApiAdapter", return_value=fake_adapter), \
         patch.object(snapshot_odds, "PredictionStore", return_value=fake_store), \
         patch.object(snapshot_odds, "Settings") as settings_cls:
        settings_cls.from_env.return_value = MagicMock(odds_api_key="key")
        snapshot_odds.main([])
    fake_store.save_odds_snapshot.assert_called_once()
    args = fake_store.save_odds_snapshot.call_args.args
    assert args[3] == "Premier League"
    assert args[6]["Book"]["home"] == 2.0
