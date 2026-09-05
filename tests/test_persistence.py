import json
import sqlite3
from datetime import datetime, timezone, timedelta

from football_vnext.infrastructure.persistence import PredictionStore


def test_prediction_round_trip_and_unsettled(tmp_path):
    db = tmp_path / "test.db"
    store = PredictionStore(f"sqlite:///{db}")
    kickoff = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
    store.save_prediction("p1", kickoff, "EPL", "A", "B", {"edge": 0.05})
    rows = store.list_recent()
    assert len(rows) == 1
    assert json.loads(rows[0][6])["edge"] == 0.05
    assert len(store.list_unsettled()) == 1


def test_latest_odds_before_uses_match_kickoff_as_cutoff(tmp_path):
    db = tmp_path / "test.db"
    store = PredictionStore(f"sqlite:///{db}")
    kickoff = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
    before = kickoff - timedelta(minutes=5)
    after = kickoff + timedelta(minutes=5)
    store.save_odds_snapshot("before", before, kickoff, "EPL", "A", "B", {"home": 2.0})
    store.save_odds_snapshot("after", after, kickoff, "EPL", "A", "B", {"home": 1.5})
    captured_at, odds = store.latest_odds_before("EPL", "A", "B", kickoff)
    assert captured_at == before.isoformat()
    assert json.loads(odds)["home"] == 2.0


def test_failed_write_does_not_poison_store(tmp_path):
    db = tmp_path / "test.db"
    store = PredictionStore(f"sqlite:///{db}")
    from datetime import datetime, timezone
    try:
        store.save_prediction("bad", datetime.now(timezone.utc), "EPL", "A", "B", {"bad": {1}})
    except TypeError:
        pass
    # The failed transaction must not leave the connection open/locked.
    store.save_prediction("good", datetime.now(timezone.utc), "EPL", "A", "B", {"ok": True})
    assert len(store.list_recent()) == 1
