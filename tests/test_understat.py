import json
from unittest.mock import MagicMock, patch
import requests

from football_vnext.infrastructure.data_sources.understat import UnderstatAdapter


def test_parses_dates_data_and_maps_league():
    payload = [{
        "datetime": 1754067600,
        "h": {"title": "Arsenal"}, "a": {"title": "Chelsea"},
        "xG": {"h": "1.72", "a": "0.64"},
    }]
    html = "<script>datesData = JSON.parse('" + json.dumps(payload) + "')</script>"
    response = MagicMock(status_code=200, text=html, content=html.encode())
    with patch("requests.Session.get", return_value=response):
        rows = UnderstatAdapter().fetch_league_matches("E0", 2025)
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Arsenal"
    assert rows[0]["home_xg"] == 1.72


def test_returns_empty_when_understat_has_no_dataset():
    response = MagicMock(status_code=200, text="<html>no data</html>", content=b"<html>no data</html>")
    with patch("requests.Session.get", return_value=response):
        assert UnderstatAdapter().fetch_league_matches("F1", 2025) == []


def test_returns_empty_on_network_failure():
    with patch("requests.Session.get", side_effect=requests.RequestException("offline")):
        assert UnderstatAdapter().fetch_league_matches("E0", 2025) == []
