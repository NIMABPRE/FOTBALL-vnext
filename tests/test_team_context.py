from datetime import date
from unittest.mock import MagicMock

from football_vnext.infrastructure.data_sources.team_context import TeamContextAdapter


def test_uses_public_api_request_method():
    api = MagicMock()
    api.base_url = "https://api.example"
    api.request_json.return_value = {"response": [{"team": {"id": 1, "name": "Arsenal"}}]}
    out = TeamContextAdapter(api).fetch_team_ids(39, 2025)
    assert out == {"Arsenal": 1}
    api.request_json.assert_called_once_with(
        "https://api.example/teams", {"league": 39, "season": 2025}
    )


def test_parses_injuries_and_lineups():
    api = MagicMock(base_url="https://api.example")
    api.request_json.side_effect = [
        {"response": [{"player": {"name": "Player A", "type": "Missing", "position": "Defender"}}]},
        {"response": [{"team": {"id": 1}, "startXI": [{"player": {"name": "Starter"}}]}]},
    ]
    adapter = TeamContextAdapter(api)
    availability = adapter.fetch_injuries(1, 39, 2025)
    lineups = adapter.fetch_lineups(99)
    assert availability.absences[0].player_name == "Player A"
    assert lineups == {1: ["Starter"]}


def test_finds_fixture_by_date_and_team_names():
    api = MagicMock(base_url="https://api.example")
    api.request_json.return_value = {"response": [{
        "fixture": {"id": 123},
        "teams": {"home": {"name": "Arsenal FC"}, "away": {"name": "Chelsea"}},
    }]}
    result = TeamContextAdapter(api).find_fixture_id(39, 2025, date(2025, 8, 17), "Arsenal", "Chelsea")
    assert result == 123
