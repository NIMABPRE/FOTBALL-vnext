"""API-Football structured injuries/lineups adapter."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Any
from .api_football import ApiFootballAdapter

@dataclass(frozen=True)
class PlayerAbsence:
    player_name: str
    reason: str
    position: str = ""

@dataclass(frozen=True)
class TeamAvailability:
    team_id: int
    absences: tuple[PlayerAbsence, ...]
    starters_confirmed: bool = False
    starters_missing: int = 0

class TeamContextAdapter:
    def __init__(self, api: ApiFootballAdapter) -> None:
        self.api = api

    def fetch_team_ids(self, league_id: int, season: int) -> dict[str, int]:
        data = self.api.request_json(f"{self.api.base_url}/teams", {"league": league_id, "season": season})
        out={}
        for row in data.get("response", []):
            try: out[str(row["team"]["name"])] = int(row["team"]["id"])
            except (KeyError, TypeError, ValueError): continue
        return out

    def fetch_injuries(self, team_id: int, league_id: int, season: int) -> TeamAvailability:
        data = self.api.request_json(f"{self.api.base_url}/injuries", {"team": team_id, "league": league_id, "season": season})
        absences=[]
        for row in data.get("response", []):
            player=row.get("player", {}) or {}
            typ=(row.get("player", {}) or {}).get("type") or row.get("type") or "injury/suspension"
            absences.append(PlayerAbsence(str(player.get("name", "unknown")), str(typ), str((player.get("position") or ""))))
        return TeamAvailability(team_id, tuple(absences))

    def find_fixture_id(self, league_id: int, season: int, target_date: date, home: str, away: str) -> int | None:
        data = self.api.request_json(f"{self.api.base_url}/fixtures", {"league": league_id, "season": season, "date": target_date.isoformat()})
        hn, an = home.lower(), away.lower()
        for row in data.get("response", []):
            teams=row.get("teams", {})
            h=str((teams.get("home") or {}).get("name", "")).lower()
            a=str((teams.get("away") or {}).get("name", "")).lower()
            if h == hn and a == an or (hn in h and an in a):
                try: return int(row["fixture"]["id"])
                except (KeyError, TypeError, ValueError): pass
        return None

    def fetch_lineups(self, fixture_id: int) -> dict[int, list[str]]:
        data = self.api.request_json(f"{self.api.base_url}/fixtures/lineups", {"fixture": fixture_id})
        out={}
        for row in data.get("response", []):
            team_id=(row.get("team") or {}).get("id")
            if not team_id: continue
            starters=[str(x.get("player", {}).get("name", "")) for x in row.get("startXI", []) if x.get("player")]
            out[int(team_id)] = [x for x in starters if x]
        return out
