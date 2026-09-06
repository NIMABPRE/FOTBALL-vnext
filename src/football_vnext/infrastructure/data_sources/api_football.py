"""
API-Football (api-sports.io) adapter — a second, independent real data source
for both fixtures/results AND odds, used for redundancy alongside
football-data.org (fixtures) and The Odds API (odds).

Uses the direct api-sports.io host (header: x-apisports-key). If you signed
up via RapidAPI instead, switch base_url to
"https://api-football-v1.p.rapidapi.com/v3" and the header to
"X-RapidAPI-Key" (plus "X-RapidAPI-Host") — both are the same underlying API,
just different access routes; only the auth header/host differs.

NOTE ON THIS ENVIRONMENT: like the other adapters, tested against
constructed sample JSON matching the documented v3 schema, not a live call —
this dev sandbox's network egress does not include api-sports.io and no API
key was available to it. Run scripts/verify_api_football.py with a real key
before trusting it live.

Docs: https://www.api-football.com/documentation-v3
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from football_vnext.domain.models.match import Match, MatchResult, MatchStatus
from football_vnext.domain.odds.models import BookmakerOdds, MatchOddsQuote
from football_vnext.infrastructure.data_sources.exceptions import (
    AuthenticationError,
    DataSourceUnavailableError,
    RateLimitExceededError,
)

logger = logging.getLogger(__name__)

# API-Football fixture status codes -> our MatchStatus
_STATUS_MAP = {
    "TBD": MatchStatus.SCHEDULED, "NS": MatchStatus.SCHEDULED,
    "1H": MatchStatus.SCHEDULED, "HT": MatchStatus.SCHEDULED, "2H": MatchStatus.SCHEDULED,
    "ET": MatchStatus.SCHEDULED, "P": MatchStatus.SCHEDULED, "LIVE": MatchStatus.SCHEDULED,
    "FT": MatchStatus.FINISHED, "AET": MatchStatus.FINISHED, "PEN": MatchStatus.FINISHED,
    "PST": MatchStatus.POSTPONED, "SUSP": MatchStatus.POSTPONED, "INT": MatchStatus.POSTPONED,
    "CANC": MatchStatus.CANCELLED, "ABD": MatchStatus.CANCELLED, "AWD": MatchStatus.CANCELLED,
    "WO": MatchStatus.CANCELLED,
}

# "Match Winner" (1X2) market bet ID, per API-Football's documented bet list.
_MATCH_WINNER_BET_ID = 1


class ApiFootballAdapter:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://v3.football.api-sports.io",
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise AuthenticationError("An API key is required for ApiFootballAdapter.")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_factor < 0:
            raise ValueError("backoff_factor must be >= 0")

        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._session = requests.Session()
        self._session.headers.update({"x-apisports-key": self.api_key})

    def fetch_fixtures(self, league_id: int, season: int, date: Optional[str] = None) -> List[Match]:
        """Fetch fixtures for a league/season, optionally narrowed to one calendar date.

        :param league_id: API-Football numeric league ID (e.g. 39 = Premier League)
        :param season: e.g. 2024 for the 2024/25 season
        :param date: optional YYYY-MM-DD fixture date; preferred for daily slate loading
        """
        params = {"league": league_id, "season": season}
        if date:
            params["date"] = date
        payload = self.request_json(f"{self.base_url}/fixtures", params)
        raw_fixtures = payload.get("response", [])

        matches: List[Match] = []
        skipped = 0
        for raw in raw_fixtures:
            try:
                matches.append(self._parse_fixture(raw))
            except (KeyError, ValueError, TypeError) as exc:
                skipped += 1
                fid = raw.get("fixture", {}).get("id", "<unknown>") if isinstance(raw, dict) else "<unknown>"
                logger.warning("Skipping malformed fixture id=%s: %s", fid, exc)

        logger.info(
            "Fetched %d fixtures for league=%s season=%s (%d skipped as malformed)",
            len(matches), league_id, season, skipped,
        )
        return matches

    def fetch_odds_for_fixture(self, fixture_id: int) -> Optional[MatchOddsQuote]:
        """Fetch 1X2 (Match Winner) odds for a single fixture, across bookmakers."""
        payload = self.request_json(f"{self.base_url}/odds", {"fixture": fixture_id, "bet": _MATCH_WINNER_BET_ID})
        raw_response = payload.get("response", [])
        if not raw_response:
            return None

        try:
            return self._parse_odds(raw_response[0])
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed odds response for fixture=%s: %s", fixture_id, exc)
            return None

    def _parse_fixture(self, raw: Dict[str, Any]) -> Match:
        fixture_info = raw["fixture"]
        league_info = raw.get("league", {})
        teams = raw["teams"]
        goals = raw.get("goals", {})

        status_short = fixture_info.get("status", {}).get("short", "NS")
        status = _STATUS_MAP.get(status_short, MatchStatus.SCHEDULED)

        result: Optional[MatchResult] = None
        if status == MatchStatus.FINISHED and goals.get("home") is not None and goals.get("away") is not None:
            result = MatchResult(home_goals=int(goals["home"]), away_goals=int(goals["away"]))

        return Match(
            match_id=str(fixture_info["id"]),
            competition=league_info.get("name", "Unknown"),
            season=str(league_info.get("season", "Unknown")),
            matchday=None,  # API-Football uses a "round" string, not a simple int matchday
            kickoff=datetime.fromisoformat(fixture_info["date"]),
            home_team_id=str(teams["home"]["id"]),
            away_team_id=str(teams["away"]["id"]),
            home_team_name=teams["home"]["name"],
            away_team_name=teams["away"]["name"],
            status=status,
            result=result,
        )

    def _parse_odds(self, raw_event: Dict[str, Any]) -> Optional[MatchOddsQuote]:
        fixture_info = raw_event["fixture"]
        commence_time = datetime.fromisoformat(fixture_info["date"])

        bookmaker_quotes: List[BookmakerOdds] = []
        home_name = away_name = None
        for bookmaker in raw_event.get("bookmakers", []):
            match_winner_bet = next(
                (b for b in bookmaker.get("bets", []) if b.get("id") == _MATCH_WINNER_BET_ID), None
            )
            if match_winner_bet is None:
                continue

            prices: Dict[str, float] = {}
            for value in match_winner_bet.get("values", []):
                prices[value["value"]] = float(value["odd"])

            if "Home" not in prices or "Draw" not in prices or "Away" not in prices:
                continue

            bookmaker_quotes.append(
                BookmakerOdds(
                    bookmaker=bookmaker.get("name", "unknown"),
                    home=prices["Home"], draw=prices["Draw"], away=prices["Away"],
                )
            )

        if not bookmaker_quotes:
            return None

        # API-Football's odds response doesn't repeat team names at the top
        # level the way its fixtures response does -- team names aren't
        # needed here since this method is keyed by fixture_id, not matched
        # by name (unlike TheOddsApiAdapter). Placeholder names are fine
        # because callers already know which fixture they asked for.
        return MatchOddsQuote(
            home_team_name="(fixture-id-keyed)", away_team_name="(fixture-id-keyed)",
            commence_time=commence_time, bookmaker_quotes=bookmaker_quotes,
        )

    def request_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.RequestException as exc:
                last_exception = exc
                logger.warning(
                    "Request error on attempt %d/%d: %s", attempt + 1, self.max_retries + 1, exc
                )
                time.sleep(self.backoff_factor * (2**attempt))
                continue

            if response.status_code == 200:
                data = response.json()
                if data.get("errors"):
                    errors = data["errors"]
                    if isinstance(errors, dict) and ("token" in errors or "key" in str(errors).lower()):
                        raise AuthenticationError(f"API-Football rejected the request: {errors}")
                return data

            if response.status_code in (401, 403):
                raise AuthenticationError(
                    f"API-Football rejected the API key (HTTP {response.status_code})."
                )

            if response.status_code == 429:
                logger.warning("Rate limited (attempt %d/%d)", attempt + 1, self.max_retries + 1)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor * (2**attempt) + 1.0)
                    continue
                raise RateLimitExceededError(f"Rate limit exceeded after {self.max_retries + 1} attempts.")

            if 500 <= response.status_code < 600:
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor * (2**attempt))
                    continue
                raise DataSourceUnavailableError(
                    f"API-Football returned {response.status_code} after {self.max_retries + 1} attempts."
                )

            raise DataSourceUnavailableError(
                f"Unexpected HTTP status {response.status_code}: {response.text[:200]}"
            )

        raise DataSourceUnavailableError(
            f"Failed to reach API-Football after {self.max_retries + 1} attempts: {last_exception}"
        )

    def _request_with_retry(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compatible alias; new callers should use request_json()."""
        return self.request_json(url, params)
