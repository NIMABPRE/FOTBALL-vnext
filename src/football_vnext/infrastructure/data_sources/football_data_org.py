"""
football-data.org (v4) data source adapter.

NOTE ON THIS ENVIRONMENT: this module was developed and unit-tested against
MOCKED HTTP responses (see tests/test_football_data_org.py) because this
development sandbox's network egress does not include api.football-data.org
and no API key was available to it. The retry/backoff/error-handling logic
itself is fully implemented and tested — what's untested is live connectivity
to the real API. Before relying on this in production: run it once against
the real API with a real key (https://www.football-data.org/client/register)
and confirm a live fetch succeeds.

API docs: https://docs.football-data.org/general/v4/index.html
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from football_vnext.domain.models.match import Match, MatchResult, MatchStatus
from football_vnext.infrastructure.data_sources.exceptions import (
    AuthenticationError,
    DataSourceUnavailableError,
    RateLimitExceededError,
)

logger = logging.getLogger(__name__)

_STATUS_MAP = {
    "SCHEDULED": MatchStatus.SCHEDULED,
    "TIMED": MatchStatus.SCHEDULED,
    "IN_PLAY": MatchStatus.SCHEDULED,
    "PAUSED": MatchStatus.SCHEDULED,
    "FINISHED": MatchStatus.FINISHED,
    "POSTPONED": MatchStatus.POSTPONED,
    "SUSPENDED": MatchStatus.POSTPONED,
    "CANCELLED": MatchStatus.CANCELLED,
}


class FootballDataOrgAdapter:
    """
    Fetches match data from football-data.org's v4 API and maps it to the
    domain `Match` model.

    :param api_key: football-data.org API token (sent as X-Auth-Token header)
    :param base_url: API base URL, overridable for testing
    :param timeout: per-request timeout in seconds
    :param max_retries: retries for rate-limit (429) and server errors (5xx)
    :param backoff_factor: base seconds for exponential backoff between retries
        (actual wait = backoff_factor * 2**attempt, capped by Retry-After if present)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.football-data.org/v4",
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise AuthenticationError("An API key is required for FootballDataOrgAdapter.")
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
        self._session.headers.update({"X-Auth-Token": self.api_key})

    def fetch_matches(
        self, competition_code: str, season: Optional[int] = None
    ) -> List[Match]:
        """
        Fetch all matches for a competition (e.g. "PL", "SA", "BL1") and,
        optionally, a specific season (e.g. 2024 for the 2024/25 season).
        Malformed individual match records are skipped with a logged warning
        rather than aborting the whole fetch — a partial result is more
        useful than none, and one bad record from the API should not block
        the other 379.
        """
        if not competition_code or not competition_code.strip():
            raise ValueError("competition_code cannot be empty")

        params: Dict[str, Any] = {}
        if season is not None:
            params["season"] = season

        url = f"{self.base_url}/competitions/{competition_code.strip()}/matches"
        payload = self._request_with_retry(url, params)

        raw_matches = payload.get("matches", [])
        matches: List[Match] = []
        skipped = 0

        for raw in raw_matches:
            try:
                matches.append(self._parse_match(raw))
            except (KeyError, ValueError, TypeError) as exc:
                skipped += 1
                match_id = raw.get("id", "<unknown>") if isinstance(raw, dict) else "<unknown>"
                logger.warning("Skipping malformed match record id=%s: %s", match_id, exc)

        logger.info(
            "Fetched %d matches for competition=%s season=%s (%d skipped as malformed)",
            len(matches), competition_code, season, skipped,
        )
        return matches

    def _parse_match(self, raw: Dict[str, Any]) -> Match:
        home_team = raw["homeTeam"]
        away_team = raw["awayTeam"]
        status_raw = raw.get("status", "SCHEDULED")
        status = _STATUS_MAP.get(status_raw, MatchStatus.SCHEDULED)

        result: Optional[MatchResult] = None
        if status == MatchStatus.FINISHED:
            full_time = raw.get("score", {}).get("fullTime", {})
            home_goals = full_time.get("home")
            away_goals = full_time.get("away")
            if home_goals is not None and away_goals is not None:
                result = MatchResult(home_goals=int(home_goals), away_goals=int(away_goals))

        return Match(
            match_id=str(raw["id"]),
            competition=raw.get("competition", {}).get("name", "Unknown"),
            season=str(raw.get("season", {}).get("startDate", "Unknown"))[:4],
            matchday=raw.get("matchday"),
            kickoff=datetime.fromisoformat(raw["utcDate"].replace("Z", "+00:00")),
            home_team_id=str(home_team["id"]),
            away_team_id=str(away_team["id"]),
            home_team_name=home_team["name"],
            away_team_name=away_team["name"],
            status=status,
            result=result,
        )

    def _request_with_retry(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.RequestException as exc:
                last_exception = exc
                logger.warning(
                    "Request error on attempt %d/%d: %s", attempt + 1, self.max_retries + 1, exc
                )
                self._sleep_before_retry(attempt, retry_after=None)
                continue

            if response.status_code == 200:
                return response.json()

            if response.status_code in (401, 403):
                raise AuthenticationError(
                    f"football-data.org rejected the API key (HTTP {response.status_code})."
                )

            if response.status_code == 429:
                retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
                logger.warning(
                    "Rate limited (attempt %d/%d), waiting %.1fs before retry",
                    attempt + 1, self.max_retries + 1, retry_after,
                )
                if attempt < self.max_retries:
                    time.sleep(retry_after)
                    continue
                raise RateLimitExceededError(
                    f"Rate limit exceeded after {self.max_retries + 1} attempts."
                )

            if 500 <= response.status_code < 600:
                logger.warning(
                    "Server error %d on attempt %d/%d",
                    response.status_code, attempt + 1, self.max_retries + 1,
                )
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt, retry_after=None)
                    continue
                raise DataSourceUnavailableError(
                    f"football-data.org returned {response.status_code} after "
                    f"{self.max_retries + 1} attempts."
                )

            # Any other unexpected status code: fail fast, don't retry blindly.
            raise DataSourceUnavailableError(
                f"Unexpected HTTP status {response.status_code}: {response.text[:200]}"
            )

        # Only reachable if every attempt raised a RequestException.
        raise DataSourceUnavailableError(
            f"Failed to reach football-data.org after {self.max_retries + 1} attempts: "
            f"{last_exception}"
        )

    def _sleep_before_retry(self, attempt: int, retry_after: Optional[float]) -> None:
        if retry_after is not None:
            time.sleep(retry_after)
        else:
            time.sleep(self.backoff_factor * (2**attempt))

    @staticmethod
    def _parse_retry_after(value: Optional[str]) -> float:
        if value is None:
            return 5.0
        try:
            return max(float(value), 0.0)
        except ValueError:
            return 5.0
