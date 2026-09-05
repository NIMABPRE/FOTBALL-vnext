"""
The Odds API (https://the-odds-api.com) adapter — a live/near-live bookmaker
odds feed covering many sports and bookmakers, with a free tier suitable for
prototyping.

NOTE ON THIS ENVIRONMENT: like football_data_org.py, this module was
developed and unit-tested against MOCKED HTTP responses because this dev
sandbox's network egress does not include api.the-odds-api.com and no API
key was available to it. The retry/backoff/error-handling/parsing logic is
fully implemented and tested — what's untested is live connectivity. Run
scripts/verify_odds_api.py with a real key before trusting this live.

API docs: https://the-odds-api.com/liveapi/guides/v4/
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from football_vnext.domain.odds.models import BookmakerOdds, MatchOddsQuote
from football_vnext.infrastructure.data_sources.exceptions import (
    AuthenticationError,
    DataSourceUnavailableError,
    RateLimitExceededError,
)

logger = logging.getLogger(__name__)


class TheOddsApiAdapter:
    """
    :param api_key: The Odds API key (sent as a query parameter, per their API)
    :param sport_key: e.g. "soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a"
    :param regions: bookmaker region(s), e.g. "uk", "eu", "us" (comma-separated)
    :param markets: odds market(s); "h2h" is the 3-way match-result market this
        project uses
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.the-odds-api.com/v4",
        sport_key: str = "soccer_epl",
        regions: str = "uk",
        markets: str = "h2h",
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise AuthenticationError("An API key is required for TheOddsApiAdapter.")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_factor < 0:
            raise ValueError("backoff_factor must be >= 0")

        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.sport_key = sport_key
        self.regions = regions
        self.markets = markets
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._session = requests.Session()

    def fetch_odds(self) -> List[MatchOddsQuote]:
        """
        Fetch current odds for all upcoming matches in `self.sport_key`.
        Malformed individual events/bookmaker entries are skipped with a
        logged warning, same policy as FootballDataOrgAdapter — one bad
        record from the feed should not block everything else.
        """
        url = f"{self.base_url}/sports/{self.sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": "decimal",
        }
        payload = self._request_with_retry(url, params)

        quotes: List[MatchOddsQuote] = []
        skipped = 0
        for raw_event in payload:
            try:
                quote = self._parse_event(raw_event)
                if quote is not None:
                    quotes.append(quote)
            except (KeyError, ValueError, TypeError) as exc:
                skipped += 1
                event_id = raw_event.get("id", "<unknown>") if isinstance(raw_event, dict) else "<unknown>"
                logger.warning("Skipping malformed odds event id=%s: %s", event_id, exc)

        logger.info(
            "Fetched odds for %d events (sport=%s, regions=%s), %d skipped as malformed",
            len(quotes), self.sport_key, self.regions, skipped,
        )
        return quotes

    def _parse_event(self, raw_event: Dict[str, Any]) -> Optional[MatchOddsQuote]:
        home_team_name = raw_event["home_team"]
        away_team_name = raw_event["away_team"]
        commence_time = datetime.fromisoformat(raw_event["commence_time"].replace("Z", "+00:00"))

        bookmaker_quotes: List[BookmakerOdds] = []
        for bookmaker in raw_event.get("bookmakers", []):
            h2h_market = next(
                (m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None
            )
            if h2h_market is None:
                continue

            prices: Dict[str, float] = {}
            for outcome in h2h_market.get("outcomes", []):
                prices[outcome["name"]] = float(outcome["price"])

            if home_team_name not in prices or away_team_name not in prices or "Draw" not in prices:
                continue  # incomplete 3-way market from this bookmaker; skip it, not the whole event

            bookmaker_quotes.append(
                BookmakerOdds(
                    bookmaker=bookmaker.get("title", bookmaker.get("key", "unknown")),
                    home=prices[home_team_name],
                    draw=prices["Draw"],
                    away=prices[away_team_name],
                )
            )

        if not bookmaker_quotes:
            return None  # no bookmaker on this event had a usable 3-way market

        return MatchOddsQuote(
            home_team_name=home_team_name,
            away_team_name=away_team_name,
            commence_time=commence_time,
            bookmaker_quotes=bookmaker_quotes,
        )

    def _request_with_retry(self, url: str, params: Dict[str, Any]) -> Any:
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
                return response.json()

            if response.status_code in (401, 403):
                raise AuthenticationError(
                    f"The Odds API rejected the API key (HTTP {response.status_code})."
                )

            if response.status_code == 429:
                logger.warning(
                    "Rate limited (attempt %d/%d)", attempt + 1, self.max_retries + 1
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor * (2**attempt) + 1.0)
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
                    time.sleep(self.backoff_factor * (2**attempt))
                    continue
                raise DataSourceUnavailableError(
                    f"The Odds API returned {response.status_code} after "
                    f"{self.max_retries + 1} attempts."
                )

            raise DataSourceUnavailableError(
                f"Unexpected HTTP status {response.status_code}: {response.text[:200]}"
            )

        raise DataSourceUnavailableError(
            f"Failed to reach The Odds API after {self.max_retries + 1} attempts: {last_exception}"
        )
