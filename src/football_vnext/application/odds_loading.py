"""
Use case: get bookmaker odds for a specific match, from whichever real
source actually works.

The Odds API is tried first (team-name-matched — works for a match from ANY
fixture source). API-Football's odds endpoint is tried next, but ONLY when
the caller can supply `api_football_fixture_id` — that endpoint is keyed by
API-Football's own numeric fixture ID, which only exists if the match itself
came from API-Football in the first place (see data_loading.py). Guessing a
fixture ID for a match from a different source would risk silently attaching
the wrong match's odds, which is worse than not trying at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from football_vnext.config import Settings
from football_vnext.domain.models.match import Match
from football_vnext.domain.odds.aggregation import AggregationMethod, OddsAggregator
from football_vnext.domain.odds.matching import TeamNameMatcher
from football_vnext.domain.odds.models import BookmakerOdds
from football_vnext.infrastructure.data_sources.api_football import ApiFootballAdapter
from football_vnext.infrastructure.data_sources.exceptions import DataSourceError
from football_vnext.infrastructure.data_sources.odds_api import TheOddsApiAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedOdds:
    odds: BookmakerOdds
    source: str  # "the-odds-api" | "api-football" | "example"
    fallback_reason: Optional[str] = None


def _try_the_odds_api(match: Match, sport_key: str, settings: Settings) -> Optional[BookmakerOdds]:
    if not settings.has_odds_api_key:
        return None
    try:
        adapter = TheOddsApiAdapter(api_key=settings.odds_api_key, sport_key=sport_key)
        quotes = adapter.fetch_odds()
        found = TeamNameMatcher().find_match(match, quotes)
        if found is None:
            raise DataSourceError(
                f"No odds-feed quote matched {match.home_team_name} vs "
                f"{match.away_team_name} (kickoff {match.kickoff})."
            )
        aggregated = OddsAggregator(AggregationMethod.MEDIAN).aggregate(found.bookmaker_quotes)
        logger.info(
            "Loaded real odds for %s vs %s from %d bookmakers via The Odds API",
            match.home_team_name, match.away_team_name, len(found.bookmaker_quotes),
        )
        return aggregated
    except DataSourceError as exc:
        logger.warning("The Odds API lookup failed: %s", exc)
        return None


def _try_api_football_odds(
    api_football_fixture_id: Optional[int], settings: Settings
) -> Optional[BookmakerOdds]:
    if not settings.has_api_football_key or api_football_fixture_id is None:
        return None
    try:
        adapter = ApiFootballAdapter(api_key=settings.api_football_key)
        quote = adapter.fetch_odds_for_fixture(api_football_fixture_id)
        if quote is None:
            raise DataSourceError(f"No odds available from API-Football for fixture {api_football_fixture_id}.")
        aggregated = OddsAggregator(AggregationMethod.MEDIAN).aggregate(quote.bookmaker_quotes)
        logger.info(
            "Loaded real odds for fixture %s from %d bookmakers via API-Football",
            api_football_fixture_id, len(quote.bookmaker_quotes),
        )
        return aggregated
    except DataSourceError as exc:
        logger.warning("API-Football odds lookup failed: %s", exc)
        return None


def load_match_odds(
    match: Match,
    example_odds: BookmakerOdds,
    sport_key: str = "soccer_epl",
    api_football_fixture_id: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> LoadedOdds:
    settings = settings or Settings.from_env()

    odds = _try_the_odds_api(match, sport_key, settings)
    if odds is not None:
        return LoadedOdds(odds=odds, source="the-odds-api")

    odds = _try_api_football_odds(api_football_fixture_id, settings)
    if odds is not None:
        return LoadedOdds(odds=odds, source="api-football")

    reason = None
    if settings.has_odds_api_key or (settings.has_api_football_key and api_football_fixture_id is not None):
        reason = "all configured real odds sources failed or had no matching quote"
    else:
        logger.info("No real odds source available for this match -- using example odds.")

    return LoadedOdds(odds=example_odds, source="example", fallback_reason=reason)
