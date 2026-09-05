"""
Use case: load historical matches for model fitting, from whichever real
source actually works.

Tries sources in order and falls back to the next one (never silently to
synthetic while a real source might still work) — this is the redundancy
the project explicitly wanted: if football-data.org is down, rate-limited,
or doesn't have a key configured, API-Football is tried next before finally
falling back to synthetic data. Every step is logged so it's always clear
which source actually produced the data in front of you.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from football_vnext.config import Settings
from football_vnext.domain.models.match import Match
from football_vnext.infrastructure.data_sources.api_football import ApiFootballAdapter
from football_vnext.infrastructure.data_sources.exceptions import DataSourceError
from football_vnext.infrastructure.data_sources.football_data_org import FootballDataOrgAdapter
from football_vnext.sample_data import generate_sample_matches

logger = logging.getLogger(__name__)

MIN_FINISHED_MATCHES = 20  # Dixon-Coles' own hard minimum

# Maps a common competition code to each provider's own identifier scheme.
# Illustrative starter set -- extend as needed. football-data.org uses short
# codes; API-Football uses a numeric league ID that must be paired with a
# season year.
_LEAGUE_REGISTRY = {
    "PL": {"football_data_org_code": "PL", "api_football_league_id": 39},
    "PD": {"football_data_org_code": "PD", "api_football_league_id": 140},  # La Liga
    "SA": {"football_data_org_code": "SA", "api_football_league_id": 135},  # Serie A
    "BL1": {"football_data_org_code": "BL1", "api_football_league_id": 78},  # Bundesliga
    "FL1": {"football_data_org_code": "FL1", "api_football_league_id": 61},  # Ligue 1
}


@dataclass(frozen=True)
class LoadedMatches:
    matches: List[Match]
    source: str  # "football-data.org" | "api-football" | "synthetic"
    competition_label: str
    fallback_reason: Optional[str] = None


def _current_season_year() -> int:
    """European football seasons span two calendar years; API-Football's
    `season` param wants the year the season STARTED in (e.g. 2024 for
    2024/25). Before July, still count as the previous season's year."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _try_football_data_org(
    competition_code: str, settings: Settings
) -> Optional[LoadedMatches]:
    if not settings.has_football_data_api_key:
        return None
    try:
        adapter = FootballDataOrgAdapter(api_key=settings.football_data_api_key)
        matches = adapter.fetch_matches(competition_code)
        finished = [m for m in matches if m.result is not None]
        if len(finished) < MIN_FINISHED_MATCHES:
            raise DataSourceError(
                f"Only {len(finished)} finished matches from football-data.org "
                f"for '{competition_code}'; need at least {MIN_FINISHED_MATCHES}."
            )
        label = matches[0].competition if matches else competition_code
        logger.info(
            "Loaded %d matches (%d finished) from football-data.org [%s]",
            len(matches), len(finished), competition_code,
        )
        return LoadedMatches(matches=matches, source="football-data.org", competition_label=label)
    except DataSourceError as exc:
        logger.warning("football-data.org fetch failed for '%s': %s", competition_code, exc)
        return None


def _try_api_football(competition_code: str, settings: Settings) -> Optional[LoadedMatches]:
    if not settings.has_api_football_key:
        return None
    league_info = _LEAGUE_REGISTRY.get(competition_code)
    if league_info is None:
        logger.info(
            "No API-Football league_id mapping for '%s' -- add one to "
            "_LEAGUE_REGISTRY to enable this fallback for it.", competition_code,
        )
        return None
    try:
        adapter = ApiFootballAdapter(api_key=settings.api_football_key)
        season = _current_season_year()
        matches = adapter.fetch_fixtures(league_info["api_football_league_id"], season)
        finished = [m for m in matches if m.result is not None]
        if len(finished) < MIN_FINISHED_MATCHES:
            raise DataSourceError(
                f"Only {len(finished)} finished matches from API-Football for "
                f"'{competition_code}' season {season}; need at least {MIN_FINISHED_MATCHES}."
            )
        label = matches[0].competition if matches else competition_code
        logger.info(
            "Loaded %d matches (%d finished) from API-Football [%s season %d]",
            len(matches), len(finished), competition_code, season,
        )
        return LoadedMatches(matches=matches, source="api-football", competition_label=label)
    except DataSourceError as exc:
        logger.warning("API-Football fetch failed for '%s': %s", competition_code, exc)
        return None


def load_training_matches(
    competition_code: str = "PL",
    n_synthetic_rounds: int = 20,
    settings: Optional[Settings] = None,
) -> LoadedMatches:
    settings = settings or Settings.from_env()

    result = _try_football_data_org(competition_code, settings)
    if result is not None:
        return result

    result = _try_api_football(competition_code, settings)
    if result is not None:
        return result

    reason = None
    if settings.has_football_data_api_key or settings.has_api_football_key:
        reason = "all configured real data sources failed or returned insufficient data"
    else:
        logger.info("No real data source API key configured -- using synthetic sample data.")

    synthetic = generate_sample_matches(n_rounds=n_synthetic_rounds)
    return LoadedMatches(
        matches=synthetic, source="synthetic", competition_label="Sample League", fallback_reason=reason,
    )
