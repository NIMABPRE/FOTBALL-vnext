"""
football-data.co.uk adapter — a free, public archive of historical football
results AND bookmaker odds (opening and, for recent seasons, closing lines
too) distributed as downloadable CSV files, explicitly intended for public
research/analysis use (see https://www.football-data.co.uk/notes.txt).

This is NOT the same site as football-data.org (the fixtures/results API
used elsewhere in this project) — different site, different data, similar
name, genuinely confusing, worth double-checking if something looks wrong.

THIS IS THE ANSWER to "no real historical odds archive exists" flagged
elsewhere in this project: for the leagues and seasons it covers, this gives
real opening odds (via the "Avg*" or "B365*" columns) and, for many seasons
since ~2019, real closing odds (via "AvgC*" or "PSC*" — Pinnacle closing,
often used as a sharp-market proxy) — enough to compute REAL CLV instead of
`SyntheticOddsGenerator`'s simulated version.

NOTE ON THIS ENVIRONMENT: like the other adapters, this was developed and
tested against embedded sample CSV text, not a live download — this dev
sandbox's network egress does not include football-data.co.uk. Run
scripts/verify_football_data_co_uk.py yourself to confirm live download works
(no API key needed — the data is public).

Column reference: https://www.football-data.co.uk/notes.txt
URL pattern: https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv
  season examples: "2324" = 2023/24, "2425" = 2024/25
  league_code examples: E0=Premier League, E1=Championship, SP1=La Liga,
  I1=Serie A, D1=Bundesliga, F1=Ligue 1
"""

from __future__ import annotations

import io
import logging
import time
from datetime import timezone
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from football_vnext.domain.backtest.historical_odds import HistoricalMatchOdds
from football_vnext.domain.models.match import Match, MatchResult, MatchStatus
from football_vnext.domain.odds.models import BookmakerOdds
from football_vnext.infrastructure.data_sources.exceptions import (
    DataSourceUnavailableError,
)

logger = logging.getLogger(__name__)

# Preference order: try the more informative/complete column set first, fall
# back to columns available further back in history. Bet365 (B365) has the
# longest unbroken column history on this site, which is why it's the
# opening-odds fallback rather than the primary choice.
_OPENING_COLUMN_GROUPS: Sequence[Tuple[str, str, str]] = (
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
)
_CLOSING_COLUMN_GROUPS: Sequence[Tuple[str, str, str]] = (
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSCH", "PSCD", "PSCA"),
)


class FootballDataCoUkAdapter:
    def __init__(
        self,
        base_url: str = "https://www.football-data.co.uk/mmz4281",
        timeout: float = 15.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_factor < 0:
            raise ValueError("backoff_factor must be >= 0")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._session = requests.Session()

    def fetch_historical_matches(
        self, league_code: str, season: str, competition_name: Optional[str] = None
    ) -> List[HistoricalMatchOdds]:
        """
        :param league_code: e.g. "E0" (Premier League), "SP1" (La Liga)
        :param season: e.g. "2324" for the 2023/24 season
        :param competition_name: display name; defaults to league_code if omitted
        """
        if not league_code or not league_code.strip():
            raise ValueError("league_code cannot be empty")
        if not season or not season.strip():
            raise ValueError("season cannot be empty")

        url = f"{self.base_url}/{season.strip()}/{league_code.strip()}.csv"
        csv_text = self._download_with_retry(url)
        return self.parse_csv(csv_text, competition_name=competition_name or league_code, season=season.strip())

    def parse_csv(self, csv_text: str, competition_name: str, season: Optional[str] = None) -> List[HistoricalMatchOdds]:
        try:
            df = pd.read_csv(io.StringIO(csv_text), on_bad_lines="skip", engine="python")
        except Exception as exc:  # pandas can raise a variety of parser errors
            raise DataSourceUnavailableError(f"Failed to parse CSV: {exc}") from exc

        records: List[HistoricalMatchOdds] = []
        skipped = 0
        for _, row in df.iterrows():
            try:
                records.append(self._parse_row(row, competition_name, season=season))
            except (KeyError, ValueError, TypeError) as exc:
                skipped += 1
                logger.warning("Skipping malformed row: %s", exc)

        logger.info(
            "Parsed %d matches for %s (%d rows skipped as malformed)",
            len(records), competition_name, skipped,
        )
        return records

    def _parse_row(self, row: "pd.Series[Any]", competition_name: str, season: Optional[str] = None) -> HistoricalMatchOdds:
        date = pd.to_datetime(row["Date"], dayfirst=True)
        if pd.isna(date):
            raise ValueError(f"Unparseable date: {row.get('Date')}")

        home_name = str(row["HomeTeam"]).strip()
        away_name = str(row["AwayTeam"]).strip()
        if not home_name or not away_name or home_name == "nan" or away_name == "nan":
            raise ValueError("Missing team name")

        result: Optional[MatchResult] = None
        home_goals, away_goals = row.get("FTHG"), row.get("FTAG")
        if pd.notna(home_goals) and pd.notna(away_goals):
            result = MatchResult(home_goals=int(home_goals), away_goals=int(away_goals))

        kickoff = date.to_pydatetime().replace(tzinfo=timezone.utc)
        match = Match(
            match_id=f"{competition_name}-{date.date()}-{home_name}-{away_name}",
            competition=competition_name,
            season=(season.strip() if season else "unknown"),
            kickoff=kickoff,
            home_team_id=home_name,  # no numeric team IDs on this site -- name doubles as ID, same
            away_team_id=away_name,  # convention used by sample_data.py for synthetic matches
            home_team_name=home_name,
            away_team_name=away_name,
            status=MatchStatus.FINISHED if result else MatchStatus.SCHEDULED,
            result=result,
        )

        opening_odds = self._extract_odds(row, _OPENING_COLUMN_GROUPS)
        closing_odds = self._extract_odds(row, _CLOSING_COLUMN_GROUPS)
        return HistoricalMatchOdds(match=match, opening_odds=opening_odds, closing_odds=closing_odds)

    @staticmethod
    def _extract_odds(
        row: "pd.Series[Any]", column_groups: Sequence[Tuple[str, str, str]]
    ) -> Optional[BookmakerOdds]:
        for h_col, d_col, a_col in column_groups:
            if h_col not in row or d_col not in row or a_col not in row:
                continue
            h, d, a = row[h_col], row[d_col], row[a_col]
            if pd.isna(h) or pd.isna(d) or pd.isna(a):
                continue
            try:
                return BookmakerOdds(bookmaker=h_col[:-1], home=float(h), draw=float(d), away=float(a))
            except (ValueError, TypeError):
                continue  # this column group had garbage data; try the next fallback
        return None

    def _download_with_retry(self, url: str) -> str:
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.get(url, timeout=self.timeout)
            except requests.exceptions.RequestException as exc:
                last_exception = exc
                logger.warning(
                    "Request error on attempt %d/%d: %s", attempt + 1, self.max_retries + 1, exc
                )
                time.sleep(self.backoff_factor * (2**attempt))
                continue

            if response.status_code == 200:
                return response.text

            if response.status_code == 404:
                raise DataSourceUnavailableError(
                    f"No data found at {url} (404) — check the league_code/season."
                )

            if 500 <= response.status_code < 600 and attempt < self.max_retries:
                logger.warning(
                    "Server error %d on attempt %d/%d",
                    response.status_code, attempt + 1, self.max_retries + 1,
                )
                time.sleep(self.backoff_factor * (2**attempt))
                continue

            raise DataSourceUnavailableError(
                f"Unexpected HTTP status {response.status_code} fetching {url}"
            )

        raise DataSourceUnavailableError(
            f"Failed to download {url} after {self.max_retries + 1} attempts: {last_exception}"
        )
