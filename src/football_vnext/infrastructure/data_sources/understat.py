"""Small, dependency-light Understat league xG adapter.

Understat exposes the league match dataset in the HTML page as a JSON blob.
This adapter parses that public data and returns rows that can be merged with
football-data.co.uk Match objects.  It is deliberately best-effort: if the
provider changes its page structure, callers receive an empty list rather
than silently fabricated xG.
"""
from __future__ import annotations
import html
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
import requests

logger = logging.getLogger(__name__)

LEAGUE_MAP = {"E0": "EPL", "SP1": "La_liga", "I1": "Serie_A", "D1": "Bundesliga", "F1": "Ligue_1"}

class UnderstatAdapter:
    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 football-vnext/1.0"})

    def fetch_league_matches(self, league_code: str, season_start_year: int) -> list[dict[str, Any]]:
        league = LEAGUE_MAP.get(league_code, league_code)
        url = f"https://understat.com/league/{league}/{season_start_year}"
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Understat unavailable: %s", exc)
            return []
        # Current Understat pages use datesData = JSON.parse('...'). Keep a
        # second regex for minor formatting changes.
        patterns = [r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", r"datesData\s*=\s*JSON\.parse\(\"(.+?)\"\)"]
        encoded = None
        for pat in patterns:
            m = re.search(pat, r.text, re.S)
            if m:
                encoded = m.group(1)
                break
        if encoded is None:
            logger.warning("Understat league page did not contain datesData: %s", url)
            return []
        try:
            decoded = html.unescape(encoded).encode("utf-8").decode("unicode_escape")
            data = json.loads(decoded)
        except Exception:
            try:
                data = json.loads(html.unescape(encoded))
            except Exception as exc:
                logger.warning("Could not parse Understat datesData: %s", exc)
                return []
        rows: list[dict[str, Any]] = []
        for row in data if isinstance(data, list) else []:
            try:
                dt = datetime.fromtimestamp(int(row["datetime"]), tz=timezone.utc)
                rows.append({
                    "date": dt.date(),
                    "kickoff": dt,
                    "home_team": str(row["h"]["title"]),
                    "away_team": str(row["a"]["title"]),
                    "home_xg": float(row["xG"]["h"]),
                    "away_xg": float(row["xG"]["a"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
        logger.info("Understat returned %d xG matches for %s/%s", len(rows), league, season_start_year)
        return rows
