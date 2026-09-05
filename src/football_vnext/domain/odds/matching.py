"""
Matching odds-feed quotes to fixtures.

Different data providers use different identifiers for the same real-world
match: football-data.org gives each team a numeric ID; an odds feed (e.g.
The Odds API) only gives team name strings. There is no shared key, so
matches have to be reconciled by (normalized team name + kickoff time
proximity) — this is inherently fuzzy and is a well-known integration
headache in sports data, not a corner this module is cutting.

This matcher is deliberately conservative: it requires BOTH team names to
match (after normalization) AND kickoff times within a tolerance window,
and returns None rather than a low-confidence guess when it can't find a
confident match. A wrong match here would silently corrupt the whole
downstream pipeline (wrong odds attached to a match), which is worse than
no match at all.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import List, Optional

from football_vnext.domain.models.match import Match
from football_vnext.domain.odds.models import MatchOddsQuote

# Common suffixes/prefixes that differ between providers for the same club
# (e.g. football-data.org's "Arsenal FC" vs an odds feed's "Arsenal").
_NOISE_WORDS = {"fc", "cf", "afc", "sc", "ac", "cd", "the"}


def _normalize_team_name(name: str) -> str:
    lowered = name.lower()
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    words = [w for w in lowered.split() if w not in _NOISE_WORDS]
    return " ".join(words).strip()


class TeamNameMatcher:
    def __init__(self, kickoff_tolerance: timedelta = timedelta(hours=12)) -> None:
        if kickoff_tolerance.total_seconds() < 0:
            raise ValueError("kickoff_tolerance must be non-negative")
        self.kickoff_tolerance = kickoff_tolerance

    def find_match(
        self, match: Match, odds_quotes: List[MatchOddsQuote]
    ) -> Optional[MatchOddsQuote]:
        target_home = _normalize_team_name(match.home_team_name)
        target_away = _normalize_team_name(match.away_team_name)

        for quote in odds_quotes:
            quote_home = _normalize_team_name(quote.home_team_name)
            quote_away = _normalize_team_name(quote.away_team_name)

            names_match = target_home == quote_home and target_away == quote_away
            if not names_match:
                continue

            time_diff = abs(match.kickoff - quote.commence_time)
            if time_diff <= self.kickoff_tolerance:
                return quote

        return None
