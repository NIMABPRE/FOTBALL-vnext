from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from football_vnext.domain.models.match import Match
from football_vnext.domain.odds.models import BookmakerOdds


@dataclass(frozen=True)
class HistoricalMatchOdds:
    """
    A settled historical match paired with the odds that were actually
    available around it — this is the real thing `SyntheticOddsGenerator`
    was standing in for. `closing_odds` is None for older seasons/sources
    that never recorded a closing line; callers doing CLV analysis must
    skip records where it's None (there is nothing to compute CLV against).
    """

    match: Match
    opening_odds: Optional[BookmakerOdds]
    closing_odds: Optional[BookmakerOdds]
