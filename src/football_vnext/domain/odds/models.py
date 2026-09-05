from __future__ import annotations

from datetime import datetime
from typing import List

import numpy as np
from pydantic import BaseModel, Field, field_validator


class BookmakerOdds(BaseModel):
    """Decimal odds for a single bookmaker on a single 3-way match market."""

    bookmaker: str
    home: float = Field(..., gt=1.0)
    draw: float = Field(..., gt=1.0)
    away: float = Field(..., gt=1.0)

    @field_validator("bookmaker")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("bookmaker name cannot be empty")
        return v.strip()

    def implied_probabilities(self) -> np.ndarray:
        """
        Raw implied probabilities (1/odds), NOT de-vigged — these still sum
        to more than 1.0 (the overround / bookmaker margin).
        """
        return np.array([1.0 / self.home, 1.0 / self.draw, 1.0 / self.away])

    def overround(self) -> float:
        """The bookmaker's margin, e.g. 0.05 means a 5% overround."""
        return float(self.implied_probabilities().sum() - 1.0)


class MatchOddsQuote(BaseModel):
    """
    All bookmaker quotes for one upcoming match, as returned by a live odds
    feed (e.g. The Odds API), keyed by the odds provider's own team name
    strings and kickoff time — NOT yet matched to a domain `Match.match_id`,
    since odds providers and fixture providers (e.g. football-data.org) use
    different IDs for the same real-world match. See domain/odds/matching.py
    for how the two get reconciled.
    """

    home_team_name: str
    away_team_name: str
    commence_time: datetime
    bookmaker_quotes: List[BookmakerOdds]

    @field_validator("bookmaker_quotes")
    @classmethod
    def non_empty_quotes(cls, v: List[BookmakerOdds]) -> List[BookmakerOdds]:
        if not v:
            raise ValueError("bookmaker_quotes cannot be empty")
        return v
