from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MatchStatus(str, Enum):
    SCHEDULED = "scheduled"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class MatchResult(BaseModel):
    home_goals: int = Field(..., ge=0)
    away_goals: int = Field(..., ge=0)

    @property
    def goal_difference(self) -> int:
        return self.home_goals - self.away_goals

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def outcome(self) -> int:
        """0 = home win, 1 = draw, 2 = away win."""
        if self.home_goals > self.away_goals:
            return 0
        if self.home_goals == self.away_goals:
            return 1
        return 2


class Match(BaseModel):
    match_id: str
    competition: str
    season: str
    matchday: Optional[int] = None
    kickoff: datetime
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    status: MatchStatus = MatchStatus.SCHEDULED
    result: Optional[MatchResult] = None
    # Optional pre-match/settled xG observations. Kept nullable so the core
    # model remains usable when an xG provider is unavailable.
    home_xg: Optional[float] = Field(default=None, ge=0.0)
    away_xg: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("match_id", "home_team_id", "away_team_id")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ID cannot be empty")
        return v.strip()
