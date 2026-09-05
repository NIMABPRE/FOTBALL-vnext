from __future__ import annotations

from pydantic import BaseModel, Field


class ScorelineProbability(BaseModel):
    home_goals: int
    away_goals: int
    probability: float = Field(..., ge=0.0, le=1.0)


class MatchPrediction(BaseModel):
    match_id: str
    home_team_name: str
    away_team_name: str
    lambda_home: float = Field(..., gt=0.0)
    lambda_away: float = Field(..., gt=0.0)
    scoreline_probs: list[ScorelineProbability]
    home_win_prob: float = Field(..., ge=0.0, le=1.0)
    draw_prob: float = Field(..., ge=0.0, le=1.0)
    away_win_prob: float = Field(..., ge=0.0, le=1.0)
    most_likely_score: tuple[int, int]
    model_name: str
    model_version: str = "1.0.0"
