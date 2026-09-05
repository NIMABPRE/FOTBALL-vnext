from __future__ import annotations

from pydantic import BaseModel, Field


class TeamStrength(BaseModel):
    """Attack and defence strengths used by Poisson / Dixon-Coles."""

    team_id: str
    attack: float = Field(..., gt=0.0)
    defence: float = Field(..., gt=0.0)
    home_advantage: float = Field(default=1.0, gt=0.0)
