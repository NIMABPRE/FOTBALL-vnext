"""Unified pre-match context adjustments (xG + structured team news)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class TeamContext:
    attack_multiplier: float = 1.0
    defense_multiplier: float = 1.0
    confidence: float = 0.0
    source: str = "none"
    reasoning: str = ""


def apply_context(lambda_home: float, lambda_away: float, home: Optional[TeamContext] = None, away: Optional[TeamContext] = None):
    h = home or TeamContext()
    a = away or TeamContext()
    # Own attack and opponent defence. defense < 1 means worse defense.
    lh = lambda_home * h.attack_multiplier / max(a.defense_multiplier, 1e-9)
    la = lambda_away * a.attack_multiplier / max(h.defense_multiplier, 1e-9)
    return max(0.05, lh), max(0.05, la)
