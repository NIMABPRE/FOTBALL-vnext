"""
Synthetic sample data so the pipeline can be run end-to-end today, without
waiting on a real data source integration (Phase 1.B on the roadmap).

Replace `generate_sample_matches` with a real adapter (football-data.org,
Understat, etc.) once Phase 1.B is implemented — everything downstream
(fit, predict, calibrate) consumes the same `Match` model either way.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from football_vnext.domain.models.match import Match, MatchResult
from football_vnext.domain.models.probability import OutcomeProbabilities

TEAMS = [
    "Arsenal", "Chelsea", "Liverpool", "Man City", "Man United",
    "Tottenham", "Newcastle", "Aston Villa", "Brighton", "West Ham",
]

# Rough relative strengths purely to make the synthetic data non-uniform.
_TRUE_ATTACK = {
    "Arsenal": 1.35, "Chelsea": 1.05, "Liverpool": 1.40, "Man City": 1.50,
    "Man United": 1.10, "Tottenham": 1.20, "Newcastle": 1.15, "Aston Villa": 1.05,
    "Brighton": 1.00, "West Ham": 0.90,
}
_TRUE_DEFENCE = {
    "Arsenal": 0.85, "Chelsea": 1.00, "Liverpool": 0.80, "Man City": 0.75,
    "Man United": 1.05, "Tottenham": 1.10, "Newcastle": 0.95, "Aston Villa": 1.05,
    "Brighton": 1.05, "West Ham": 1.20,
}


def generate_sample_matches(n_rounds: int = 12, seed: int = 42) -> list[Match]:
    """
    Generate a synthetic round-robin-ish history using true Poisson rates,
    so the fitted Dixon-Coles parameters can be sanity-checked against the
    known `_TRUE_ATTACK` / `_TRUE_DEFENCE` ordering.
    """
    rng = np.random.default_rng(seed)
    matches: list[Match] = []
    start = datetime.now(timezone.utc) - timedelta(days=n_rounds * 7)
    match_counter = 0

    for round_idx in range(n_rounds):
        kickoff = start + timedelta(days=round_idx * 7)
        shuffled = list(TEAMS)
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            home, away = shuffled[i], shuffled[i + 1]
            lam = _TRUE_ATTACK[home] * _TRUE_DEFENCE[away] * 1.3
            mu = _TRUE_ATTACK[away] * _TRUE_DEFENCE[home]
            home_goals = int(rng.poisson(lam))
            away_goals = int(rng.poisson(mu))

            match_counter += 1
            matches.append(
                Match(
                    match_id=f"SAMPLE-{match_counter:04d}",
                    competition="Sample League",
                    season="2025/26",
                    matchday=round_idx + 1,
                    kickoff=kickoff,
                    home_team_id=home,
                    away_team_id=away,
                    home_team_name=home,
                    away_team_name=away,
                    status="finished",
                    result=MatchResult(home_goals=home_goals, away_goals=away_goals),
                )
            )
    return matches


def true_match_probabilities(home_team: str, away_team: str, max_goals: int = 10) -> OutcomeProbabilities:
    """
    The GROUND-TRUTH outcome probability for a synthetic match, computed from
    the same `_TRUE_ATTACK` / `_TRUE_DEFENCE` rates used to generate the
    sample data (not from a fitted model). This only exists because the data
    is synthetic — with real data there is no ground truth, which is exactly
    why CLV (not backtested "ground truth accuracy") is the real validation
    metric. Used by the backtester to simulate realistic opening/closing
    bookmaker odds that are centered on a genuine underlying probability.
    """
    if home_team not in _TRUE_ATTACK or away_team not in _TRUE_ATTACK:
        raise ValueError(f"Unknown synthetic team(s): {home_team}, {away_team}")

    lam = _TRUE_ATTACK[home_team] * _TRUE_DEFENCE[away_team] * 1.3
    mu = _TRUE_ATTACK[away_team] * _TRUE_DEFENCE[home_team]

    from scipy.stats import poisson  # local import: keeps this a lightweight, optional helper

    home_probs = poisson.pmf(np.arange(max_goals + 1), lam)
    away_probs = poisson.pmf(np.arange(max_goals + 1), mu)
    matrix = np.outer(home_probs, away_probs)
    matrix = matrix / matrix.sum()

    home_win = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, 1)))
    return OutcomeProbabilities.from_array(np.array([home_win, draw, away_win]))
