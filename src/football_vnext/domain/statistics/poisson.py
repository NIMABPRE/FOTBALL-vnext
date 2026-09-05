from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.stats import poisson

from football_vnext.domain.models.prediction import MatchPrediction, ScorelineProbability
from football_vnext.domain.models.team import TeamStrength

logger = logging.getLogger(__name__)


class PoissonEngine:
    """
    Independent Poisson model for football scores.

    lambda_home = attack_home * defence_away * home_advantage * league_avg_home
    lambda_away = attack_away * defence_home * league_avg_away
    """

    def __init__(
        self,
        max_goals: int = 10,
        league_avg_home: float = 1.45,
        league_avg_away: float = 1.15,
    ) -> None:
        if max_goals < 5:
            raise ValueError("max_goals should be at least 5")
        self.max_goals = max_goals
        self.league_avg_home = league_avg_home
        self.league_avg_away = league_avg_away

    def calculate_lambdas(self, home: TeamStrength, away: TeamStrength) -> Tuple[float, float]:
        lambda_home = home.attack * away.defence * home.home_advantage * self.league_avg_home
        lambda_away = away.attack * home.defence * self.league_avg_away
        return max(lambda_home, 1e-6), max(lambda_away, 1e-6)

    def scoreline_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        home_probs = poisson.pmf(np.arange(self.max_goals + 1), lambda_home)
        away_probs = poisson.pmf(np.arange(self.max_goals + 1), lambda_away)
        matrix = np.outer(home_probs, away_probs)
        total = matrix.sum()
        if total <= 0:
            raise RuntimeError("Probability matrix summed to zero")
        return matrix / total

    def predict(self, match_id: str, home: TeamStrength, away: TeamStrength) -> MatchPrediction:
        lambda_home, lambda_away = self.calculate_lambdas(home, away)
        matrix = self.scoreline_matrix(lambda_home, lambda_away)
        return self._matrix_to_prediction(
            match_id, home.team_id, away.team_id, lambda_home, lambda_away, matrix, "poisson_independent"
        )

    def _matrix_to_prediction(
        self,
        match_id: str,
        home_name: str,
        away_name: str,
        lambda_home: float,
        lambda_away: float,
        matrix: np.ndarray,
        model_name: str,
    ) -> MatchPrediction:
        scorelines: List[ScorelineProbability] = []
        home_win = draw = away_win = 0.0
        best_prob = -1.0
        best_score = (0, 0)

        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                p = float(matrix[i, j])
                scorelines.append(ScorelineProbability(home_goals=i, away_goals=j, probability=p))
                if i > j:
                    home_win += p
                elif i == j:
                    draw += p
                else:
                    away_win += p
                if p > best_prob:
                    best_prob = p
                    best_score = (i, j)

        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total

        return MatchPrediction(
            match_id=match_id,
            home_team_name=home_name,
            away_team_name=away_name,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            scoreline_probs=scorelines,
            home_win_prob=home_win,
            draw_prob=draw,
            away_win_prob=away_win,
            most_likely_score=best_score,
            model_name=model_name,
        )
