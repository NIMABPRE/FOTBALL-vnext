from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from football_vnext.domain.models.match import Match
from football_vnext.domain.models.prediction import MatchPrediction
from football_vnext.domain.models.team import TeamStrength
from football_vnext.domain.statistics.poisson import PoissonEngine
from football_vnext.domain.features.xg import XGFeatureEngine

logger = logging.getLogger(__name__)


class DixonColesFitError(Exception):
    """Raised when the Dixon-Coles MLE optimization fails or inputs are invalid."""


class DixonColesEngine(PoissonEngine):
    """
    Dixon-Coles model: independent-Poisson base plus a low-score correlation
    correction (tau) and exponential time-decay weighting of historical
    matches, with attack/defence strengths fit by maximum likelihood.
    """

    def __init__(
        self,
        max_goals: int = 10,
        xi: float = 0.0018,
    ) -> None:
        """
        :param xi: time-decay rate applied to days-since-match (higher = faster forgetting)
        """
        super().__init__(max_goals=max_goals)
        self.xi = xi
        self.rho: Optional[float] = None
        self.home_advantage: Optional[float] = None
        self.team_strengths: Dict[str, TeamStrength] = {}
        self.team_match_counts: Dict[str, int] = {}
        self._is_fit = False
        self.xg_features = XGFeatureEngine()

    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1.0 - lam * mu * rho
        if x == 1 and y == 0:
            return 1.0 + lam * rho
        if x == 0 and y == 1:
            return 1.0 + mu * rho
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def fit(self, matches: List[Match], ref_date: Optional[datetime] = None) -> None:
        settled = [m for m in matches if m.result is not None]
        if len(settled) < 20:
            raise DixonColesFitError(
                f"Need at least 20 settled matches to fit Dixon-Coles, got {len(settled)}."
            )

        if ref_date is None:
            ref_date = max(m.kickoff for m in settled)

        teams = sorted({m.home_team_id for m in settled} | {m.away_team_id for m in settled})
        n_teams = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        self.team_match_counts = {t: 0 for t in teams}
        for m in settled:
            self.team_match_counts[m.home_team_id] += 1
            self.team_match_counts[m.away_team_id] += 1

        home_idx = np.array([idx[m.home_team_id] for m in settled])
        away_idx = np.array([idx[m.away_team_id] for m in settled])
        home_goals = np.array([m.result.home_goals for m in settled])
        away_goals = np.array([m.result.away_goals for m in settled])

        days_diff = np.array([max((ref_date - m.kickoff).days, 0) for m in settled])
        weights = np.exp(-self.xi * days_diff)

        def neg_log_likelihood(params: np.ndarray) -> float:
            gamma = params[0]
            rho = params[1]
            # Enforce the attack-strength identification constraint
            # sum(att) = n_teams without a constrained optimizer. Centering
            # the free attack parameters keeps the objective smooth and lets
            # L-BFGS-B handle the fit much more reliably than SLSQP on larger
            # rolling windows.
            raw_att = params[2 : n_teams + 2]
            att = 1.0 + raw_att - np.mean(raw_att)
            dfc = params[n_teams + 2 :]

            lam = np.exp(att[home_idx] + dfc[away_idx] + gamma)
            mu = np.exp(att[away_idx] + dfc[home_idx])

            p_home = poisson.pmf(home_goals, lam)
            p_away = poisson.pmf(away_goals, mu)
            # Vectorized low-score Dixon-Coles correction. This is materially
            # faster than constructing one Python-level tau value per match
            # on every optimizer evaluation, while preserving the exact
            # correction definition.
            tau_vec = np.ones_like(lam, dtype=float)
            mask_00 = (home_goals == 0) & (away_goals == 0)
            mask_10 = (home_goals == 1) & (away_goals == 0)
            mask_01 = (home_goals == 0) & (away_goals == 1)
            mask_11 = (home_goals == 1) & (away_goals == 1)
            tau_vec[mask_00] = 1.0 - lam[mask_00] * mu[mask_00] * rho
            tau_vec[mask_10] = 1.0 + lam[mask_10] * rho
            tau_vec[mask_01] = 1.0 + mu[mask_01] * rho
            tau_vec[mask_11] = 1.0 - rho
            probs = np.maximum(tau_vec * p_home * p_away, 1e-12)
            return float(-np.sum(weights * np.log(probs)))

        init = np.zeros(2 * n_teams + 2)
        init[0] = 0.25
        init[1] = -0.05

        logger.info("Fitting Dixon-Coles on %d matches (%d teams)...", len(settled), n_teams)
        # rho is weakly bounded to keep the low-score correction numerically
        # well behaved. Attack identification is handled inside the objective.
        bounds = [(None, None), (-0.99, 0.99)] + [(None, None)] * (2 * n_teams)
        result = minimize(
            neg_log_likelihood, init, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-6, "maxls": 40},
        )
        if not result.success:
            raise DixonColesFitError(f"Optimization failed: {result.message}")

        opt = result.x
        self.home_advantage = float(opt[0])
        self.rho = float(opt[1])
        raw_att_opt = opt[2 : n_teams + 2]
        att_opt = 1.0 + raw_att_opt - np.mean(raw_att_opt)
        dfc_opt = opt[n_teams + 2 :]

        # xG is a bounded correction layer fitted only on this training window.
        self.xg_features.fit(settled, ref_date=ref_date)

        self.team_strengths = {
            team: TeamStrength(
                team_id=team,
                attack=float(np.exp(att_opt[i])),
                defence=float(np.exp(dfc_opt[i])),
                home_advantage=float(np.exp(self.home_advantage)),
            )
            for team, i in idx.items()
        }
        self._is_fit = True
        logger.info(
            "Dixon-Coles fit complete: home_adv=%.4f, rho=%.4f", self.home_advantage, self.rho
        )

    def scoreline_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        if self.rho is None:
            raise DixonColesFitError("Model has not been fit yet. Call fit() first.")

        home_probs = poisson.pmf(np.arange(self.max_goals + 1), lambda_home)
        away_probs = poisson.pmf(np.arange(self.max_goals + 1), lambda_away)

        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                tau = self._tau(i, j, lambda_home, lambda_away, self.rho)
                matrix[i, j] = home_probs[i] * away_probs[j] * tau

        matrix = np.maximum(matrix, 0.0)
        total = matrix.sum()
        if total <= 0:
            raise RuntimeError("Dixon-Coles matrix summed to zero or negative")
        return matrix / total

    def predict_match(self, match_id: str, home_team_id: str, away_team_id: str) -> MatchPrediction:
        if not self._is_fit:
            raise DixonColesFitError("Model has not been fit yet. Call fit() first.")
        if home_team_id not in self.team_strengths or away_team_id not in self.team_strengths:
            raise ValueError("One or both teams were not present in the training data.")

        home = self.team_strengths[home_team_id]
        away = self.team_strengths[away_team_id]

        # Note: home.home_advantage already carries exp(gamma); away side uses raw attack/defence.
        lambda_home = home.attack * away.defence * home.home_advantage
        lambda_away = away.attack * home.defence

        xg_home_mult, xg_away_mult = self.xg_features.multipliers(home_team_id, away_team_id)
        lambda_home *= xg_home_mult
        lambda_away *= xg_away_mult

        matrix = self.scoreline_matrix(lambda_home, lambda_away)
        model_name = "dixon_coles+xg" if (xg_home_mult, xg_away_mult) != (1.0, 1.0) else "dixon_coles"
        return self._matrix_to_prediction(
            match_id, home_team_id, away_team_id, lambda_home, lambda_away, matrix, model_name
        )
