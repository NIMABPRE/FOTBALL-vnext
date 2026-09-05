"""
De-vig (margin removal) engines.

Why not simple proportional de-vig
-----------------------------------
Proportional de-vig (p_i = pi_i / sum(pi_j), where pi_i = 1/odds_i) spreads
the bookmaker's margin evenly across all outcomes in proportion to their raw
implied probability. This is simple, but empirically bookmakers do not apply
their margin evenly: they tend to shade extra margin onto longshots (the
market's well-documented "favorite-longshot bias"). Proportional de-vig does
not correct for this, so it systematically overstates the fair probability
of longshots and understates favorites — a model that only ever finds "value"
on underdogs against a proportionally de-vigged market is very likely
detecting this bias, not a real edge.

Shin's method (Shin, 1992/1993) and the Power method both model *how* the
margin is distributed instead of assuming it is uniform, and solve for a
single latent parameter via root-finding.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import brentq

from football_vnext.domain.models.probability import OutcomeProbabilities
from football_vnext.domain.odds.models import BookmakerOdds

logger = logging.getLogger(__name__)


class DevigError(Exception):
    """Raised when a de-vig method fails to find a valid solution."""


class ProportionalDevig:
    """
    Naive baseline: spreads the margin evenly. Kept for comparison purposes
    only — prefer ShinDevig or PowerDevig for actual edge detection.
    """

    name = "proportional"

    def remove_vig(self, odds: BookmakerOdds) -> OutcomeProbabilities:
        raw = odds.implied_probabilities()
        return OutcomeProbabilities.from_array(raw)


class ShinDevig:
    """
    Shin's (1992, 1993) insider-trading model of the bookmaker's overround.

    Assumes a fraction `z` of market activity comes from perfectly informed
    insiders, and solves for both `z` and the fair probabilities `p_i` that
    satisfy, for raw implied probabilities pi_i = 1/odds_i:

        p_i = ( sqrt(z^2 + 4*(1-z)*pi_i^2 / S) - z ) / (2*(1-z))

    where S = sum_j(pi_j) is the total overround (> 1). `z` is found by
    root-finding on the constraint sum_i(p_i) == 1.
    """

    name = "shin"

    def __init__(self, z_upper_bound: float = 0.2) -> None:
        """
        :param z_upper_bound: upper search bound for the insider-trading
            fraction z. Realistic bookmaker overrounds correspond to z well
            under 0.2 (20%); widen only if `remove_vig` raises DevigError.
        """
        if not (0.0 < z_upper_bound < 1.0):
            raise ValueError("z_upper_bound must be in (0, 1)")
        self.z_upper_bound = z_upper_bound

    @staticmethod
    def _probabilities_for_z(pi: np.ndarray, s: float, z: float) -> np.ndarray:
        discriminant = z**2 + 4.0 * (1.0 - z) * (pi**2) / s
        discriminant = np.maximum(discriminant, 0.0)
        return (np.sqrt(discriminant) - z) / (2.0 * (1.0 - z))

    def remove_vig(self, odds: BookmakerOdds) -> OutcomeProbabilities:
        pi = odds.implied_probabilities()
        s = float(pi.sum())

        if s <= 1.0 + 1e-9:
            logger.info("Overround is ~0 (S=%.6f); no vig to remove.", s)
            return OutcomeProbabilities.from_array(pi)

        def residual(z: float) -> float:
            return float(np.sum(self._probabilities_for_z(pi, s, z))) - 1.0

        lo, hi = 1e-9, self.z_upper_bound
        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo * f_hi > 0:
            raise DevigError(
                f"Shin's method: no sign change in [{lo}, {hi}] "
                f"(f_lo={f_lo:.4f}, f_hi={f_hi:.4f}). Overround may be too "
                f"large for this z_upper_bound — try a higher z_upper_bound."
            )

        z_solution = brentq(residual, lo, hi, xtol=1e-10)
        fair = self._probabilities_for_z(pi, s, z_solution)
        logger.info("Shin's method solved z=%.6f (overround=%.4f)", z_solution, s - 1.0)
        return OutcomeProbabilities.from_array(fair)


class PowerDevig:
    """
    Power method: finds an exponent `c` in (0, 1] such that raising the raw
    implied probabilities pi_i = 1/odds_i to the power `c` removes the
    margin exactly:

        sum_i( pi_i ** c ) == 1,   p_i = pi_i ** c

    Compared to proportional de-vig, this compresses the whole distribution
    multiplicatively rather than additively, which — like Shin's method —
    shifts more of the margin removal onto longshots than onto favorites.
    """

    name = "power"

    def remove_vig(self, odds: BookmakerOdds) -> OutcomeProbabilities:
        pi = odds.implied_probabilities()
        s = float(pi.sum())

        if s <= 1.0 + 1e-9:
            logger.info("Overround is ~0 (S=%.6f); no vig to remove.", s)
            return OutcomeProbabilities.from_array(pi)

        def residual(c: float) -> float:
            return float(np.sum(pi**c)) - 1.0

        lo, hi = 1.0, 5.0
        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo * f_hi > 0:
            raise DevigError(
                f"Power method: no sign change in [{lo}, {hi}] "
                f"(f_lo={f_lo:.4f}, f_hi={f_hi:.4f})."
            )

        c_solution = brentq(residual, lo, hi, xtol=1e-10)
        fair = pi**c_solution
        logger.info("Power method solved c=%.6f (overround=%.4f)", c_solution, s - 1.0)
        return OutcomeProbabilities.from_array(fair)
