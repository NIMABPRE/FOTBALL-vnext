"""
Probability Calibration / Shrinkage Engine
============================================

Why this module exists
-----------------------
Raw model probabilities (from Poisson / Dixon-Coles) are point estimates fit on a
finite, noisy sample. Feeding them directly into Edge/EV/Kelly calculations
transmits that estimation uncertainty straight into stake sizing, which
systematically causes overbetting on "value" that is partly just sampling noise.

The standard fix is Bayesian shrinkage: blend the raw model probability toward
the market-implied (de-vigged) probability, which itself encodes the aggregate
information of many informed participants. The blend weight `w` is not guessed
(e.g. an arbitrary 0.7/0.3 split) — it is *fit* on historical settled matches by
minimizing log loss (equivalently, maximizing likelihood) via a 1-D bounded
optimization, and validated with calibration diagnostics (Brier score, log loss,
Expected Calibration Error) on a held-out, time-ordered slice.

This module is self-contained (numpy + scipy only) so it can be dropped into
`src/models/` in the existing football_vnext package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from football_vnext.domain.models.probability import OutcomeProbabilities

logger = logging.getLogger(__name__)
# NOTE: library modules should not attach their own handlers — the
# application entry point (main.py / the Streamlit app) configures logging
# once via logging.basicConfig(). Attaching a handler here as well caused
# every message to be printed twice (once via this handler, once via
# propagation to the root logger).


class CalibrationError(Exception):
    """Raised when calibration fitting or application fails."""


class InsufficientDataError(CalibrationError):
    """Raised when there is not enough historical data to fit a blend weight."""


@dataclass(frozen=True)
class CalibrationSample:
    """
    One historical, settled match used to fit/validate the shrinkage weight.

    `actual_outcome` is 0 for home win, 1 for draw, 2 for away win.
    """

    model_prob: OutcomeProbabilities
    market_prob: OutcomeProbabilities
    actual_outcome: int

    def __post_init__(self) -> None:
        if self.actual_outcome not in (0, 1, 2):
            raise ValueError(
                f"actual_outcome must be 0 (home), 1 (draw) or 2 (away); got {self.actual_outcome}"
            )


@dataclass(frozen=True)
class CalibrationMetrics:
    """Diagnostic metrics for a set of predicted-vs-actual outcomes."""

    log_loss: float
    brier_score: float
    expected_calibration_error: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"CalibrationMetrics(n={self.n_samples}, "
            f"log_loss={self.log_loss:.4f}, brier={self.brier_score:.4f}, "
            f"ece={self.expected_calibration_error:.4f})"
        )


class ProbabilityCalibrator:
    """
    Fits and applies a scalar shrinkage weight `w` such that:

        blended = w * model_prob + (1 - w) * market_prob

    `w` is bounded in [0, 1] and fit by minimizing multiclass log loss on a
    historical sample of settled matches. w=1.0 means "trust the model fully"
    (no shrinkage); w=0.0 means "defer fully to the market" (no model edge is
    ever usable). In practice the fitted value typically lands well below 1.0,
    which is itself useful evidence about how much genuine edge the model has.
    """

    def __init__(self, min_samples: int = 200, ece_bins: int = 10) -> None:
        if min_samples < 30:
            raise ValueError("min_samples must be at least 30 for a stable fit")
        if ece_bins < 2:
            raise ValueError("ece_bins must be at least 2")
        self.min_samples = min_samples
        self.ece_bins = ece_bins
        self._weight: float | None = None
        self._fit_metrics: CalibrationMetrics | None = None

    @property
    def weight(self) -> float:
        if self._weight is None:
            raise CalibrationError("Calibrator has not been fit yet. Call `fit()` first.")
        return self._weight

    @property
    def fit_metrics(self) -> CalibrationMetrics:
        if self._fit_metrics is None:
            raise CalibrationError("Calibrator has not been fit yet. Call `fit()` first.")
        return self._fit_metrics

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def fit(self, samples: Sequence[CalibrationSample]) -> float:
        """
        Fit the shrinkage weight `w` on a chronologically-ordered sample of
        settled matches. Caller is responsible for ensuring `samples` only
        contains matches that occurred *before* the point in time this
        calibrator will be used to predict (no look-ahead).

        Returns the fitted weight.
        """
        n = len(samples)
        if n < self.min_samples:
            raise InsufficientDataError(
                f"Need at least {self.min_samples} settled matches to fit calibration, got {n}."
            )

        model_matrix = np.array([s.model_prob.as_array() for s in samples])
        market_matrix = np.array([s.market_prob.as_array() for s in samples])
        outcomes = np.array([s.actual_outcome for s in samples])

        def neg_log_likelihood(w: float) -> float:
            blended = w * model_matrix + (1.0 - w) * market_matrix
            blended = np.clip(blended, 1e-12, 1.0)
            blended = blended / blended.sum(axis=1, keepdims=True)
            picked = blended[np.arange(n), outcomes]
            return float(-np.sum(np.log(picked)))

        logger.info("Fitting calibration weight on %d historical samples...", n)
        result = minimize_scalar(
            neg_log_likelihood, bounds=(0.0, 1.0), method="bounded",
            options={"xatol": 1e-4},
        )
        if not result.success:
            raise CalibrationError(f"Weight optimization failed: {result.message}")

        self._weight = float(result.x)
        self._fit_metrics = self._evaluate(samples, self._weight)
        logger.info(
            "Calibration fit complete: w=%.4f | %s", self._weight, self._fit_metrics
        )
        return self._weight

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #

    def apply(
        self, model_prob: OutcomeProbabilities, market_prob: OutcomeProbabilities
    ) -> OutcomeProbabilities:
        """Blend a single model/market probability pair using the fitted weight."""
        w = self.weight  # raises if not fit
        blended = w * model_prob.as_array() + (1.0 - w) * market_prob.as_array()
        return OutcomeProbabilities.from_array(blended)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def evaluate_on_holdout(self, samples: Sequence[CalibrationSample]) -> CalibrationMetrics:
        """
        Evaluate the already-fitted weight on a held-out, chronologically-later
        set of samples the weight was NOT fit on. Always call this before trusting
        a calibrator in production — a good in-sample fit with a bad holdout score
        means the blend weight is overfit to the fitting window.
        """
        if not samples:
            raise InsufficientDataError("Holdout sample set is empty.")
        return self._evaluate(samples, self.weight)

    def _evaluate(
        self, samples: Sequence[CalibrationSample], w: float
    ) -> CalibrationMetrics:
        n = len(samples)
        model_matrix = np.array([s.model_prob.as_array() for s in samples])
        market_matrix = np.array([s.market_prob.as_array() for s in samples])
        outcomes = np.array([s.actual_outcome for s in samples])

        blended = w * model_matrix + (1.0 - w) * market_matrix
        blended = np.clip(blended, 1e-12, 1.0)
        blended = blended / blended.sum(axis=1, keepdims=True)

        picked = blended[np.arange(n), outcomes]
        log_loss = float(-np.mean(np.log(picked)))

        one_hot = np.zeros_like(blended)
        one_hot[np.arange(n), outcomes] = 1.0
        brier = float(np.mean(np.sum((blended - one_hot) ** 2, axis=1)))

        ece = self._expected_calibration_error(blended, one_hot)

        return CalibrationMetrics(
            log_loss=log_loss,
            brier_score=brier,
            expected_calibration_error=ece,
            n_samples=n,
        )

    def _expected_calibration_error(
        self, predicted: np.ndarray, one_hot_actual: np.ndarray
    ) -> float:
        """
        ECE across all (match, outcome-class) predictions pooled together:
        bins predictions by confidence and compares average confidence to
        observed frequency within each bin.
        """
        flat_pred = predicted.flatten()
        flat_actual = one_hot_actual.flatten()

        bin_edges = np.linspace(0.0, 1.0, self.ece_bins + 1)
        total = len(flat_pred)
        ece = 0.0

        for i in range(self.ece_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == self.ece_bins - 1:
                mask = (flat_pred >= lo) & (flat_pred <= hi)
            else:
                mask = (flat_pred >= lo) & (flat_pred < hi)

            count = int(mask.sum())
            if count == 0:
                continue

            avg_confidence = float(flat_pred[mask].mean())
            observed_freq = float(flat_actual[mask].mean())
            ece += (count / total) * abs(avg_confidence - observed_freq)

        return ece
