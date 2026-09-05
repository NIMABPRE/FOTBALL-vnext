"""
Unit tests for src/models/calibration.py

Run with: python -m pytest tests/test_calibration.py -v
(or plain `python tests/test_calibration.py` — a manual runner is included
at the bottom so this also works without pytest installed.)
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from football_vnext.domain.statistics.calibration import (  # noqa: E402
    CalibrationError,
    CalibrationSample,
    InsufficientDataError,
    OutcomeProbabilities,
    ProbabilityCalibrator,
)


def test_outcome_probabilities_validates_sum():
    with pytest.raises(ValueError):
        OutcomeProbabilities(home=0.5, draw=0.5, away=0.5)


def test_outcome_probabilities_validates_range():
    with pytest.raises(ValueError):
        OutcomeProbabilities(home=1.5, draw=-0.3, away=-0.2)


def test_from_array_renormalizes():
    probs = OutcomeProbabilities.from_array(np.array([2.0, 1.0, 1.0]))
    assert np.isclose(probs.home + probs.draw + probs.away, 1.0)
    assert np.isclose(probs.home, 0.5)


def test_fit_raises_on_insufficient_data():
    calibrator = ProbabilityCalibrator(min_samples=200)
    tiny_sample = [
        CalibrationSample(
            model_prob=OutcomeProbabilities(0.5, 0.25, 0.25),
            market_prob=OutcomeProbabilities(0.45, 0.28, 0.27),
            actual_outcome=0,
        )
    ] * 5
    with pytest.raises(InsufficientDataError):
        calibrator.fit(tiny_sample)


def test_weight_and_metrics_raise_before_fit():
    calibrator = ProbabilityCalibrator(min_samples=30)
    with pytest.raises(CalibrationError):
        _ = calibrator.weight
    with pytest.raises(CalibrationError):
        _ = calibrator.fit_metrics


def test_fit_recovers_known_weight_on_synthetic_data():
    """
    Construct synthetic data where the TRUE data-generating probability is a
    known blend of model/market at w_true=0.6, then verify the optimizer
    recovers a weight close to 0.6 from simulated outcomes.
    """
    rng = np.random.default_rng(42)
    n = 4000
    w_true = 0.6

    samples = []
    for _ in range(n):
        model_raw = rng.dirichlet([6, 3, 5])
        market_raw = rng.dirichlet([5, 4, 5])
        model_p = OutcomeProbabilities.from_array(model_raw)
        market_p = OutcomeProbabilities.from_array(market_raw)

        true_p = w_true * model_p.as_array() + (1 - w_true) * market_p.as_array()
        outcome = int(rng.choice([0, 1, 2], p=true_p))

        samples.append(
            CalibrationSample(model_prob=model_p, market_prob=market_p, actual_outcome=outcome)
        )

    calibrator = ProbabilityCalibrator(min_samples=200)
    fitted_w = calibrator.fit(samples)

    assert abs(fitted_w - w_true) < 0.08, f"Expected w~{w_true}, got {fitted_w:.4f}"
    assert calibrator.fit_metrics.n_samples == n
    assert calibrator.fit_metrics.log_loss > 0


def test_apply_returns_valid_probability_distribution():
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(300):
        model_p = OutcomeProbabilities.from_array(rng.dirichlet([5, 3, 4]))
        market_p = OutcomeProbabilities.from_array(rng.dirichlet([5, 3, 4]))
        outcome = int(rng.choice([0, 1, 2]))
        samples.append(CalibrationSample(model_p, market_p, outcome))

    calibrator = ProbabilityCalibrator(min_samples=200)
    calibrator.fit(samples)

    blended = calibrator.apply(
        OutcomeProbabilities(0.6, 0.2, 0.2), OutcomeProbabilities(0.4, 0.3, 0.3)
    )
    total = blended.home + blended.draw + blended.away
    assert np.isclose(total, 1.0)
    assert 0.0 <= blended.home <= 1.0


def test_evaluate_on_holdout_uses_fitted_weight():
    rng = np.random.default_rng(7)
    fit_samples = []
    holdout_samples = []
    for i in range(500):
        model_p = OutcomeProbabilities.from_array(rng.dirichlet([5, 3, 4]))
        market_p = OutcomeProbabilities.from_array(rng.dirichlet([5, 3, 4]))
        outcome = int(rng.choice([0, 1, 2], p=model_p.as_array()))
        sample = CalibrationSample(model_p, market_p, outcome)
        (fit_samples if i < 350 else holdout_samples).append(sample)

    calibrator = ProbabilityCalibrator(min_samples=200)
    calibrator.fit(fit_samples)
    holdout_metrics = calibrator.evaluate_on_holdout(holdout_samples)

    assert holdout_metrics.n_samples == len(holdout_samples)
    assert holdout_metrics.log_loss > 0
    assert 0.0 <= holdout_metrics.expected_calibration_error <= 1.0


def test_evaluate_on_holdout_raises_on_empty():
    calibrator = ProbabilityCalibrator(min_samples=30)
    rng = np.random.default_rng(0)
    fit_samples = [
        CalibrationSample(
            OutcomeProbabilities.from_array(rng.dirichlet([5, 3, 4])),
            OutcomeProbabilities.from_array(rng.dirichlet([5, 3, 4])),
            int(rng.choice([0, 1, 2])),
        )
        for _ in range(50)
    ]
    calibrator.fit(fit_samples)
    with pytest.raises(InsufficientDataError):
        calibrator.evaluate_on_holdout([])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
