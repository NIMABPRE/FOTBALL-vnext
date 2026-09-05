from __future__ import annotations

import pytest

from football_vnext.domain.backtest.synthetic_odds import SyntheticOddsGenerator
from football_vnext.domain.models.probability import OutcomeProbabilities


def test_generate_returns_valid_odds():
    gen = SyntheticOddsGenerator(seed=1)
    true_p = OutcomeProbabilities(home=0.5, draw=0.25, away=0.25)
    opening, closing = gen.generate(true_p)
    for odds in (opening, closing):
        assert odds.home > 1.0
        assert odds.draw > 1.0
        assert odds.away > 1.0
        assert odds.overround() > 0  # margin is applied


def test_closing_odds_are_less_noisy_than_opening_on_average():
    gen = SyntheticOddsGenerator(opening_noise_std=0.10, closing_noise_std=0.01, seed=2)
    true_p = OutcomeProbabilities(home=0.5, draw=0.25, away=0.25)

    opening_probs = []
    closing_probs = []
    for _ in range(500):
        opening, closing = gen.generate(true_p)
        opening_probs.append(opening.implied_probabilities()[0])
        closing_probs.append(closing.implied_probabilities()[0])

    import numpy as np
    # closing quotes should cluster more tightly around the true probability
    assert np.std(closing_probs) < np.std(opening_probs)


def test_rejects_invalid_params():
    with pytest.raises(ValueError):
        SyntheticOddsGenerator(opening_noise_std=-0.1)
    with pytest.raises(ValueError):
        SyntheticOddsGenerator(margin=-0.1)
