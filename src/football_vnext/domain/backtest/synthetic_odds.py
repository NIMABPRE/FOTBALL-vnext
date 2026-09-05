"""
Synthetic opening/closing odds generator — FOR BACKTESTING ON SYNTHETIC DATA ONLY.

A real backtest needs real historical opening AND closing odds per match,
which requires a historical odds archive (not yet integrated — see the
project roadmap). Until then, this module lets the walk-forward backtester
exercise its full mechanics (staking, CLV measurement, gate evaluation) end
to end using synthetic data, by generating two noisy quotes around a known
ground-truth probability (only available because the underlying match data
is itself synthetic): a noisier "opening" quote (what you could have bet at)
and a less noisy "closing" quote (what the market settles on right before
kickoff, used as the CLV benchmark).

Replace this module's usage with a real historical-odds adapter before
trusting any backtest result produced with it.
"""

from __future__ import annotations

import numpy as np

from football_vnext.domain.models.probability import OutcomeProbabilities
from football_vnext.domain.odds.models import BookmakerOdds


class SyntheticOddsGenerator:
    def __init__(
        self,
        opening_noise_std: float = 0.06,
        closing_noise_std: float = 0.02,
        margin: float = 0.05,
        seed: int = 123,
    ) -> None:
        if opening_noise_std < 0 or closing_noise_std < 0:
            raise ValueError("noise std values must be >= 0")
        if margin < 0:
            raise ValueError("margin must be >= 0")
        self.opening_noise_std = opening_noise_std
        self.closing_noise_std = closing_noise_std
        self.margin = margin
        self._rng = np.random.default_rng(seed)

    def _probs_to_odds(self, probs: np.ndarray, noise_std: float) -> BookmakerOdds:
        noisy = probs + self._rng.normal(0, noise_std, size=3)
        noisy = np.clip(noisy, 1e-3, None)
        noisy = noisy / noisy.sum()
        inflated = noisy * (1.0 + self.margin)
        # Guard against inflated probabilities approaching 1.0 (which would
        # produce decimal odds <= 1.0, invalid for BookmakerOdds) -- this can
        # happen when noise pushes one outcome's probability very high before
        # normalization, e.g. for a heavy favorite plus unlucky noise draws.
        inflated = np.minimum(inflated, 0.99)
        odds = np.maximum(1.0 / inflated, 1.01)
        return BookmakerOdds(
            bookmaker="synthetic",
            home=float(odds[0]),
            draw=float(odds[1]),
            away=float(odds[2]),
        )

    def generate(self, true_probs: OutcomeProbabilities) -> tuple[BookmakerOdds, BookmakerOdds]:
        """Returns (opening_odds, closing_odds) for one match."""
        arr = true_probs.as_array()
        opening = self._probs_to_odds(arr, self.opening_noise_std)
        closing = self._probs_to_odds(arr, self.closing_noise_std)
        return opening, closing
