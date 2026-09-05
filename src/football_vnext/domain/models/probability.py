from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OutcomeProbabilities:
    """
    A single 3-way probability vector (home / draw / away) for a football match.
    Values must be non-negative and sum to ~1.0 (validated on construction).
    """

    home: float
    draw: float
    away: float

    def __post_init__(self) -> None:
        for name, value in (("home", self.home), ("draw", self.draw), ("away", self.away)):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} probability out of range [0,1]: {value}")
        total = self.home + self.draw + self.away
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Probabilities must sum to 1.0, got {total:.6f}")

    def as_array(self) -> np.ndarray:
        return np.array([self.home, self.draw, self.away], dtype=float)

    @staticmethod
    def from_array(arr: np.ndarray) -> "OutcomeProbabilities":
        """
        Build a valid, normalized 3-way distribution from raw (possibly
        unnormalized) non-negative values. Only the lower bound is clipped
        before normalizing — clipping the upper bound first would corrupt
        inputs whose components exceed 1.0 prior to normalization.
        """
        arr = np.clip(arr, 1e-12, None)
        arr = arr / arr.sum()
        return OutcomeProbabilities(home=float(arr[0]), draw=float(arr[1]), away=float(arr[2]))
