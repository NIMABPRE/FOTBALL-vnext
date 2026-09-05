from __future__ import annotations

import pytest

from football_vnext.domain.value.edge import (
    EdgeDetector,
    Outcome,
    ValueBetCandidate,
    ValueDetectionError,
    build_candidates_from_predictions,
)
from football_vnext.domain.models.probability import OutcomeProbabilities


def test_candidate_rejects_odds_at_or_below_one():
    with pytest.raises(ValueDetectionError):
        ValueBetCandidate(
            match_id="M1", outcome=Outcome.HOME, decimal_odds=1.0,
            calibrated_prob=0.5, fair_market_prob=0.45,
        )


def test_candidate_rejects_out_of_range_probability():
    with pytest.raises(ValueDetectionError):
        ValueBetCandidate(
            match_id="M1", outcome=Outcome.HOME, decimal_odds=2.0,
            calibrated_prob=1.5, fair_market_prob=0.45,
        )


def test_edge_and_ev_known_values():
    candidate = ValueBetCandidate(
        match_id="M1", outcome=Outcome.HOME, decimal_odds=2.5,
        calibrated_prob=0.50, fair_market_prob=0.42,
    )
    assert candidate.edge == pytest.approx(0.08)
    # EV = 0.5 * 2.5 - 1 = 0.25
    assert candidate.expected_value == pytest.approx(0.25)
    assert candidate.net_odds == pytest.approx(1.5)


def test_build_candidates_from_predictions_creates_three_outcomes():
    calibrated = OutcomeProbabilities(home=0.5, draw=0.25, away=0.25)
    fair = OutcomeProbabilities(home=0.45, draw=0.28, away=0.27)
    candidates = build_candidates_from_predictions(
        "M1", calibrated, fair, home_odds=2.2, draw_odds=3.5, away_odds=4.0
    )
    assert len(candidates) == 3
    assert {c.outcome for c in candidates} == {Outcome.HOME, Outcome.DRAW, Outcome.AWAY}


def test_edge_detector_filters_by_both_thresholds():
    candidates = [
        # passes both
        ValueBetCandidate("M1", Outcome.HOME, 2.5, 0.50, 0.42),
        # good edge but bad EV (odds too low despite edge)
        ValueBetCandidate("M2", Outcome.AWAY, 1.05, 0.99, 0.90),
        # good EV but small edge
        ValueBetCandidate("M3", Outcome.DRAW, 10.0, 0.12, 0.10),
        # neither
        ValueBetCandidate("M4", Outcome.HOME, 2.0, 0.40, 0.42),
    ]
    detector = EdgeDetector(min_edge=0.03, min_ev=0.03)
    signals = detector.detect(candidates)
    match_ids = {s.candidate.match_id for s in signals}
    assert "M1" in match_ids
    assert "M4" not in match_ids


def test_edge_detector_sorts_by_ev_descending():
    candidates = [
        ValueBetCandidate("LOW", Outcome.HOME, 2.0, 0.55, 0.45),   # EV=0.1
        ValueBetCandidate("HIGH", Outcome.HOME, 3.0, 0.55, 0.30),  # EV=0.65
    ]
    detector = EdgeDetector(min_edge=0.0, min_ev=0.0)
    signals = detector.detect(candidates)
    assert signals[0].candidate.match_id == "HIGH"
    assert signals[1].candidate.match_id == "LOW"


def test_edge_detector_rejects_negative_thresholds():
    with pytest.raises(ValueError):
        EdgeDetector(min_edge=-0.1)
    with pytest.raises(ValueError):
        EdgeDetector(min_ev=-0.1)
