from __future__ import annotations

import numpy as np
import pytest

from football_vnext.domain.value.edge import Outcome, ValueBetCandidate, ValueBetSignal
from football_vnext.domain.value.kelly import (
    KellyError,
    PortfolioKellyStaker,
    single_bet_kelly_fraction,
)


def _signal(match_id: str, prob: float, odds: float) -> ValueBetSignal:
    candidate = ValueBetCandidate(
        match_id=match_id, outcome=Outcome.HOME, decimal_odds=odds,
        calibrated_prob=prob, fair_market_prob=max(prob - 0.05, 0.01),
    )
    return ValueBetSignal(candidate=candidate, edge=candidate.edge, expected_value=candidate.expected_value)


def test_single_bet_kelly_known_value():
    # p=0.6, odds=2.0 -> b=1.0 -> f* = (1*0.6 - 0.4) / 1 = 0.2
    f = single_bet_kelly_fraction(probability=0.6, net_odds=1.0)
    assert f == pytest.approx(0.2)


def test_single_bet_kelly_no_edge_returns_zero():
    # p=0.4, odds=2.0 -> b=1.0 -> f* = (0.4 - 0.6)/1 = -0.2 -> clipped to 0
    f = single_bet_kelly_fraction(probability=0.4, net_odds=1.0)
    assert f == 0.0


def test_single_bet_kelly_rejects_invalid_inputs():
    with pytest.raises(KellyError):
        single_bet_kelly_fraction(probability=1.5, net_odds=1.0)
    with pytest.raises(KellyError):
        single_bet_kelly_fraction(probability=0.5, net_odds=0.0)


def test_portfolio_staker_rejects_invalid_params():
    with pytest.raises(KellyError):
        PortfolioKellyStaker(kelly_fraction=0.0)
    with pytest.raises(KellyError):
        PortfolioKellyStaker(max_stake_per_bet=1.5)
    with pytest.raises(KellyError):
        PortfolioKellyStaker(max_total_exposure=0.0)


def test_compute_stakes_empty_returns_empty():
    staker = PortfolioKellyStaker()
    assert staker.compute_stakes([]) == []


def test_compute_stakes_single_signal_matches_naive_fractional_kelly():
    staker = PortfolioKellyStaker(kelly_fraction=0.25, max_stake_per_bet=0.5, max_total_exposure=0.5)
    signal = _signal("M1", prob=0.6, odds=2.0)
    recs = staker.compute_stakes([signal])
    assert len(recs) == 1
    expected_full_kelly = single_bet_kelly_fraction(0.6, 1.0)  # 0.2
    expected_fractional = expected_full_kelly * 0.25  # 0.05
    assert recs[0].naive_kelly_fraction == pytest.approx(expected_fractional)
    # single bet has no correlation to scale against -> unaffected
    assert recs[0].stake_fraction_of_bankroll == pytest.approx(expected_fractional)


def test_correlated_same_match_bets_get_scaled_down_more_than_independent():
    staker = PortfolioKellyStaker(kelly_fraction=0.25, max_stake_per_bet=0.5, max_total_exposure=1.0)

    correlated_signals = [_signal("SAME_MATCH", 0.6, 2.0), _signal("SAME_MATCH", 0.55, 2.2)]
    independent_signals = [_signal("MATCH_A", 0.6, 2.0), _signal("MATCH_B", 0.55, 2.2)]

    correlated_recs = staker.compute_stakes(correlated_signals)
    independent_recs = staker.compute_stakes(independent_signals)

    correlated_total = sum(r.stake_fraction_of_bankroll for r in correlated_recs)
    independent_total = sum(r.stake_fraction_of_bankroll for r in independent_recs)

    assert correlated_total < independent_total


def test_max_total_exposure_is_enforced():
    staker = PortfolioKellyStaker(kelly_fraction=1.0, max_stake_per_bet=1.0, max_total_exposure=0.1)
    signals = [_signal(f"M{i}", 0.7, 3.0) for i in range(5)]
    recs = staker.compute_stakes(signals)
    total = sum(r.stake_fraction_of_bankroll for r in recs)
    assert total <= 0.1 + 1e-9


def test_max_stake_per_bet_caps_individual_naive_fraction():
    staker = PortfolioKellyStaker(kelly_fraction=1.0, max_stake_per_bet=0.05, max_total_exposure=1.0)
    # p=0.9, odds=5.0 -> huge uncapped Kelly fraction
    signal = _signal("M1", prob=0.9, odds=5.0)
    recs = staker.compute_stakes([signal])
    assert recs[0].naive_kelly_fraction <= 0.05 + 1e-9


def test_explicit_correlation_matrix_is_respected():
    staker = PortfolioKellyStaker(kelly_fraction=0.25, max_stake_per_bet=0.5, max_total_exposure=1.0)
    signals = [_signal("M1", 0.6, 2.0), _signal("M2", 0.6, 2.0)]

    zero_corr = np.eye(2)
    full_corr = np.ones((2, 2))

    zero_corr_recs = staker.compute_stakes(signals, correlation_matrix=zero_corr)
    full_corr_recs = staker.compute_stakes(signals, correlation_matrix=full_corr)

    zero_total = sum(r.stake_fraction_of_bankroll for r in zero_corr_recs)
    full_total = sum(r.stake_fraction_of_bankroll for r in full_corr_recs)
    assert full_total < zero_total


def test_wrong_shape_correlation_matrix_raises():
    staker = PortfolioKellyStaker()
    signals = [_signal("M1", 0.6, 2.0), _signal("M2", 0.6, 2.0)]
    with pytest.raises(KellyError):
        staker.compute_stakes(signals, correlation_matrix=np.eye(3))
