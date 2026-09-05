from __future__ import annotations

import pytest

from football_vnext.domain.backtest.engine import (
    BacktestEngineError,
    WalkForwardBacktester,
)
from football_vnext.sample_data import generate_sample_matches, true_match_probabilities


def test_requires_true_probability_fn_for_run():
    # true_probability_fn is only required by run() (synthetic mode), not at
    # construction time, since run_with_historical_records() doesn't need it.
    backtester = WalkForwardBacktester(min_train_matches=60, true_probability_fn=None)
    matches = generate_sample_matches(n_rounds=80, seed=1)
    with pytest.raises(BacktestEngineError):
        backtester.run(matches)


def test_rejects_too_small_min_train_matches():
    with pytest.raises(BacktestEngineError):
        WalkForwardBacktester(min_train_matches=5, true_probability_fn=true_match_probabilities)


def test_raises_when_not_enough_history_at_all():
    backtester = WalkForwardBacktester(
        min_train_matches=60, true_probability_fn=true_match_probabilities, seed=1,
    )
    matches = generate_sample_matches(n_rounds=10)  # too few to ever clear min_train_matches
    with pytest.raises(BacktestEngineError):
        backtester.run(matches)


def test_full_run_produces_consistent_result_on_synthetic_data():
    backtester = WalkForwardBacktester(
        min_train_matches=60,
        min_edge=0.0,  # loosen thresholds so the small synthetic dataset actually produces bets
        min_ev=0.0,
        max_risk_score=1.0,
        true_probability_fn=true_match_probabilities,
        seed=7,
    )
    matches = generate_sample_matches(n_rounds=20, seed=7)
    result = backtester.run(matches)

    assert result.metrics.n_bets == len(result.bets)
    assert result.metrics.n_bets > 0
    assert result.equity_curve[0] == pytest.approx(backtester.starting_bankroll)
    assert result.equity_curve[-1] == pytest.approx(result.metrics.ending_bankroll)
    assert result.clv_summary.n_bets == result.metrics.n_bets

    # every bet's stake must be non-negative and every recorded outcome
    # consistent with the win/pnl sign
    for bet in result.bets:
        assert bet.stake_amount >= 0
        if bet.won:
            assert bet.pnl > 0
        else:
            assert bet.pnl < 0 or bet.stake_amount == 0


def test_strict_thresholds_can_produce_zero_bets_and_raise():
    backtester = WalkForwardBacktester(
        min_train_matches=60,
        min_edge=0.5,  # deliberately impossible threshold
        min_ev=0.5,
        true_probability_fn=true_match_probabilities,
        seed=3,
    )
    matches = generate_sample_matches(n_rounds=20, seed=3)
    with pytest.raises(BacktestEngineError):
        backtester.run(matches)


def test_run_with_historical_records_end_to_end():
    from football_vnext.domain.backtest.historical_odds import HistoricalMatchOdds
    from football_vnext.domain.backtest.synthetic_odds import SyntheticOddsGenerator

    matches = generate_sample_matches(n_rounds=20, seed=11)
    # matchday grouping shouldn't matter here -- strip it to prove the
    # kickoff-date fallback grouping (used for real sources with no
    # matchday column) works correctly.
    stripped_matches = [m.model_copy(update={"matchday": None}) for m in matches]

    gen = SyntheticOddsGenerator(seed=11)
    records = []
    for m in stripped_matches:
        true_p = true_match_probabilities(m.home_team_id, m.away_team_id)
        opening, closing = gen.generate(true_p)
        records.append(HistoricalMatchOdds(match=m, opening_odds=opening, closing_odds=closing))

    backtester = WalkForwardBacktester(
        min_train_matches=60, min_edge=0.0, min_ev=0.0, max_risk_score=1.0, seed=11,
    )
    result = backtester.run_with_historical_records(records)

    assert result.metrics.n_bets > 0
    assert result.skipped_matches_missing_odds == 0
    assert result.clv_summary.n_bets == result.metrics.n_bets


def test_run_with_historical_records_counts_missing_odds():
    from football_vnext.domain.backtest.historical_odds import HistoricalMatchOdds
    from football_vnext.domain.backtest.synthetic_odds import SyntheticOddsGenerator

    matches = generate_sample_matches(n_rounds=20, seed=13)
    gen = SyntheticOddsGenerator(seed=13)
    records = []
    for i, m in enumerate(matches):
        if i % 5 == 0:
            records.append(HistoricalMatchOdds(match=m, opening_odds=None, closing_odds=None))
            continue
        true_p = true_match_probabilities(m.home_team_id, m.away_team_id)
        opening, closing = gen.generate(true_p)
        records.append(HistoricalMatchOdds(match=m, opening_odds=opening, closing_odds=closing))

    backtester = WalkForwardBacktester(
        min_train_matches=60, min_edge=0.0, min_ev=0.0, max_risk_score=1.0, seed=13,
    )
    result = backtester.run_with_historical_records(records)

    assert result.skipped_matches_missing_odds == len([r for i, r in enumerate(records) if i % 5 == 0])
