"""
Run the walk-forward backtest and print the final go/no-go gate decision.

Synthetic demo (proves the mechanics, not real edge):
    python -m football_vnext.run_backtest

REAL historical data + REAL historical odds (football-data.co.uk, free, no
API key needed):
    python -m football_vnext.run_backtest --real E0 2223 2324 2425

The --real form fetches one or more seasons of a league (e.g. E0=Premier
League) including real opening/closing bookmaker odds, and runs the exact
same pipeline on them. THIS is what makes the gate decision meaningful —
the synthetic default only proves the code runs.
"""

from __future__ import annotations

import logging
import sys
from typing import List

from football_vnext.domain.backtest.engine import BacktestEngineError, WalkForwardBacktester
from football_vnext.domain.backtest.gate import BacktestGate, BacktestGateConfig
from football_vnext.domain.backtest.historical_odds import HistoricalMatchOdds
from football_vnext.infrastructure.data_sources.exceptions import DataSourceError
from football_vnext.infrastructure.data_sources.football_data_co_uk import FootballDataCoUkAdapter
from football_vnext.sample_data import generate_sample_matches, true_match_probabilities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    stream=sys.stdout,
)
# The de-vig and calibration modules log at INFO per-candidate/per-sample,
# which is useful when debugging a single prediction but produces thousands
# of lines across a multi-round backtest. Quiet those two down here; the
# engine's own per-round summary logging (fit, edge detection, risk, Kelly)
# stays at INFO.
for _noisy_logger in (
    "football_vnext.domain.odds.devig",
    "football_vnext.domain.statistics.calibration",
    "football_vnext.domain.value.edge",
    "football_vnext.domain.value.risk",
    "football_vnext.domain.value.kelly",
    "football_vnext.domain.statistics.dixon_coles",
):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger("football_vnext.run_backtest")


def _fetch_real_records(league_code: str, seasons: List[str]) -> List[HistoricalMatchOdds]:
    adapter = FootballDataCoUkAdapter()
    all_records: List[HistoricalMatchOdds] = []
    for season in seasons:
        try:
            records = adapter.fetch_historical_matches(league_code, season)
            all_records.extend(records)
            print(f"  {league_code} {season}: {len(records)} matches")
        except DataSourceError as exc:
            print(f"  {league_code} {season}: FAILED ({exc})")
    return all_records


def _print_report(result, gate: BacktestGate) -> None:
    print(f"\nRounds skipped (insufficient training history): {result.skipped_rounds}")
    print(f"Matches skipped (team unseen in training window): {result.skipped_matches_unknown_team}")
    if result.skipped_matches_missing_odds:
        print(f"Matches skipped (missing opening/closing odds): {result.skipped_matches_missing_odds}")

    print(f"\n=== Performance metrics ===")
    m = result.metrics
    print(f"Bets placed:       {m.n_bets}")
    print(f"Starting bankroll: {m.starting_bankroll:.2f}")
    print(f"Ending bankroll:   {m.ending_bankroll:.2f}")
    print(f"ROI:               {m.roi:+.2%}")
    print(f"Win rate:          {m.win_rate:.1%}")
    print(f"Max drawdown:      {m.max_drawdown:.2%}")
    print(f"Sharpe-like ratio: {m.sharpe_like_ratio:.3f}")
    print(f"ROI p-value:       {m.p_value:.4f} (significant: {m.returns_significantly_positive})")

    print(f"\n=== Closing Line Value (the real evidence of edge) ===")
    c = result.clv_summary
    print(f"Mean CLV:          {c.mean_clv:+.2%}")
    print(f"% bets with +CLV:  {c.pct_positive_clv:.1%}")
    print(f"CLV p-value:       {c.p_value:.4f} (significant: {c.clv_significantly_positive})")

    gate_result = gate.evaluate(m, c)
    print(f"\n=== GATE DECISION ===")
    print(gate_result)
    if not gate_result.passed:
        print(
            "\nThis means: do NOT proceed to real-money execution with this "
            "configuration/data. Each unmet criterion above is an independent "
            "veto — fix the underlying cause (more data, a better model, "
            "different thresholds) rather than lowering the gate's bar."
        )


def main() -> None:
    gate = BacktestGate(BacktestGateConfig(min_bets=300, max_acceptable_drawdown=0.30))

    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        if len(sys.argv) < 4:
            print("Usage: python -m football_vnext.run_backtest --real LEAGUE_CODE SEASON [SEASON ...]")
            print("Example: python -m football_vnext.run_backtest --real E0 2223 2324 2425")
            sys.exit(1)

        league_code = sys.argv[2]
        seasons = sys.argv[3:]

        print("=" * 70)
        print(f"FOOTBALL vNext — Walk-Forward Backtest (REAL DATA: {league_code} {seasons})")
        print("=" * 70)
        print(f"\nFetching from football-data.co.uk...")
        records = _fetch_real_records(league_code, seasons)

        if not records:
            print("\nNo data fetched — check league_code/seasons, or network access.")
            return

        backtester = WalkForwardBacktester(
            min_train_matches=60, min_edge=0.02, min_ev=0.02, kelly_fraction=0.25,
            max_stake_per_bet=0.05, max_total_exposure=0.15, max_risk_score=0.6,
        )
        try:
            result = backtester.run_with_historical_records(records)
        except BacktestEngineError as exc:
            print(f"\nBacktest could not run: {exc}")
            return

        _print_report(result, gate)

    else:
        print("=" * 70)
        print("FOOTBALL vNext — Walk-Forward Backtest (SYNTHETIC DATA)")
        print("=" * 70)
        print(
            "\nTip: run with real historical data + real odds instead: "
            "python -m football_vnext.run_backtest --real E0 2223 2324 2425"
        )

        matches = generate_sample_matches(n_rounds=120, seed=99)
        print(f"\nGenerated {len(matches)} synthetic historical matches across 120 rounds.")

        backtester = WalkForwardBacktester(
            min_train_matches=60, min_edge=0.02, min_ev=0.02, kelly_fraction=0.25,
            max_stake_per_bet=0.05, max_total_exposure=0.15, max_risk_score=0.6,
            true_probability_fn=true_match_probabilities, seed=99,
        )
        try:
            result = backtester.run(matches)
        except BacktestEngineError as exc:
            print(f"\nBacktest could not run: {exc}")
            return

        _print_report(result, gate)


if __name__ == "__main__":
    main()
