"""
Walk-Forward Backtester.

Runs the ENTIRE pipeline (Dixon-Coles fit -> calibration -> de-vig -> Edge/EV
-> Risk Score -> portfolio Kelly) round by round over historical data,
strictly forward in time: a round's bets are always decided using a model
fit only on matches that happened strictly before that round. This is what
"walk-forward" means and why it matters — fitting on a random train/test
split (instead of a chronological one) leaks future information into the
model and inflates backtest performance in a way that will not survive
contact with live markets.

KNOWN SIMPLIFICATION (see also main.py / app.py): the calibration weight is
refit each round using the round's own fitted engine to retrodict
probabilities for matches already in the training window. This is a mild,
documented form of in-sample leakage for the calibration weight specifically
(not for the bets themselves, which only ever see strictly-prior data) and
should be hardened (e.g. via nested walk-forward or held-out folds) before
this module is trusted for real-money decisions.

TWO WAYS TO RUN THIS:
  - `run(matches)` — synthetic odds via `SyntheticOddsGenerator`, for testing
    the pipeline's mechanics on data with no real odds attached (e.g. output
    of `sample_data.py` or a fixtures-only source like football-data.org).
  - `run_with_historical_records(records)` — REAL historical odds (opening
    and, where available, closing) from a `HistoricalMatchOdds` list, e.g.
    from `FootballDataCoUkAdapter`. This is what a trustworthy backtest
    needs; `run()` exists for mechanics-testing, not for real conclusions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from football_vnext.domain.backtest.clv import BetRecord, CLVAnalyzer, CLVSummary
from football_vnext.domain.backtest.historical_odds import HistoricalMatchOdds
from football_vnext.domain.backtest.metrics import BacktestMetrics, BacktestMetricsCalculator
from football_vnext.domain.backtest.synthetic_odds import SyntheticOddsGenerator
from football_vnext.domain.models.match import Match
from football_vnext.domain.models.probability import OutcomeProbabilities
from football_vnext.domain.odds.devig import ShinDevig
from football_vnext.domain.odds.models import BookmakerOdds
from football_vnext.domain.statistics.calibration import CalibrationSample, ProbabilityCalibrator
from football_vnext.domain.statistics.dixon_coles import DixonColesEngine, DixonColesFitError
from football_vnext.domain.value.edge import EdgeDetector, Outcome, build_candidates_from_predictions
from football_vnext.domain.value.kelly import PortfolioKellyStaker
from football_vnext.domain.value.risk import RiskScoreCalculator

logger = logging.getLogger(__name__)

# A function that, given a Match, returns (opening_odds, closing_odds) for it.
OddsProvider = Callable[[Match], Tuple[BookmakerOdds, BookmakerOdds]]


class BacktestEngineError(Exception):
    """Raised on invalid backtest configuration or unrecoverable run errors."""


@dataclass
class BacktestResult:
    bets: List[BetRecord]
    equity_curve: List[float]
    metrics: BacktestMetrics
    clv_summary: CLVSummary
    skipped_rounds: int
    skipped_matches_unknown_team: int
    skipped_matches_missing_odds: int = 0


class WalkForwardBacktester:
    def __init__(
        self,
        min_train_matches: int = 60,
        xi: float = 0.0018,
        min_edge: float = 0.02,
        min_ev: float = 0.02,
        kelly_fraction: float = 0.25,
        max_stake_per_bet: float = 0.05,
        max_total_exposure: float = 0.15,
        max_risk_score: float = 0.6,
        min_reliable_sample_size: int = 20,
        starting_bankroll: float = 1.0,
        odds_generator: Optional[SyntheticOddsGenerator] = None,
        true_probability_fn=None,
        seed: int = 42,
    ) -> None:
        """
        :param true_probability_fn: callable(home_team_id, away_team_id) ->
            OutcomeProbabilities, used ONLY by `run()` (synthetic mode) to
            drive the synthetic odds generator. Not needed for
            `run_with_historical_records()`.
        """
        if min_train_matches < 20:
            raise BacktestEngineError("min_train_matches must be >= 20 (Dixon-Coles minimum)")
        if starting_bankroll <= 0:
            raise BacktestEngineError("starting_bankroll must be > 0")

        self.min_train_matches = min_train_matches
        self.xi = xi
        self.min_edge = min_edge
        self.min_ev = min_ev
        self.kelly_fraction = kelly_fraction
        self.max_stake_per_bet = max_stake_per_bet
        self.max_total_exposure = max_total_exposure
        self.max_risk_score = max_risk_score
        self.min_reliable_sample_size = min_reliable_sample_size
        self.starting_bankroll = starting_bankroll
        self.odds_generator = odds_generator or SyntheticOddsGenerator(seed=seed)
        self.true_probability_fn = true_probability_fn
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #

    def run(self, matches: List[Match]) -> BacktestResult:
        """Synthetic-odds mode — see module docstring."""
        if self.true_probability_fn is None:
            raise BacktestEngineError(
                "true_probability_fn is required for run() (e.g. "
                "sample_data.true_match_probabilities). Use "
                "run_with_historical_records() for real historical odds instead."
            )

        def odds_provider(m: Match) -> Tuple[BookmakerOdds, BookmakerOdds]:
            true_p = self.true_probability_fn(m.home_team_id, m.away_team_id)
            return self.odds_generator.generate(true_p)

        settled = sorted([m for m in matches if m.result is not None], key=lambda m: m.kickoff)
        return self._run_core(settled, odds_provider)

    def run_with_historical_records(self, records: List[HistoricalMatchOdds]) -> BacktestResult:
        """
        REAL historical odds mode — e.g. records from `FootballDataCoUkAdapter`.
        Records missing either opening or closing odds are excluded up front
        (both are required: opening to place the bet, closing to measure CLV).
        """
        usable = [r for r in records if r.match.result is not None
                  and r.opening_odds is not None and r.closing_odds is not None]
        skipped_missing_odds = len(records) - len(usable)

        odds_lookup: Dict[str, Tuple[BookmakerOdds, BookmakerOdds]] = {
            r.match.match_id: (r.opening_odds, r.closing_odds) for r in usable
        }

        def odds_provider(m: Match) -> Tuple[BookmakerOdds, BookmakerOdds]:
            return odds_lookup[m.match_id]

        settled = sorted([r.match for r in usable], key=lambda m: m.kickoff)
        result = self._run_core(settled, odds_provider)
        result.skipped_matches_missing_odds = skipped_missing_odds
        return result

    # ------------------------------------------------------------------ #
    # Shared core
    # ------------------------------------------------------------------ #

    def _fit_calibrator(
        self, engine: DixonColesEngine, train_matches: List[Match], devig: ShinDevig,
        odds_provider: OddsProvider,
    ) -> Optional[ProbabilityCalibrator]:
        """Fit calibration without using the final engine's in-sample predictions.

        The training window is split chronologically: an earlier slice fits a
        calibration model, while a later slice supplies out-of-sample model
        probabilities and outcomes. The final `engine` is never used to
        retrodict its own training observations.
        """
        ordered = sorted([m for m in train_matches if m.result is not None], key=lambda m: m.kickoff)
        if len(ordered) < 80:
            return None
        split = int(len(ordered) * 0.70)
        fit_slice, calib_slice = ordered[:split], ordered[split:]
        if len(fit_slice) < self.min_train_matches or len(calib_slice) < 30:
            return None
        try:
            calib_engine = DixonColesEngine(xi=self.xi)
            calib_engine.fit(fit_slice)
        except DixonColesFitError:
            return None

        samples = []
        for m in calib_slice:
            try:
                retrodicted = calib_engine.predict_match("calib", m.home_team_id, m.away_team_id)
            except ValueError:
                continue
            model_p = OutcomeProbabilities(
                home=retrodicted.home_win_prob, draw=retrodicted.draw_prob, away=retrodicted.away_win_prob
            )
            opening_odds, _ = odds_provider(m)
            market_p = devig.remove_vig(opening_odds)
            samples.append(CalibrationSample(model_prob=model_p, market_prob=market_p, actual_outcome=m.result.outcome))

        if len(samples) < 30:
            return None
        calibrator = ProbabilityCalibrator(min_samples=30)
        calibrator.fit(samples)
        return calibrator

    def _run_core(self, settled: List[Match], odds_provider: OddsProvider) -> BacktestResult:
        if len(settled) < self.min_train_matches + 1:
            raise BacktestEngineError(
                f"Need more than min_train_matches ({self.min_train_matches}) settled "
                f"matches to run any backtest round, got {len(settled)}."
            )

        rounds: dict = {}
        for m in settled:
            # matchday is often absent from real data sources (e.g. the
            # football-data.co.uk CSVs have no matchday column) -- fall back
            # to grouping by kickoff date so those sources don't silently
            # produce zero usable rounds.
            round_key = m.matchday if m.matchday is not None else m.kickoff.date()
            rounds.setdefault(round_key, []).append(m)
        round_keys = sorted(rounds.keys())

        devig = ShinDevig()
        detector = EdgeDetector(min_edge=self.min_edge, min_ev=self.min_ev)
        risk_calc = RiskScoreCalculator(
            min_reliable_sample_size=self.min_reliable_sample_size,
            max_acceptable_risk_score=self.max_risk_score,
        )
        staker = PortfolioKellyStaker(
            kelly_fraction=self.kelly_fraction,
            max_stake_per_bet=self.max_stake_per_bet,
            max_total_exposure=self.max_total_exposure,
        )

        bankroll = self.starting_bankroll
        equity_curve = [bankroll]
        all_bets: List[BetRecord] = []
        skipped_rounds = 0
        skipped_matches_unknown_team = 0

        for round_key in round_keys:
            round_matches = rounds[round_key]
            first_kickoff = min(m.kickoff for m in round_matches)
            train_matches = [m for m in settled if m.kickoff < first_kickoff]

            if len(train_matches) < self.min_train_matches:
                skipped_rounds += 1
                continue

            try:
                engine = DixonColesEngine(xi=self.xi)
                engine.fit(train_matches)
            except DixonColesFitError as exc:
                logger.warning("Skipping round %s: fit failed (%s)", round_key, exc)
                skipped_rounds += 1
                continue

            calibrator = self._fit_calibrator(engine, train_matches, devig, odds_provider)

            round_candidates = []
            round_context = {}  # match_id -> (Match, opening_odds, closing_odds)
            for m in round_matches:
                if m.home_team_id not in engine.team_strengths or m.away_team_id not in engine.team_strengths:
                    skipped_matches_unknown_team += 1
                    continue

                prediction = engine.predict_match(m.match_id, m.home_team_id, m.away_team_id)
                model_p = OutcomeProbabilities(
                    home=prediction.home_win_prob, draw=prediction.draw_prob, away=prediction.away_win_prob
                )
                opening_odds, closing_odds = odds_provider(m)
                fair_market_p = devig.remove_vig(opening_odds)

                calibrated_p = calibrator.apply(model_p, fair_market_p) if calibrator else model_p

                candidates = build_candidates_from_predictions(
                    match_id=m.match_id,
                    calibrated_probs=calibrated_p,
                    fair_market_probs=fair_market_p,
                    home_odds=opening_odds.home,
                    draw_odds=opening_odds.draw,
                    away_odds=opening_odds.away,
                )
                round_candidates.extend(candidates)
                round_context[m.match_id] = (m, opening_odds, closing_odds)

            if not round_candidates:
                continue

            signals = detector.detect(round_candidates)
            if not signals:
                continue

            accepted_signals = []
            for signal in signals:
                m, opening_odds, _ = round_context[signal.candidate.match_id]
                home_matches = engine.team_match_counts.get(m.home_team_id, 0)
                away_matches = engine.team_match_counts.get(m.away_team_id, 0)
                assessment = risk_calc.compute(
                    bookmaker_quotes=[opening_odds],
                    outcome=signal.candidate.outcome,
                    home_team_matches=home_matches,
                    away_team_matches=away_matches,
                    competition=m.competition,
                )
                if assessment.is_acceptable:
                    accepted_signals.append(signal)

            if not accepted_signals:
                continue

            recommendations = staker.compute_stakes(accepted_signals)

            for rec in recommendations:
                if rec.stake_fraction_of_bankroll <= 0:
                    continue
                candidate = rec.signal.candidate
                m, opening_odds, closing_odds = round_context[candidate.match_id]

                stake_amount = rec.stake_fraction_of_bankroll * bankroll
                actual_outcome = m.result.outcome  # 0=home, 1=draw, 2=away
                outcome_index = {Outcome.HOME: 0, Outcome.DRAW: 1, Outcome.AWAY: 2}[candidate.outcome]
                won = actual_outcome == outcome_index
                pnl = stake_amount * (candidate.decimal_odds - 1.0) if won else -stake_amount
                closing_price = getattr(closing_odds, candidate.outcome.value)

                bet = BetRecord(
                    match_id=candidate.match_id,
                    outcome=candidate.outcome,
                    odds_taken=candidate.decimal_odds,
                    closing_odds=closing_price,
                    stake_fraction=rec.stake_fraction_of_bankroll,
                    stake_amount=stake_amount,
                    won=won,
                    pnl=pnl,
                )
                all_bets.append(bet)
                bankroll += pnl

            equity_curve.append(bankroll)

        if not all_bets:
            raise BacktestEngineError(
                "No bets were placed during the entire backtest window — check "
                "min_edge/min_ev/max_risk_score thresholds or the data size."
            )

        metrics_calc = BacktestMetricsCalculator()
        metrics = metrics_calc.compute(all_bets, equity_curve, self.starting_bankroll)

        clv_analyzer = CLVAnalyzer()
        clv_summary = clv_analyzer.summarize(all_bets)

        logger.info(
            "Backtest complete: %d bets, %d rounds skipped (insufficient history), "
            "%d matches skipped (unknown team). %s | %s",
            len(all_bets), skipped_rounds, skipped_matches_unknown_team, metrics, clv_summary,
        )

        return BacktestResult(
            bets=all_bets,
            equity_curve=equity_curve,
            metrics=metrics,
            clv_summary=clv_summary,
            skipped_rounds=skipped_rounds,
            skipped_matches_unknown_team=skipped_matches_unknown_team,
        )
