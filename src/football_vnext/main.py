"""
CLI entry point: runs the current pipeline end-to-end on synthetic sample data.

    python -m football_vnext.main

This demonstrates: fit Dixon-Coles -> predict a match -> (placeholder) market
probabilities -> calibrate/shrink -> print results. The market-probability
step is a stand-in until the Odds Ingestion + De-vig module (next phase) is
built; it is clearly labeled as such below.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

from football_vnext.application.data_loading import load_training_matches
from football_vnext.application.odds_loading import load_match_odds
from football_vnext.config import Settings
from football_vnext.domain.models.probability import OutcomeProbabilities
from football_vnext.domain.odds.devig import ShinDevig
from football_vnext.domain.odds.models import BookmakerOdds
from football_vnext.domain.statistics.calibration import CalibrationSample, ProbabilityCalibrator
from football_vnext.domain.statistics.dixon_coles import DixonColesEngine
from football_vnext.domain.value.edge import EdgeDetector, build_candidates_from_predictions
from football_vnext.domain.value.kelly import PortfolioKellyStaker
from football_vnext.domain.value.risk import RiskScoreCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("football_vnext.main")

# STUB — until a real odds-feed adapter exists (Phase 2.B), this is a
# hand-entered example of what a bookmaker quote looks like. Delete once a
# real data source is wired in. De-vig itself (ShinDevig below) is real math,
# not a stub — only the *input* odds here are placeholders.
EXAMPLE_BOOKMAKER_ODDS = BookmakerOdds(bookmaker="example_bookmaker", home=1.72, draw=3.80, away=5.25)


def run_demo() -> None:
    rng = np.random.default_rng(7)

    logger.info("Loading training matches (real data source if configured, else synthetic)...")
    loaded = load_training_matches(competition_code="PL", n_synthetic_rounds=20)
    matches = loaded.matches
    print(f"\nData source: {loaded.source} ({loaded.competition_label})")
    if loaded.fallback_reason:
        print(f"NOTE: fell back to synthetic data — {loaded.fallback_reason}")
    if loaded.source == "synthetic":
        print(
            "To use real data instead: set FOOTBALL_DATA_API_KEY (get a free key at "
            "https://www.football-data.org/client/register) and re-run."
        )

    logger.info("Fitting Dixon-Coles model...")
    engine = DixonColesEngine(xi=0.0018)
    engine.fit(matches)

    # Pick the two teams from the most recent match in the training set --
    # guaranteed to be present in engine.team_strengths either way.
    demo_match = matches[-1]
    home_team_id, away_team_id = demo_match.home_team_id, demo_match.away_team_id
    home_team, away_team = demo_match.home_team_name, demo_match.away_team_name

    prediction = engine.predict_match("DEMO-0001", home_team_id, away_team_id)

    print(f"\n=== Raw Dixon-Coles prediction: {home_team} vs {away_team} ===")
    print(f"Home win: {prediction.home_win_prob:.2%}")
    print(f"Draw:     {prediction.draw_prob:.2%}")
    print(f"Away win: {prediction.away_win_prob:.2%}")
    print(f"Most likely scoreline: {prediction.most_likely_score}")

    # --- OPTIONAL: LLM-based news adjustment (off by default) -------------
    # This is an auxiliary signal, not a replacement for the statistical
    # model above. Opt in with both an ANTHROPIC_API_KEY and
    # ENABLE_NEWS_ADJUSTMENT=1 -- deliberately two separate switches so a
    # configured key alone never silently changes predictions.
    settings = Settings.from_env()
    if settings.has_anthropic_api_key and os.environ.get("ENABLE_NEWS_ADJUSTMENT") == "1":
        from football_vnext.domain.features.news_impact import (
            NewsAnalysisError,
            NewsImpactAnalyzer,
            apply_news_adjustments,
        )

        print(f"\n=== Optional: LLM news adjustment (enabled) ===")
        # STUB input text -- a real integration would pull this from a news
        # API or injury-report source. The LLM call and adjustment math
        # themselves are real, not a stub; only this example text is.
        example_news = f"No major team news reported for {home_team} this week."
        try:
            analyzer = NewsImpactAnalyzer(api_key=settings.anthropic_api_key)
            home_assessment = analyzer.assess(home_team, example_news)
            print(f"{home_team} assessment: attack x{home_assessment.attack_multiplier:.3f}, "
                  f"defense x{home_assessment.defense_multiplier:.3f}, "
                  f"confidence {home_assessment.confidence:.2f}")
            print(f"Reasoning: {home_assessment.reasoning}")

            adjusted = apply_news_adjustments(
                engine, "DEMO-0001", home_team, away_team,
                prediction.lambda_home, prediction.lambda_away,
                home_assessment=home_assessment, away_assessment=None,
            )
            print(f"\nAdjusted prediction: home {adjusted.home_win_prob:.2%}, "
                  f"draw {adjusted.draw_prob:.2%}, away {adjusted.away_win_prob:.2%}")
            prediction = adjusted  # downstream steps use the adjusted prediction
        except NewsAnalysisError as exc:
            print(f"News adjustment failed ({exc}) -- continuing with the unadjusted prediction.")
    elif settings.has_anthropic_api_key:
        print(
            "\n(LLM news adjustment available but disabled -- set "
            "ENABLE_NEWS_ADJUSTMENT=1 to try it.)"
        )

    # --- Odds: real feed if configured, else the hand-entered example ----
    loaded_odds = load_match_odds(match=demo_match, example_odds=EXAMPLE_BOOKMAKER_ODDS)
    match_odds = loaded_odds.odds
    print(f"\nOdds source: {loaded_odds.source}")
    if loaded_odds.fallback_reason:
        print(f"NOTE: fell back to example odds — {loaded_odds.fallback_reason}")
    if loaded_odds.source == "example":
        print(
            "To use a real odds feed instead: set ODDS_API_KEY (free tier at "
            "https://the-odds-api.com) and re-run. Note the feed only covers "
            "UPCOMING matches, so a match already in the past (like this "
            "historical demo match) will always fall back."
        )

    # --- De-vig: turn the odds quote into a fair probability --------------
    devig = ShinDevig()
    fair_market_probs = devig.remove_vig(match_odds)

    print(f"\n=== De-vigged market (Shin's method) ===")
    print(f"Quote: home {match_odds.home} / draw {match_odds.draw} / away {match_odds.away}")
    print(f"Overround: {match_odds.overround():.2%}")
    print(f"Fair prob -> home: {fair_market_probs.home:.2%}, "
          f"draw: {fair_market_probs.draw:.2%}, away: {fair_market_probs.away:.2%}")

    # --- Calibration layer ---------------------------------------------
    # NOTE: still using a placeholder model_prob (league-average prior) for
    # the historical training samples below, since Dixon-Coles predictions
    # are not yet stored per historical match. This will be replaced once
    # historical predictions are persisted (Phase 1.B).
    logger.info("Building calibration training set (model prior vs de-vigged historical odds)...")
    calibration_samples = []
    for m in matches:
        if m.result is None:
            continue
        model_p = OutcomeProbabilities.from_array(np.array([1 / 3, 1 / 3, 1 / 3]))
        # STUB input odds (randomly perturbed around the example quote) —
        # the de-vig computation itself is real, only these odds are fake.
        noise = rng.normal(0, 0.15, size=3)
        synthetic_odds = BookmakerOdds(
            bookmaker="synthetic",
            home=max(match_odds.home + noise[0], 1.05),
            draw=max(match_odds.draw + noise[1], 1.05),
            away=max(match_odds.away + noise[2], 1.05),
        )
        market_p = devig.remove_vig(synthetic_odds)
        calibration_samples.append(
            CalibrationSample(model_prob=model_p, market_prob=market_p, actual_outcome=m.result.outcome)
        )

    calibrator = ProbabilityCalibrator(min_samples=30)
    calibrator.fit(calibration_samples)
    print(f"\n=== Calibration ===")
    print(f"Fitted blend weight w = {calibrator.weight:.4f}")
    print(calibrator.fit_metrics)

    model_probs = OutcomeProbabilities(
        home=prediction.home_win_prob, draw=prediction.draw_prob, away=prediction.away_win_prob
    )
    blended = calibrator.apply(model_probs, fair_market_probs)

    print(f"\n=== Calibrated (shrunk) prediction: {home_team} vs {away_team} ===")
    print(f"Home win: {blended.home:.2%}")
    print(f"Draw:     {blended.draw:.2%}")
    print(f"Away win: {blended.away:.2%}")

    print(
        f"\nNOTE: odds source was '{loaded_odds.source}' (see above). "
        "The de-vig math itself (Shin's method) is real either way."
    )

    # --- Edge / EV detection + Portfolio Kelly staking -------------------
    candidates = build_candidates_from_predictions(
        match_id="DEMO-0001",
        calibrated_probs=blended,
        fair_market_probs=fair_market_probs,
        home_odds=match_odds.home,
        draw_odds=match_odds.draw,
        away_odds=match_odds.away,
    )

    detector = EdgeDetector(min_edge=0.02, min_ev=0.02)
    signals = detector.detect(candidates)

    print(f"\n=== Edge / EV detection ===")
    for c in candidates:
        print(
            f"{c.outcome.value:5s} | odds {c.decimal_odds:.2f} | "
            f"calibrated {c.calibrated_prob:.2%} | fair_market {c.fair_market_prob:.2%} | "
            f"edge {c.edge:+.2%} | EV {c.expected_value:+.2%}"
        )

    if not signals:
        print("No value-bet signals passed the Edge/EV filters on this example.")
    else:
        # --- Risk Score -----------------------------------------------
        risk_calc = RiskScoreCalculator(min_reliable_sample_size=20, max_acceptable_risk_score=0.6)
        home_matches = engine.team_match_counts.get(home_team_id, 0)
        away_matches = engine.team_match_counts.get(away_team_id, 0)

        print(f"\n=== Risk assessment ===")
        accepted_signals = []
        for signal in signals:
            assessment = risk_calc.compute(
                bookmaker_quotes=[match_odds],  # only one quote available in this demo
                outcome=signal.candidate.outcome,
                home_team_matches=home_matches,
                away_team_matches=away_matches,
                competition=loaded.competition_label,
            )
            status = "ACCEPTED" if assessment.is_acceptable else "REJECTED (too risky)"
            print(
                f"{signal.candidate.outcome.value:5s} | risk_score={assessment.risk_score:.2f} "
                f"(disagreement={assessment.bookmaker_disagreement:.2f}, "
                f"data_quality={assessment.data_quality_risk:.2f}, "
                f"efficiency={assessment.market_efficiency_risk:.2f}/{assessment.market_efficiency_tier.value}) "
                f"-> {status}"
            )
            if assessment.is_acceptable:
                accepted_signals.append((signal, assessment))

        if not accepted_signals:
            print("\nAll signals rejected by the Risk Score filter.")
        else:
            staker = PortfolioKellyStaker(kelly_fraction=0.25, max_stake_per_bet=0.05, max_total_exposure=0.15)
            recommendations = staker.compute_stakes([s for s, _ in accepted_signals])

            print(f"\n=== Stake recommendations (0.25x Kelly, portfolio + risk adjusted) ===")
            for rec, (_, assessment) in zip(recommendations, accepted_signals):
                c = rec.signal.candidate
                risk_adjusted_stake = rec.stake_fraction_of_bankroll * (1 - assessment.risk_score)
                team_label = (
                    home_team if c.outcome.value == "home"
                    else away_team if c.outcome.value == "away" else "Draw"
                )
                print(
                    f"BET: {team_label} @ {c.decimal_odds:.2f} | EV {rec.signal.expected_value:+.2%} | "
                    f"Kelly stake {rec.stake_fraction_of_bankroll:.2%} -> "
                    f"risk-adjusted {risk_adjusted_stake:.2%} of bankroll "
                    f"(risk_score={assessment.risk_score:.2f})"
                )


if __name__ == "__main__":
    run_demo()
