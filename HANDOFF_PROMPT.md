# FOOTBALL vNext — Project Context Handoff

Paste this entire document as your first message to any AI coding assistant
to continue this project with full context.

## Your role (assign this to the AI)

Act as CTO and Quantitative Lead for FOOTBALL vNext. Full ownership of
software architecture, technology stack, data pipelines, mathematical
algorithms, and financial/staking strategy. All code must be
production-ready: modular, Clean Architecture, full type hints, explicit
error handling, structured logging. No stub/placeholder functions unless
explicitly labeled as a temporary stand-in (see "Known stubs" below).

## Project goal

A data-driven prediction and value-betting system for football matches.
**The end goal is sustained, long-term profitability via statistical and
financial methodology — not just predicting match winners.** Every design
decision should be evaluated against this: does it improve real, durable
edge (validated by Closing Line Value), not just backtest ROI or model
accuracy.

## High-level pipeline (agreed algorithm, "Pipeline v2")

```
1.  Data Ingestion          -> historical results + xG (if available) + injuries/lineups
2.  Feature Engineering     -> time-decay weighting, per-team home advantage, form
3.  Dixon-Coles Fit         -> MLE for attack/defence + rho (fitted, not fixed)
4.  Raw Probability         -> lambda_home, lambda_away -> scoreline matrix
5.  Calibration/Shrinkage   -> raw_prob -> blend toward market prior (Bayesian blend,
                               weight fitted via log-loss minimization on historical data)
6.  Odds Aggregation        -> multiple bookmakers, median/best-price
7.  De-vig (Shin/Power)     -> fair market probability (NOT simple proportional —
                               proportional de-vig is biased against favorites)
8.  Edge Detection          -> Edge = shrunk_prob - fair_probability
9.  EV Check                -> EV = (shrunk_prob * odds) - 1
10. Market Efficiency Filter-> down-weight/exclude highly efficient leagues/markets
11. Filtering               -> Edge >= min_edge AND EV >= min_ev
12. Risk Score              -> bookmaker disagreement + data quality + sample size
13. Portfolio-Level Kelly   -> Fractional Kelly (~0.25-0.5x) accounting for correlation
                               between simultaneous bets (not independent per-bet Kelly)
14. Final Ranking           -> combine EV, Risk, Confidence
15. Suggestion Output       -> Bet, Odds, EV, Stake, Risk

=== GATE before real-money execution ===
16. Walk-Forward Backtest   -> train on past, test moving forward in time (never random split)
17. CLV Validation          -> does the model consistently beat the closing line?
                               This is the real success metric, more than raw ROI.
18. Statistical Significance-> t-test on ROI, need hundreds of bets minimum
19. Drawdown/Sharpe Check   -> is the risk tolerable with real capital?
    -> Only proceed to live betting if ALL FOUR pass.
```

## Tech stack (already decided)

- Python 3.11+
- Pydantic v2 for domain models and validation
- NumPy / SciPy for numerical optimization (MLE fitting via `scipy.optimize.minimize`)
- Pandas for data wrangling
- Streamlit for the current MVP dashboard (FastAPI + proper frontend planned for
  later production phase, not yet started)
- pytest for testing
- Clean Architecture: `domain/` (pure logic, no I/O) -> `application/` (use-cases,
  not yet built) -> `infrastructure/` (I/O, not yet built)

## What is actually implemented (Phase 1 only)

Repository structure:
```
football-vnext/
├── app.py                                          # Streamlit dashboard
├── pyproject.toml / requirements.txt
├── src/football_vnext/
│   ├── main.py                                     # CLI demo
│   ├── sample_data.py                              # SYNTHETIC data generator (temporary)
│   └── domain/
│       ├── models/{match,team,prediction,probability}.py  # Pydantic/dataclass domain models
│       ├── statistics/
│       │   ├── poisson.py                          # Independent Poisson base model
│       │   ├── dixon_coles.py                      # Dixon-Coles + MLE fit + time decay
│       │   └── calibration.py                      # Bayesian shrinkage (fitted weight)
│       ├── odds/
│       │   ├── models.py                           # BookmakerOdds (decimal odds, overround)
│       │   ├── aggregation.py                       # Multi-bookmaker median/best-price
│       │   └── devig.py                             # ProportionalDevig, ShinDevig, PowerDevig
│       └── value/
│           ├── edge.py                              # ValueBetCandidate, EdgeDetector
│           ├── kelly.py                             # single_bet_kelly_fraction, PortfolioKellyStaker
│           ├── market_efficiency.py                 # MarketEfficiencyClassifier (league tiers)
│           └── risk.py                              # RiskScoreCalculator
│       └── backtest/
│           ├── synthetic_odds.py                    # SyntheticOddsGenerator (STAND-IN, not real odds)
│           ├── historical_odds.py                    # HistoricalMatchOdds (real-data record)
│           ├── clv.py                                # BetRecord, CLVAnalyzer
│           ├── metrics.py                            # BacktestMetricsCalculator
│           ├── gate.py                                # BacktestGate (the go/no-go decision)
│           └── engine.py                             # WalkForwardBacktester (run() + run_with_historical_records())
│       └── features/
│           └── news_impact.py                        # NewsImpactAnalyzer (optional LLM signal, off by default)
│       └── infrastructure/
│           └── data_sources/
│               ├── football_data_org.py              # REAL fixtures/results adapter (tested against mocks)
│               ├── odds_api.py                       # REAL live odds adapter (tested against mocks)
│               ├── api_football.py                    # REAL 2nd source: fixtures+odds (tested against mocks)
│               ├── football_data_co_uk.py             # REAL historical results+odds archive (free, no key)
│               └── exceptions.py
│       └── application/
│           ├── data_loading.py                       # multi-source match data (fdo -> api-football -> synthetic)
│           └── odds_loading.py                       # multi-source odds (odds-api -> api-football -> example)
├── config.py                                          # Settings (all API keys from env)
├── scripts/
│   ├── verify_football_data_org.py                   # run once with a real API key
│   ├── verify_odds_api.py                            # run once with a real API key
│   ├── verify_football_data_co_uk.py                 # no key needed, public data
│   ├── verify_api_football.py                        # run once with a real API key
│   └── verify_news_impact.py                         # run with an Anthropic API key (network reachable, no key available)
├── run_backtest.py                                   # python -m football_vnext.run_backtest [--real E0 2223 2324]
└── tests/                                          # 167 tests, all passing
```

Verified working (actually run, not just written):
- `DixonColesEngine.fit()` — MLE optimization over attack/defence/rho/home-advantage
  with exponential time-decay weighting of historical matches. Uses SLSQP with an
  identifiability constraint (sum of attack params = number of teams).
- `DixonColesEngine.predict_match()` — full scoreline probability matrix with the
  tau() low-score correlation correction.
- `ProbabilityCalibrator` — fits a scalar blend weight `w` between model and market
  probability by minimizing log loss (NOT an arbitrary fixed weight). Includes
  Brier score, log loss, and Expected Calibration Error diagnostics, plus a
  holdout-evaluation method to catch overfitting of `w` itself.
- `ShinDevig` and `PowerDevig` — both solve for their latent parameter (z for Shin,
  c for Power) via `scipy.optimize.brentq` root-finding, constrained so fair
  probabilities sum to exactly 1. A dedicated test empirically confirms both methods
  shade less margin onto favorites (and more onto longshots) than naive proportional
  de-vig — the theoretically expected direction, verified numerically, not assumed.
- `EdgeDetector` — filters `ValueBetCandidate` objects requiring BOTH minimum Edge
  (model vs. fair-market probability gap) AND minimum EV (actual expected return at
  the offered odds) to pass, ranks by EV. Kept Edge and EV as separate, both-required
  checks deliberately — see the module docstring for why conflating them is a
  common mistake.
- `PortfolioKellyStaker` — standard single-bet Kelly (`f* = (bp-q)/b`), scaled by a
  configurable fraction (default 0.25x), then scaled down further at the *portfolio*
  level when multiple simultaneous signals are correlated (default: same match_id =
  fully correlated). Verified by test that correlated same-match bets get a smaller
  combined stake than the same bets treated as independent. This correlation
  adjustment is an explicitly-documented practical heuristic (variance-matching
  against a single-bet risk budget), not a full joint-distribution multivariate
  Kelly solve — flagged as such in the module docstring for whoever continues this.
- Streamlit app runs and serves (smoke-tested with actual HTTP requests, returned 200,
  including after every UI update through this phase; the button-click code path was
  also exercised directly as a plain script each time, to catch runtime errors
  Streamlit's page-load smoke test wouldn't).
- All 61 pytest tests pass, including a synthetic-recovery test that confirms the
  calibration optimizer actually recovers a known ground-truth blend weight, a
  directional test confirming Shin/Power de-vig behave as theoretically expected,
  a directional test confirming correlated bets get scaled down more than
  independent ones, and directional tests confirming risk score responds correctly
  to bookmaker disagreement, sample size, and league efficiency tier.
- `RiskScoreCalculator` — combines bookmaker disagreement (coefficient of variation
  across quotes for the same outcome), data quality (team sample size vs. a
  configurable minimum, using `DixonColesEngine.team_match_counts`, added in this
  phase), and market efficiency tier into one risk score in [0,1]. Used as both a
  hard filter and a stake dampener (composed at the application level, not inside
  kelly.py, to keep Kelly's math pure).
- `MarketEfficiencyClassifier` — starter registry classifying competitions into
  HIGH/MEDIUM/LOW efficiency tiers; unknown competitions default to MEDIUM rather
  than silently assumed inefficient.
- `WalkForwardBacktester` — runs the ENTIRE pipeline (fit → calibrate → de-vig →
  Edge/EV → Risk → Kelly) round by round, strictly forward in time: each round's
  bets are decided using a Dixon-Coles fit ONLY on matches strictly before that
  round's kickoff. Verified end-to-end on 120 rounds of synthetic data (607 bets
  placed, 12 rounds skipped for insufficient history, 0 matches skipped for unknown
  teams) — command and exact output logged in project history for reference.
- `BacktestGate` — the final go/no-go check. FOUR independent veto conditions (too
  few bets, CLV not significant, ROI <= 0, drawdown too deep) — any one failing
  blocks going live regardless of how good the others look. Verified by test with
  each veto condition individually and combined.
- `CLVAnalyzer` / `BacktestMetricsCalculator` — statistical significance testing
  (one-sided t-test) on both CLV and raw returns, with a documented, tested fix for
  the zero-variance edge case (identical bets aren't "untestable", they're either
  maximally significant or not significant depending on sign — verified by test).
- On a demo run (120 synthetic rounds, min_edge=0.02, min_ev=0.02, max_risk=0.6):
  607 bets, ROI +7.06% (not significant, p=0.058), mean CLV +68% (significant,
  p=0.011), max drawdown 18.7% → gate PASSED. The large CLV magnitude is a known
  artifact of the arbitrary noise scale in `SyntheticOddsGenerator`, not a realistic
  number — real CLV for a genuinely skilled bettor is typically low single digits
  percent. Don't quote this run's numbers as if they mean anything about real
  football markets; they only demonstrate the mechanics work.
- `FootballDataOrgAdapter` — fetches and parses matches from football-data.org's v4
  API into the domain `Match` model. Implements exponential backoff retry, 429
  rate-limit handling (honors `Retry-After` header), 401/403 → `AuthenticationError`,
  5xx → retried then `DataSourceUnavailableError`, and skips individual malformed
  match records (logged, not fatal) rather than aborting the whole fetch. **Tested
  against mocked HTTP responses only** — this dev sandbox's network egress does not
  include api.football-data.org and no API key was available to it. Run
  `scripts/verify_football_data_org.py` with a real key before trusting it live.
- `load_training_matches()` (application layer) — decides real vs. synthetic data,
  now called from BOTH `main.py` and `app.py` (not just built and left unused).
  Verified end-to-end: `main.py` run confirmed it correctly falls back to synthetic
  when no API key is present, dynamically picks demo teams from whichever data
  source was actually loaded (no more hard-coded "Arsenal vs Chelsea"), and the
  Streamlit app's team selectors now map display names to team IDs correctly (real
  data has numeric IDs distinct from names; synthetic data happens to have identical
  ID/name strings — the code does NOT special-case this, it always looks up by ID).
- `TheOddsApiAdapter` — fetches live multi-bookmaker decimal odds from The Odds API.
  Same retry/backoff/rate-limit/auth-error pattern as `FootballDataOrgAdapter`, plus
  per-bookmaker market-completeness checking (a bookmaker missing one of the three
  h2h outcomes is skipped, not treated as a crash). **Tested against mocked HTTP
  responses only** — same network limitation as football-data.org. Run
  `scripts/verify_odds_api.py` with a real key before trusting it live.
- `TeamNameMatcher` — reconciles odds-feed team name strings with fixture-provider
  team IDs via normalization (strips "FC"/"CF"/etc., case-folds) plus a kickoff-time
  tolerance window. Deliberately returns `None` rather than guess on ambiguous or
  no match — verified by test that it correctly rejects both wrong-team and
  right-team-wrong-time cases, and picks the right quote out of several candidates.
- `load_match_odds()` (application layer) — wired into `main.py`: fetches real odds
  when `ODDS_API_KEY` is set and a confident team match is found (aggregating
  across bookmakers via median), else falls back to a caller-supplied example quote.
  Verified end-to-end via `main.py` run (falls back correctly with no key configured).
- `FootballDataCoUkAdapter` — downloads and parses free, public historical CSV data
  (results + real bookmaker odds, including closing lines for many recent seasons)
  from football-data.co.uk (a DIFFERENT site from football-data.org, confusingly
  similar name — worth double-checking if something seems off). Confirmed the real
  column format via web search before implementing (B365H/B365D/B365A, AvgH/AvgD/AvgA,
  AvgCH/AvgCD/AvgCA, PSCH/PSCD/PSCA), with a fallback chain (Avg preferred over B365
  for opening; AvgC preferred over PSC for closing) since column availability varies
  by season. Tested against embedded realistic sample CSV text matching the real
  format — all 11 tests passed on the first run, which is a good sign the column
  research was accurate. **Still untested against a live download** (network
  limitation) — run `scripts/verify_football_data_co_uk.py` (no key needed).
- `WalkForwardBacktester.run_with_historical_records()` — new second entry point
  alongside `run()`, refactored to share the same core round-processing logic via
  a `_run_core()` method parameterized by an `OddsProvider` callable. Fixed a real
  bug caught during this work: round-grouping assumed every `Match` has a
  `matchday` int, which real CSV data (no matchday column) doesn't have — silently
  producing zero usable rounds. Now falls back to grouping by kickoff date when
  `matchday` is `None`, verified by a test that explicitly strips `matchday` from
  synthetic data and confirms bets still get placed.
- `python -m football_vnext.run_backtest --real E0 2223 2324 2425` — verified
  end-to-end with a realistic (Poisson-simulated, not degenerate) mocked CSV
  response: correctly parsed matches, ran the full pipeline, computed real CLV from
  real opening/closing odds, and the gate correctly failed on too few bets (65 <
  the 300 minimum) — proving the gate's veto logic engages correctly on genuinely
  small real-shaped data, not just synthetic data manufactured to pass.

## Known stubs / explicitly NOT real yet

- `sample_data.py` generates **synthetic** historical matches (Poisson-simulated),
  not real football data. `FootballDataOrgAdapter` + `load_training_matches()` are
  wired into `main.py`/`app.py` and used automatically when a real API key is
  configured — synthetic data is now only the fallback, not the only option.
- The example bookmaker odds constant in `main.py` is now ONLY a fallback (used
  when `ODDS_API_KEY` isn't set, or no live match is found for the given fixture)
  — `load_match_odds()` tries real odds first. The Streamlit UI's manual odds entry
  is a deliberate design choice for interactive testing, not an unwired stub.
- `run()` / `SyntheticOddsGenerator` still exist and are still useful for quickly
  testing pipeline mechanics without a network call — but
  `run_with_historical_records()` + `FootballDataCoUkAdapter` are now the path to
  an actually meaningful backtest result. Don't confuse the two when reading old
  backtest output — check which method/CLI flag produced it.
- `FootballDataCoUkAdapter`'s league-code coverage in its own docstring is
  illustrative (E0, E1, SP1, I1, D1, F1) — the site covers many more leagues;
  check https://www.football-data.co.uk/data.php for the full list and URL pattern
  per country before assuming a league isn't available.
- `_LEAGUE_REGISTRY` in `data_loading.py` (mapping a competition code to both
  football-data.org's code AND API-Football's numeric league_id) covers only 5
  major leagues (PL, PD, SA, BL1, FL1) — extend it before expecting the
  API-Football fallback to trigger for anything else.
- `ApiFootballAdapter.fetch_odds_for_fixture()` is only usable when the match
  itself came from API-Football (its odds endpoint is keyed by API-Football's own
  numeric fixture ID) — it CANNOT be used to fetch odds for a match that came from
  football-data.org or synthetic data. `odds_loading.py`'s
  `api_football_fixture_id` parameter defaults to `None` and that fallback is
  simply skipped unless the caller explicitly knows the match's API-Football ID.
- `NewsImpactAnalyzer` / `apply_news_adjustments()` require the caller to supply
  news text manually — there is NO automated news/injury-report fetching wired in.
  This is genuinely optional and OFF by default: `main.py` requires BOTH
  `ANTHROPIC_API_KEY` AND `ENABLE_NEWS_ADJUSTMENT=1` to activate it, and even then
  only demonstrates it with a hard-coded placeholder news string (clearly labeled
  as a stub in the code) — replace that with a real news source before this
  feature does anything useful in practice.
- Unlike every other adapter in this project, `NewsImpactAnalyzer`'s network path
  (api.anthropic.com) WAS reachable from the dev sandbox that built this — it just
  lacked a valid API key, so live testing still wasn't possible there. If you have
  an Anthropic key, `scripts/verify_news_impact.py` should work for real, unlike
  the other verify scripts which are blocked by network restrictions specifically.
- The bookmaker odds fed into `main.py` and `app.py` are a **hand-entered example**
  (or randomly perturbed around it for the calibration training set) — there is no
  live odds feed yet. The de-vig *math* applied to these odds (Shin's method) is
  real, not a stub — only the input odds are placeholders.
- Only one bookmaker quote is ever available in the demo, so bookmaker disagreement
  in the Risk Score is always reported at a fixed moderate value (0.5) — this needs
  real multi-bookmaker data (via `OddsAggregator`) to be meaningful.
- `MarketEfficiencyClassifier`'s league registry is illustrative, not exhaustive —
  extend `_HIGH_EFFICIENCY_LEAGUES` / `_LOW_EFFICIENCY_LEAGUES` with real competition
  names once real data is wired in.
- `SyntheticOddsGenerator` (used only by the backtester) is a documented stand-in
  for a real historical odds archive — it generates noisy opening/closing quotes
  around a KNOWN ground-truth probability that only exists because the underlying
  match data is itself synthetic. A real backtest needs real historical opening AND
  closing odds per match.
- **Calibration-fitting leakage** (documented in `engine.py`'s module docstring):
  the calibration blend weight is refit each round using that round's own fitted
  Dixon-Coles engine to retrodict probabilities for matches already in its own
  training window. This is a mild, deliberate simplification for the calibration
  weight specifically — the actual BETS in each round only ever use strictly-prior
  data (true walk-forward), but the calibration weight itself has minor in-sample
  leakage. Harden this (nested walk-forward, held-out folds) before trusting this
  module's backtest results on real data.
- `FootballDataOrgAdapter` verified only against mocked responses (see above) —
  live-API connectivity is unverified until someone runs
  `scripts/verify_football_data_org.py` with a real key.
- No database/persistence layer yet (in-memory only).
- No production orchestration (Docker, scheduler, monitoring) — Phase 5, not started.

## Immediate next step

The Pipeline v2 (stages 1-19) is implemented end-to-end. Data source redundancy
is now in place on every front:
- Fixtures/results: football-data.org → **API-Football (2nd source)** → synthetic
- Live odds: The Odds API → **API-Football odds (2nd source)** → example
- Historical odds (backtesting): football-data.co.uk (real, free, public)
- **Optional LLM auxiliary signal** (news/injury impact via Claude) — off by
  default everywhere, requires BOTH `ANTHROPIC_API_KEY` and
  `ENABLE_NEWS_ADJUSTMENT=1` to activate in `main.py`

Run `python -m football_vnext.run_backtest --real E0 2223 2324 2425` for a
backtest + gate decision on REAL Premier League data across three real seasons.
What's left:

1. Harden the calibration-fitting leakage noted below (nested walk-forward or
   held-out folds instead of in-sample retrodiction) — now more important than
   ever since real data is actually being fed through it.
2. Add persistence (PostgreSQL) — everything currently runs in-memory per process.
3. Production orchestration (Docker, scheduler, monitoring) — Phase 5 on the
   original roadmap, not started.
4. Extend `FootballDataCoUkAdapter`'s league-code coverage,
   `_LEAGUE_REGISTRY`'s (data_loading.py) provider mappings, and
   `MarketEfficiencyClassifier`'s registry beyond their current illustrative sets.
5. If the LLM news signal proves useful in practice, add an automated news/injury
   fetch source (currently the CALLER supplies the news text manually — there is
   no automated scraping/fetching of team news yet) — same adapter pattern as
   everything else in this project.

**Do not skip step 1 before trusting any real backtest gate result** — it's a
real, documented limitation, not a cosmetic one. Now that a real backtest is
actually runnable, this matters more, not less.

## Conventions to follow for all new code

- Full type hints everywhere.
- Pydantic models for all structured data crossing a boundary.
- Custom exception classes per module (e.g. `DixonColesFitError`,
  `CalibrationError`, `InsufficientDataError`) rather than bare `ValueError`
  for domain-specific failures.
- `logging.getLogger(__name__)` in every module — **never** attach handlers
  inside library modules, only in the application entry point
  (`main.py` / `app.py` via `logging.basicConfig`). Attaching handlers in a
  library module caused duplicate log lines once already — don't repeat that.
- Every new module needs corresponding tests in `tests/`, and must actually be
  run (not just written) to confirm it works before being considered done.
- No look-ahead bias: any fitting/calibration on historical data must respect
  chronological order (fit on past, evaluate on a later holdout).
