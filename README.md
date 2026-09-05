# FOOTBALL vNext

Data-driven football prediction and value-betting system. See `/docs` roadmap
discussion (in project chat history) for the full 6-phase plan. This repo
currently implements **Phase 1**: the statistical engine (Dixon-Coles) plus
the Calibration/Shrinkage layer.

## What's actually implemented right now

- `src/football_vnext/domain/statistics/poisson.py` — base independent-Poisson scoreline model
- `src/football_vnext/domain/statistics/dixon_coles.py` — Dixon-Coles model with time-decay
  weighting and MLE-fitted attack/defence/rho parameters
- `src/football_vnext/domain/statistics/calibration.py` — Bayesian shrinkage layer that
  blends model probability with market probability, with the blend weight *fitted*
  (not guessed) on historical data, plus calibration diagnostics (log loss, Brier, ECE)
- `src/football_vnext/domain/odds/models.py` — `BookmakerOdds` domain model (decimal
  odds, implied probability, overround)
- `src/football_vnext/domain/odds/aggregation.py` — combine multiple bookmakers'
  quotes (median or best-price)
- `src/football_vnext/domain/odds/devig.py` — **De-vig engines**: `ProportionalDevig`
  (naive baseline), `ShinDevig` (Shin's 1992/1993 insider-trading model), and
  `PowerDevig` (power-transform method). Both Shin and Power solve for a latent
  parameter via root-finding (`scipy.optimize.brentq`) rather than assuming an even
  margin split — verified by test to shade less margin onto favorites than naive
  proportional de-vig, which is the documented direction of bookmakers' real bias.
- `src/football_vnext/domain/value/edge.py` — `ValueBetCandidate` (Edge and EV
  calculation per outcome) and `EdgeDetector` (filters candidates by minimum Edge
  AND minimum EV, ranks by EV).
- `src/football_vnext/domain/value/kelly.py` — `single_bet_kelly_fraction` (standard
  Kelly formula) and `PortfolioKellyStaker`, which scales the whole vector of
  per-bet Kelly stakes down when bets are correlated (default: same match_id =
  fully correlated), so simultaneous correlated bets don't get sized as if
  independent. This is a practical variance-matching heuristic, not a full
  joint-distribution multivariate Kelly solve — documented as such in the code.
- `src/football_vnext/domain/value/market_efficiency.py` — classifies a competition
  into HIGH/MEDIUM/LOW efficiency tiers (starter registry of top European leagues
  as HIGH; extend as needed), feeding a risk contribution into the Risk Score below.
- `src/football_vnext/domain/value/risk.py` — `RiskScoreCalculator` combines
  bookmaker disagreement (coefficient of variation across quotes), data quality
  (team sample size vs. a configurable minimum), and market efficiency tier into a
  single risk score in [0,1], with a hard acceptance threshold. Used both as a
  filter and as a stake dampener (`kelly_stake * (1 - risk_score)`, composed at the
  application level in main.py/app.py).
- `src/football_vnext/domain/backtest/` — **Walk-forward backtesting + the go/no-go
  gate**:
  - `synthetic_odds.py` — generates noisy opening/closing odds around a known
    ground-truth probability (only possible because the underlying data is
    synthetic — clearly documented as a stand-in for a real historical odds archive)
  - `clv.py` — `BetRecord` and `CLVAnalyzer`: tracks Closing Line Value per bet and
    tests whether the average CLV is statistically significantly positive
  - `metrics.py` — `BacktestMetricsCalculator`: ROI, max drawdown, a Sharpe-like
    ratio, win rate, and significance testing on returns
  - `gate.py` — `BacktestGate`: the final go/no-go check before real-money
    execution. Independently vetoes on too few bets, insignificant CLV, non-positive
    ROI, or excessive drawdown — any one failing criterion blocks going live
  - `engine.py` — `WalkForwardBacktester`: runs the entire pipeline (fit → calibrate
    → de-vig → Edge/EV → Risk → Kelly) round by round, strictly forward in time
    (a round's bets only ever see matches strictly before it)
  - `run_backtest.py` (top-level script) — `python -m football_vnext.run_backtest`
    runs a full backtest on synthetic data and prints the gate decision
- `src/football_vnext/sample_data.py` — synthetic historical matches, so the pipeline
  runs today without a live data source
- `src/football_vnext/infrastructure/data_sources/football_data_org.py` —
  **real data source adapter** for football-data.org's v4 API: retry with
  exponential backoff, rate-limit (429) handling honoring `Retry-After`, auth-error
  detection, and JSON→domain-model mapping that skips malformed records without
  aborting the whole fetch. Tested against mocked HTTP responses (this dev sandbox
  can't reach the real API) — see `scripts/verify_football_data_org.py` to confirm
  live connectivity yourself with a real API key.
- `src/football_vnext/config.py` / `src/football_vnext/application/data_loading.py` —
  **now wired into both `main.py` and `app.py`**: `load_training_matches()` uses
  football-data.org when `FOOTBALL_DATA_API_KEY` is set (env var, or typed into the
  Streamlit sidebar) and the fetch returns enough finished matches; otherwise falls
  back to synthetic data, always logged/displayed so the source is never ambiguous.
- `src/football_vnext/infrastructure/data_sources/odds_api.py` — **real live odds
  adapter** for The Odds API (multi-bookmaker decimal odds, free tier available):
  same retry/backoff/rate-limit pattern as the football-data.org adapter. Tested
  against mocked HTTP responses (same network limitation as above) — see
  `scripts/verify_odds_api.py` to confirm live connectivity yourself.
- `src/football_vnext/domain/odds/matching.py` — `TeamNameMatcher` reconciles an
  odds-feed quote (identified by team name strings) with a domain `Match`
  (identified by a fixture provider's team IDs) by normalized name + kickoff-time
  proximity. Deliberately conservative: returns no match rather than a low-
  confidence guess, since a wrong match would silently corrupt everything downstream.
- `src/football_vnext/application/odds_loading.py` — **wired into `main.py`**:
  `load_match_odds()` fetches real odds via The Odds API when `ODDS_API_KEY` is set
  and a confident team-name match is found, aggregating across bookmakers (median);
  otherwise falls back to a caller-supplied example quote, always logged. The
  Streamlit UI still uses manually-entered odds by design (useful for interactively
  testing "what if the price were X" scenarios), not because live odds aren't wired up.
- `src/football_vnext/infrastructure/data_sources/api_football.py` — **second real
  data source** (API-Football, api-sports.io) covering BOTH fixtures/results AND
  odds, used purely for redundancy: `load_training_matches()` tries football-data.org
  first, then API-Football, before falling back to synthetic; `load_match_odds()`
  tries The Odds API first, then API-Football's odds endpoint (only when the match
  itself came from API-Football, since that endpoint is keyed by API-Football's own
  numeric fixture ID). Confirmed the real v3 schema via research before implementing.
  Tested against constructed sample JSON matching that schema (same network
  limitation as the other adapters) — see `scripts/verify_api_football.py`.
- `src/football_vnext/domain/features/news_impact.py` — **optional LLM auxiliary
  signal** (Claude via the Anthropic API): turns unstructured team news/injury text
  into a small, bounded (+/-15%) attack/defence multiplier adjustment, applied
  explicitly on top of the Dixon-Coles lambdas — never replacing the statistical
  core, and OFF by default everywhere it's wired in (`main.py` requires both
  `ANTHROPIC_API_KEY` AND `ENABLE_NEWS_ADJUSTMENT=1` to activate it). A malformed
  or out-of-bounds LLM response is refused (raises `NewsAnalysisError`), never
  silently clamped into something plausible-looking. This is the one adapter whose
  network path IS reachable from the dev sandbox (api.anthropic.com), though no API
  key was available there — tested against mocked responses regardless; see
  `scripts/verify_news_impact.py` to test for real.
  free, public historical odds archive** adapter (football-data.co.uk — NOT the
  same site as football-data.org, confusingly similar name). Downloads CSV files
  with real match results AND real bookmaker odds, including opening (`Avg*` /
  `B365*` columns) and, for many recent seasons, real closing lines (`AvgC*` /
  `PSC*` columns) — this is what makes real CLV measurement possible, finally
  closing the "no historical odds" gap. Tested against embedded realistic sample
  CSV text (same network limitation as the other adapters) — see
  `scripts/verify_football_data_co_uk.py` (no API key needed, the data is public).
- `WalkForwardBacktester.run_with_historical_records()` — a **second, real-data**
  entry point alongside the original synthetic `run()`, sharing the same core
  pipeline logic. Takes a list of `HistoricalMatchOdds` (match + real opening/closing
  odds) and runs the exact same walk-forward pipeline on them. Also fixed a real bug
  found while wiring this up: round-grouping assumed a `matchday` field that real CSV
  sources don't have — now falls back to grouping by kickoff date, verified by test.
- `python -m football_vnext.run_backtest --real E0 2223 2324 2425` — runs the full
  backtest + gate decision on REAL historical Premier League data across three
  real seasons, no API key needed. This is genuinely meaningful output, unlike the
  synthetic default.
- `app.py` — Streamlit dashboard: view fitted parameters, enter example bookmaker
  odds, see de-vig output, run predictions, and get risk-filtered, risk-adjusted
  value-bet stake recommendations

## Not implemented yet (next phases)

- Persistence (PostgreSQL) — everything runs in-memory per process
- Production orchestration (Docker, scheduler, monitoring/alerting)
- Hardening the calibration-fitting leakage (see `engine.py` docstring)
- Extending `FootballDataCoUkAdapter` / `_LEAGUE_REGISTRY` beyond their current
  illustrative league coverage
- The news-impact LLM feature only analyzes text you supply — there is no
  automated news/injury-report fetching wired in yet (that would be a further
  real data source to add, following the same adapter pattern as the others)

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # or: pip install -r requirements.txt
```

**Optional environment variables** (all optional — everything falls back to
synthetic/example data if unset, always clearly logged):

| Variable | Source | Free tier | Used for |
|---|---|---|---|
| `FOOTBALL_DATA_API_KEY` | football-data.org | Yes | Fixtures/results (primary) |
| `API_FOOTBALL_KEY` | api-football.com | Yes, 100 req/day | Fixtures/results (fallback) + odds (fallback) |
| `ODDS_API_KEY` | the-odds-api.com | Yes, 500 req/month | Live odds (primary) |
| `ANTHROPIC_API_KEY` | console.anthropic.com | — | Optional LLM news-impact signal |
| `ENABLE_NEWS_ADJUSTMENT=1` | (no key, just a flag) | — | Must ALSO be set for the LLM signal to activate |

No key at all is needed for `football-data.co.uk` (public, free CSV downloads).

## Running things

**CLI demo** (fits the model, prints a prediction and calibration metrics to
the terminal):

```bash
python -m football_vnext.main
```

**Web dashboard** (interactive — pick two teams, see predictions):

```bash
streamlit run app.py
```

Then open the URL it prints (usually `http://localhost:8501`) in your browser.
Enter a football-data.org API key in the sidebar to use real data instead of
synthetic — it falls back automatically (with a visible notice) if the fetch
fails or returns too few finished matches.

**Walk-forward backtest + gate decision** (synthetic data — proves the mechanics,
not real edge):

```bash
python -m football_vnext.run_backtest
```

**Walk-forward backtest on REAL historical data + REAL historical odds** (free,
no API key needed — this is the meaningful version):

```bash
python -m football_vnext.run_backtest --real E0 2223 2324 2425
```

**Verify the football-data.org adapter against the real API** (requires a free
API key — this dev sandbox couldn't reach the real API, so this step was never
run against live data; do it once yourself before trusting the adapter):

```bash
export FOOTBALL_DATA_API_KEY=your_key_here  # get one: https://www.football-data.org/client/register
python scripts/verify_football_data_org.py PL
```

**Verify the odds API adapter against the real API** (same caveat as above):

```bash
export ODDS_API_KEY=your_key_here  # free tier: https://the-odds-api.com
python scripts/verify_odds_api.py soccer_epl
```

**Verify the football-data.co.uk adapter** (no API key needed — public data):

```bash
python scripts/verify_football_data_co_uk.py E0 2425
```

**Verify the API-Football adapter** (second real data source, free tier):

```bash
export API_FOOTBALL_KEY=your_key_here  # free tier: https://www.api-football.com
python scripts/verify_api_football.py 39 2024
```

**Verify the LLM news-impact analyzer** (optional feature — this one's network
path IS reachable from wherever you run it, since it's just the Anthropic API):

```bash
export ANTHROPIC_API_KEY=your_key_here
python scripts/verify_news_impact.py
```

**Run tests:**

```bash
pytest tests/ -v
```

## Project structure

```
football-vnext/
├── app.py                              # Streamlit UI
├── pyproject.toml
├── requirements.txt
├── src/football_vnext/
│   ├── main.py                         # CLI demo entry point
│   ├── sample_data.py                  # synthetic data generator (temporary)
│   └── domain/
│       ├── models/                     # Match, Team, Prediction (pydantic)
│       ├── statistics/
│       │   ├── poisson.py
│       │   ├── dixon_coles.py
│       │   └── calibration.py
│       ├── odds/
│       │   ├── models.py                           # BookmakerOdds (decimal odds, overround)
│       │   ├── aggregation.py                       # Multi-bookmaker median/best-price
│       │   └── devig.py                             # ProportionalDevig, ShinDevig, PowerDevig
│       └── value/
│           ├── edge.py                              # ValueBetCandidate, EdgeDetector
│           ├── kelly.py                             # single_bet_kelly_fraction, PortfolioKellyStaker
│           ├── market_efficiency.py                 # MarketEfficiencyClassifier (league tiers)
│           └── risk.py                              # RiskScoreCalculator (disagreement, data quality, efficiency)
│       └── features/
│           └── news_impact.py                        # NewsImpactAnalyzer (optional LLM signal, off by default)
│       ├── backtest/
│       │   ├── synthetic_odds.py                    # SyntheticOddsGenerator (opening/closing, STAND-IN)
│       │   ├── historical_odds.py                    # HistoricalMatchOdds (real-data record)
│       │   ├── clv.py                                # BetRecord, CLVAnalyzer
│       │   ├── metrics.py                            # BacktestMetricsCalculator (ROI, drawdown, Sharpe)
│       │   ├── gate.py                                # BacktestGate — the go/no-go decision
│       │   └── engine.py                             # WalkForwardBacktester (run() + run_with_historical_records())
│       └── infrastructure/
│           └── data_sources/
│               ├── football_data_org.py              # real fixtures/results API adapter
│               ├── odds_api.py                       # real live odds API adapter
│               ├── api_football.py                    # 2nd real source: fixtures + odds (redundancy)
│               ├── football_data_co_uk.py             # real historical results+odds archive (free)
│               └── exceptions.py
│       └── application/
│           ├── data_loading.py                       # real-vs-synthetic match data (wired into main.py & app.py)
│           └── odds_loading.py                       # real-vs-example odds (wired into main.py)
├── config.py                                          # Settings (API keys from env)
├── scripts/
│   ├── verify_football_data_org.py                   # run once with a real API key
│   ├── verify_odds_api.py                            # run once with a real API key
│   ├── verify_football_data_co_uk.py                 # no key needed, public data
│   ├── verify_api_football.py                        # run once with a real API key
│   └── verify_news_impact.py                         # run with an Anthropic API key
├── run_backtest.py (via `python -m football_vnext.run_backtest`)
└── tests/
```

## Putting this on GitHub

```bash
cd football-vnext
git init
git add .
git commit -m "Phase 1: Dixon-Coles engine + calibration layer"
git branch -M main
git remote add origin <your-empty-github-repo-url>
git push -u origin main
```

## Daily Web Dashboard

The root `app.py` is now a real-data daily dashboard. It does not use synthetic
fallbacks for live predictions.

**Data flow**

1. Historical training + historical opening odds: `football-data.co.uk` (public, no key)
2. Today's upcoming 1X2 fixtures + bookmaker prices: The Odds API (`ODDS_API_KEY`)
3. Dixon-Coles fit with time decay
4. Time-ordered calibration holdout
5. Median bookmaker aggregation
6. Shin de-vig
7. Edge + EV filtering, market-efficiency multiplier
8. Risk score and fractional portfolio Kelly
9. Dashboard shows only the selected calendar day in the selected timezone

**Windows quick start**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If the editable package installation is blocked by PyPI/network issues, this
app still bootstraps `src/` itself, so `streamlit run app.py` does not require
`pip install -e .`.

You can also double-click `run_dashboard.bat` after installing dependencies.

### Required user setup

- Obtain a The Odds API key and enter it in the sidebar.
- Select the league and timezone.
- Select 1–3 historical seasons (default: 2025/26, 2024/25, 2023/24).
- Set Edge / EV / Risk / Kelly thresholds.
- Open the dashboard each day and press **Refresh live data** when needed.

### Current intentional limitation

The daily dashboard currently covers the 1X2 market only. It does not yet pull
live injuries/lineups/xG/news into the daily prediction path. Those are separate
next integrations; the statistical core remains Dixon-Coles + market calibration.

## Integrated daily system (2026-09)

The current build also includes `README_DAILY.md` and integrates the previously missing production layers: Understat xG enrichment, API-Football injuries/confirmed lineups, automated public-news collection with bounded optional LLM scoring, local SQLite/PostgreSQL persistence, a daily job, and calibration leakage hardening. Live predictions remain fail-closed when an external source is unavailable.
