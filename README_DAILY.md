# FOOTBALL vNext — Integrated Daily System

## What is now integrated

1. **Dixon-Coles + exponential time decay + MLE**
2. **xG correction layer** from public Understat league data, bounded so xG cannot dominate the statistical model.
3. **Chronological calibration** — the calibration model is trained on earlier data and evaluated on later observations; the final daily model is not calibrated against its own in-sample predictions.
4. **Live 1X2 odds** from The Odds API with median multi-bookmaker aggregation and Shin de-vig.
5. **Edge + EV + market-efficiency filtering**.
6. **Risk score + fractional portfolio Kelly**.
7. **Structured injuries** from API-Football when `API_FOOTBALL_KEY` is configured.
8. **Confirmed lineups** from API-Football when a fixture can be matched; injury reports are cross-checked against the actual XI to reduce double-counting.
9. **Automated team news** from Google News RSS, optionally converted to a bounded +/-15% LLM adjustment through the existing `NewsImpactAnalyzer` when `ANTHROPIC_API_KEY` is configured.
10. **Persistence**: local SQLite by default; PostgreSQL supported through `DATABASE_URL`.
11. **Daily automation**: `scripts/daily_job.py` can be called from Windows Task Scheduler, cron, or a VPS. `schedule_daily.py` is a simple fallback loop.
12. **Walk-forward calibration hardening** in the backtest engine: the calibrator no longer retrodicts the same observations used to fit the final model.

## Run locally on Windows

```text
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Or run `run_dashboard.bat` if present.

## API keys

- `ODDS_API_KEY`: required for today's bookmaker prices.
- `API_FOOTBALL_KEY`: optional; adds injuries and confirmed lineups.
- `ANTHROPIC_API_KEY`: optional; adds automated news extraction/assessment.
- `DATABASE_URL`: optional; omit for local SQLite, set a PostgreSQL URL for production.

Do not commit real API keys to Git.

## Automated daily job

```text
set PYTHONPATH=%CD%\\src
.venv\\Scripts\\python.exe scripts\\daily_job.py --league "Premier League"
```

For Windows Task Scheduler, create a daily task that starts in the project directory and runs the command above. For a server, use cron/systemd instead.

## Important model discipline

The new context sources are **bounded corrections**, not replacements for Dixon-Coles. Missing external data is fail-closed: the system does not invent xG, injuries, lineups, or news.

Before real-money execution, run the real historical walk-forward gate and require positive/stable CLV, statistically credible performance, controlled drawdown and robustness across seasons. A daily prediction with positive EV is not by itself evidence of profitability.
