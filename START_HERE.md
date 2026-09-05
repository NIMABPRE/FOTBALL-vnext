# FOOTBALL vNext — Start Here

## One-click startup (recommended)

After the first setup, you do **not** need to activate `.venv` manually every day.

1. Put your API keys in `.env` in the project root.
2. Double-click `START_VNEXT.bat`.
3. The launcher automatically:
   - uses/creates `.venv`;
   - installs dependencies if they have not been installed yet;
   - sets the project `PYTHONPATH`;
   - runs the health check;
   - starts Streamlit;
   - opens `http://localhost:8501` in the browser.

You can also double-click `run_dashboard.bat`; it simply calls `START_VNEXT.bat`.

## Configure keys

Copy `.env.example` to `.env` and fill in `ODDS_API_KEY`.
Optional: `API_FOOTBALL_KEY` enables injuries + confirmed lineups. `ANTHROPIC_API_KEY` enables LLM news analysis. `DATABASE_URL` can point to PostgreSQL; otherwise SQLite is used.

**Never commit or share `.env`.**

## Manual startup (optional)

```powershell
$env:PYTHONPATH="$PWD\src"
.venv\Scripts\python.exe -m streamlit run app.py
```

## Daily automation

`run_daily.bat` runs the prediction pipeline and stores predictions. For CLV, schedule `scripts/snapshot_odds.py` every 10–15 minutes during the pre-match window so the database retains the latest pre-kickoff price.

## Real-money gate

Do not treat a single day's Value Bets as proof of profitability. Run the real-data walk-forward backtest, calibration checks, CLV analysis, statistical significance, drawdown/Sharpe and the Gate before using live bankroll.
