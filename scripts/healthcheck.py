"""Local environment/API configuration health check. Never prints secret values."""
from __future__ import annotations
import os
from football_vnext.config import Settings

def main():
    s=Settings.from_env()
    checks={
      "ODDS_API_KEY": s.has_odds_api_key,
      "API_FOOTBALL_KEY": s.has_api_football_key,
      "ANTHROPIC_API_KEY": s.has_anthropic_api_key,
      "DATABASE_URL": bool(os.getenv("DATABASE_URL")),
    }
    print("FOOTBALL vNext health check")
    for k,v in checks.items(): print(f"{k}: {'SET' if v else 'NOT SET'}")
    print("\nRequired for daily odds: ODDS_API_KEY")
    print("Optional: API_FOOTBALL_KEY (injuries/lineups), ANTHROPIC_API_KEY (LLM news), DATABASE_URL (PostgreSQL; SQLite is default)")
    raise SystemExit(0 if checks["ODDS_API_KEY"] else 2)

if __name__=="__main__": main()
