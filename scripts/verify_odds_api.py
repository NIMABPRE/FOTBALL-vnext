"""
Run this ONCE with a real Odds API key to confirm the adapter actually works
against the live API (it was only tested against mocked responses during
development — see infrastructure/data_sources/odds_api.py for why).

Get a free API key: https://the-odds-api.com (free tier: 500 requests/month)

Usage:
    export ODDS_API_KEY=your_key_here
    python scripts/verify_odds_api.py [SPORT_KEY]

SPORT_KEY defaults to "soccer_epl". Other examples: "soccer_spain_la_liga",
"soccer_italy_serie_a", "soccer_germany_bundesliga".
"""

from __future__ import annotations

import os
import sys

from football_vnext.infrastructure.data_sources.exceptions import DataSourceError
from football_vnext.infrastructure.data_sources.odds_api import TheOddsApiAdapter


def main() -> None:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ERROR: set the ODDS_API_KEY environment variable first.")
        print("Get a free key at: https://the-odds-api.com")
        sys.exit(1)

    sport_key = sys.argv[1] if len(sys.argv) > 1 else "soccer_epl"

    print(f"Fetching odds for '{sport_key}' from the live API...")
    adapter = TheOddsApiAdapter(api_key=api_key, sport_key=sport_key)

    try:
        quotes = adapter.fetch_odds()
    except DataSourceError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    print(f"\nSUCCESS: fetched odds for {len(quotes)} upcoming matches.\n")
    for quote in quotes[:5]:
        n_books = len(quote.bookmaker_quotes)
        best = quote.bookmaker_quotes[0]
        print(
            f"  {quote.home_team_name} vs {quote.away_team_name} "
            f"({quote.commence_time.date()}) — {n_books} bookmaker(s), "
            f"e.g. {best.bookmaker}: {best.home}/{best.draw}/{best.away}"
        )
    if len(quotes) > 5:
        print(f"  ... and {len(quotes) - 5} more")


if __name__ == "__main__":
    main()
