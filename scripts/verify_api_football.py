"""
Run this ONCE with a real API-Football key to confirm the adapter works
against the live API (it was only tested against constructed sample JSON
during development).

Get a free API key: https://www.api-football.com (free tier: 100 requests/day)

Usage:
    export API_FOOTBALL_KEY=your_key_here
    python scripts/verify_api_football.py [LEAGUE_ID] [SEASON]

LEAGUE_ID defaults to 39 (Premier League). Other examples: 140 (La Liga),
135 (Serie A), 78 (Bundesliga), 61 (Ligue 1). SEASON defaults to 2024
(2024/25 season).
"""

from __future__ import annotations

import os
import sys

from football_vnext.infrastructure.data_sources.api_football import ApiFootballAdapter
from football_vnext.infrastructure.data_sources.exceptions import DataSourceError


def main() -> None:
    api_key = os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        print("ERROR: set the API_FOOTBALL_KEY environment variable first.")
        print("Get a free key at: https://www.api-football.com")
        sys.exit(1)

    league_id = int(sys.argv[1]) if len(sys.argv) > 1 else 39
    season = int(sys.argv[2]) if len(sys.argv) > 2 else 2024

    print(f"Fetching fixtures for league_id={league_id} season={season}...")
    adapter = ApiFootballAdapter(api_key=api_key)

    try:
        matches = adapter.fetch_fixtures(league_id, season)
    except DataSourceError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    print(f"\nSUCCESS: fetched {len(matches)} matches.\n")
    finished = [m for m in matches if m.result is not None]
    print(f"  {len(finished)} finished, {len(matches) - len(finished)} not yet played")

    if finished:
        sample = finished[-1]
        print(f"\nMost recent finished match:")
        print(f"  {sample.home_team_name} {sample.result.home_goals} - "
              f"{sample.result.away_goals} {sample.away_team_name} ({sample.kickoff.date()})")

        print(f"\nFetching odds for fixture {sample.match_id}...")
        quote = adapter.fetch_odds_for_fixture(int(sample.match_id))
        if quote is None:
            print("  No odds available for this fixture (may be too old).")
        else:
            print(f"  {len(quote.bookmaker_quotes)} bookmaker(s) available, e.g.: {quote.bookmaker_quotes[0]}")


if __name__ == "__main__":
    main()
