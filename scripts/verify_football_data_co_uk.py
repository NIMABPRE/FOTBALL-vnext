"""
Run this to confirm the football-data.co.uk adapter works against the real
site (it was only tested against embedded sample CSV text during development
— see infrastructure/data_sources/football_data_co_uk.py for why). No API key
needed — the data is public.

Usage:
    python scripts/verify_football_data_co_uk.py [LEAGUE_CODE] [SEASON]

LEAGUE_CODE defaults to "E0" (Premier League). SEASON defaults to "2425"
(2024/25). Other league codes: "SP1" (La Liga), "I1" (Serie A), "D1"
(Bundesliga), "F1" (Ligue 1), "E1" (Championship).
"""

from __future__ import annotations

import sys

from football_vnext.infrastructure.data_sources.exceptions import DataSourceError
from football_vnext.infrastructure.data_sources.football_data_co_uk import FootballDataCoUkAdapter


def main() -> None:
    league_code = sys.argv[1] if len(sys.argv) > 1 else "E0"
    season = sys.argv[2] if len(sys.argv) > 2 else "2425"

    print(f"Downloading {league_code} season {season} from football-data.co.uk...")
    adapter = FootballDataCoUkAdapter()

    try:
        records = adapter.fetch_historical_matches(league_code, season)
    except DataSourceError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    print(f"\nSUCCESS: parsed {len(records)} matches.\n")

    with_opening = sum(1 for r in records if r.opening_odds is not None)
    with_closing = sum(1 for r in records if r.closing_odds is not None)
    print(f"  {with_opening}/{len(records)} have opening odds")
    print(f"  {with_closing}/{len(records)} have closing odds")

    if records:
        sample = records[-1]
        print(f"\nMost recent match: {sample.match.home_team_name} "
              f"{sample.match.result.home_goals if sample.match.result else '?'} - "
              f"{sample.match.result.away_goals if sample.match.result else '?'} "
              f"{sample.match.away_team_name}")
        if sample.opening_odds:
            print(f"  Opening odds ({sample.opening_odds.bookmaker}): "
                  f"{sample.opening_odds.home}/{sample.opening_odds.draw}/{sample.opening_odds.away}")
        if sample.closing_odds:
            print(f"  Closing odds ({sample.closing_odds.bookmaker}): "
                  f"{sample.closing_odds.home}/{sample.closing_odds.draw}/{sample.closing_odds.away}")


if __name__ == "__main__":
    main()
