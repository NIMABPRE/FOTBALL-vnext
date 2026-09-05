"""
Run this ONCE with a real football-data.org API key to confirm the adapter
actually works against the live API (it was only tested against mocked
responses during development — see infrastructure/data_sources/football_data_org.py
for why).

Get a free API key: https://www.football-data.org/client/register

Usage:
    export FOOTBALL_DATA_API_KEY=your_key_here
    python scripts/verify_football_data_org.py [COMPETITION_CODE]

COMPETITION_CODE defaults to "PL" (Premier League). Other examples: "SA"
(Serie A), "BL1" (Bundesliga), "PD" (La Liga), "FL1" (Ligue 1).
"""

from __future__ import annotations

import os
import sys

from football_vnext.infrastructure.data_sources.exceptions import DataSourceError
from football_vnext.infrastructure.data_sources.football_data_org import FootballDataOrgAdapter


def main() -> None:
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print("ERROR: set the FOOTBALL_DATA_API_KEY environment variable first.")
        print("Get a free key at: https://www.football-data.org/client/register")
        sys.exit(1)

    competition_code = sys.argv[1] if len(sys.argv) > 1 else "PL"

    print(f"Fetching matches for competition '{competition_code}' from the live API...")
    adapter = FootballDataOrgAdapter(api_key=api_key)

    try:
        matches = adapter.fetch_matches(competition_code)
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


if __name__ == "__main__":
    main()
