"""
Run this with a real Anthropic API key to test the news-impact analyzer for
real (unlike the sports data adapters, this one's network path IS reachable
from the dev sandbox that built this project -- it just lacked a valid key
there, so this specific integration was only verified with mocked responses,
not because of a network restriction).

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python scripts/verify_news_impact.py
"""

from __future__ import annotations

import os
import sys

from football_vnext.domain.features.news_impact import NewsAnalysisError, NewsImpactAnalyzer

EXAMPLE_CASES = [
    ("Arsenal", "Star striker ruled out for 6 weeks with a hamstring injury sustained in training. "
                "Backup forward also carrying a knock and is a doubt."),
    ("Chelsea", "Full squad available, no injuries or suspensions to report this week."),
    ("Liverpool", "Starting goalkeeper and both first-choice centre-backs all suspended after red cards "
                  "last match. Emergency loan signing has not yet registered."),
]


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set the ANTHROPIC_API_KEY environment variable first.")
        sys.exit(1)

    analyzer = NewsImpactAnalyzer(api_key=api_key)

    for team, news in EXAMPLE_CASES:
        print(f"\n--- {team} ---")
        print(f"News: {news}")
        try:
            result = analyzer.assess(team, news)
            print(f"attack_multiplier: {result.attack_multiplier:.3f}")
            print(f"defense_multiplier: {result.defense_multiplier:.3f}")
            print(f"confidence: {result.confidence:.2f}")
            print(f"reasoning: {result.reasoning}")
        except NewsAnalysisError as exc:
            print(f"FAILED: {exc}")


if __name__ == "__main__":
    main()
