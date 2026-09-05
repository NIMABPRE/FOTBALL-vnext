from __future__ import annotations

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """
    Application configuration, sourced from environment variables.
    Config-driven per the project's architecture principles — no API keys
    hard-coded in business logic.
    """

    football_data_api_key: Optional[str] = None
    odds_api_key: Optional[str] = None
    api_football_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    @property
    def has_football_data_api_key(self) -> bool:
        return bool(self.football_data_api_key)

    @property
    def has_odds_api_key(self) -> bool:
        return bool(self.odds_api_key)

    @property
    def has_api_football_key(self) -> bool:
        return bool(self.api_football_key)

    @property
    def has_anthropic_api_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        def _clean(value: Optional[str]) -> Optional[str]:
            return value.strip() if value and value.strip() else None

        return cls(
            football_data_api_key=_clean(os.environ.get("FOOTBALL_DATA_API_KEY")),
            odds_api_key=_clean(os.environ.get("ODDS_API_KEY")),
            api_football_key=_clean(os.environ.get("API_FOOTBALL_KEY")),
            anthropic_api_key=_clean(os.environ.get("ANTHROPIC_API_KEY")),
        )
