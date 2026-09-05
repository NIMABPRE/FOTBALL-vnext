from __future__ import annotations


class DataSourceError(Exception):
    """Base class for all data source adapter errors."""


class AuthenticationError(DataSourceError):
    """Raised when the API key is missing, invalid, or rejected (401/403)."""


class RateLimitExceededError(DataSourceError):
    """Raised when the API rate limit is hit and retries are exhausted."""


class DataSourceUnavailableError(DataSourceError):
    """Raised when the API returns a server error (5xx) after all retries."""
