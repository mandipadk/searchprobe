"""Exception hierarchy for Aris.

All framework exceptions inherit from ArisError, enabling callers to
catch broad or specific failure classes as needed.
"""

from __future__ import annotations


class ArisError(Exception):
    """Base exception for all Aris errors."""


class ConfigurationError(ArisError):
    """Invalid configuration or missing required settings."""


class SourceError(ArisError):
    """Data source failure (network, auth, unexpected response)."""

    def __init__(
        self,
        message: str,
        *,
        source: str = "",
        status_code: int | None = None,
    ) -> None:
        self.source = source
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(SourceError):
    """Source rate limit hit -- signals that retry with backoff is appropriate."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        source: str = "",
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, source=source, status_code=429)


class RetrievalError(ArisError):
    """Retrieval pipeline failure."""

    def __init__(self, message: str, *, stage: str | None = None) -> None:
        self.stage = stage
        super().__init__(message)


class VerificationError(ArisError):
    """Constraint verification failure."""


class RankingError(ArisError):
    """Ranking pipeline failure."""


class QueryParsingError(ArisError):
    """Query understanding / parsing failure."""


class IndexError(ArisError):
    """Document indexing failure."""
