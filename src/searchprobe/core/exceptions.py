"""Exception hierarchy for SearchProbe.

All framework exceptions inherit from SearchProbeError, enabling callers to
catch broad or specific failure classes as needed.
"""

from __future__ import annotations


class SearchProbeError(Exception):
    """Base exception for all SearchProbe errors."""


class ProviderError(SearchProbeError):
    """Search provider failure (network, auth, unexpected response)."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(ProviderError):
    """Provider rate limit hit -- signals that retry with backoff is appropriate."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        provider: str = "",
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, provider=provider, status_code=429)


class BudgetExhaustedError(SearchProbeError):
    """Cost budget exceeded."""

    def __init__(
        self,
        message: str = "Budget exhausted",
        *,
        spent: float = 0.0,
        limit: float = 0.0,
    ) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(message)


class EvaluationError(SearchProbeError):
    """LLM judge or scoring failure."""


class ConfigurationError(SearchProbeError):
    """Invalid configuration or missing required settings."""


class PipelineError(SearchProbeError):
    """Pipeline orchestration failure."""

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
    ) -> None:
        self.stage = stage
        super().__init__(message)
