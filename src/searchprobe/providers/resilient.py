"""Resilience layer: retry logic and circuit breaker for search providers.

Wraps any SearchProvider with automatic retries (exponential backoff) and
a per-provider circuit breaker that prevents hammering a failing service.
Uses tenacity (already in project dependencies).
"""

from __future__ import annotations

import logging
import time
from enum import Enum

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from searchprobe.core.exceptions import ProviderError, RateLimitError
from searchprobe.providers.base import SearchProvider
from searchprobe.providers.models import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Rejecting requests (service is down)
    HALF_OPEN = "half_open"  # Allowing a single probe request


class CircuitBreaker:
    """Per-provider circuit breaker.

    Transitions:
      CLOSED -> (failure_threshold reached) -> OPEN
      OPEN   -> (recovery_timeout elapsed) -> HALF_OPEN
      HALF_OPEN -> success -> CLOSED
      HALF_OPEN -> failure -> OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker closing (probe succeeded)")
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker opened after %d failures", self.failure_count
            )

    def can_execute(self) -> bool:
        """Check whether a request is allowed right now."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering half-open state (probing)")
                return True
            return False
        # HALF_OPEN -- allow exactly one request
        return True


# ---------------------------------------------------------------------------
# Resilient Provider Wrapper
# ---------------------------------------------------------------------------

class ResilientProvider:
    """Wraps a SearchProvider with retries + circuit breaker.

    Usage::

        base = ExaProvider(api_key="...")
        provider = ResilientProvider(base, max_retries=3)
        response = await provider.search(request)
    """

    def __init__(
        self,
        provider: SearchProvider,
        max_retries: int = 3,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.provider = provider
        self.max_retries = max_retries
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    # Expose provider metadata
    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def NAME(self) -> str:
        return self.provider.NAME

    @property
    def SUPPORTED_MODES(self) -> list[str]:
        return self.provider.SUPPORTED_MODES

    @property
    def COST_PER_QUERY(self) -> dict[str, float]:
        return self.provider.COST_PER_QUERY

    def get_cost(self, mode: str | None = None) -> float:
        return self.provider.get_cost(mode)

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Search with automatic retry and circuit breaker protection."""
        if not self.circuit_breaker.can_execute():
            return SearchResponse(
                provider=self.provider.NAME,
                search_mode=request.search_mode,
                query=request.query,
                results=[],
                cost_usd=0.0,
                error=f"Circuit breaker open for {self.provider.NAME}",
            )

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                retry=retry_if_exception_type((ProviderError, RateLimitError)),
                reraise=True,
            ):
                with attempt:
                    response = await self.provider.search(request)
                    # Treat error responses as failures worth retrying
                    if response.error:
                        raise ProviderError(
                            response.error,
                            provider=self.provider.NAME,
                        )
                    self.circuit_breaker.record_success()
                    return response

        except RetryError as exc:
            self.circuit_breaker.record_failure()
            last = exc.last_attempt
            error_msg = str(last.exception()) if last.exception() else "Unknown error after retries"
            logger.warning(
                "Provider %s failed after %d retries: %s",
                self.provider.NAME,
                self.max_retries,
                error_msg,
            )
            return SearchResponse(
                provider=self.provider.NAME,
                search_mode=request.search_mode,
                query=request.query,
                results=[],
                cost_usd=0.0,
                error=error_msg,
            )
        except (ProviderError, RateLimitError) as exc:
            self.circuit_breaker.record_failure()
            return SearchResponse(
                provider=self.provider.NAME,
                search_mode=request.search_mode,
                query=request.query,
                results=[],
                cost_usd=0.0,
                error=str(exc),
            )
        except Exception as exc:
            self.circuit_breaker.record_failure()
            return SearchResponse(
                provider=self.provider.NAME,
                search_mode=request.search_mode,
                query=request.query,
                results=[],
                cost_usd=0.0,
                error=str(exc),
            )

        # Unreachable but satisfies type checker
        raise ProviderError("Unexpected control flow", provider=self.provider.NAME)  # pragma: no cover

    async def health_check(self) -> bool:
        """Proxy health check to the wrapped provider."""
        return await self.provider.health_check()

    async def close(self) -> None:
        """Proxy close to the wrapped provider."""
        await self.provider.close()
