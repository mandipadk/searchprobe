"""Resilience layer: retry logic and circuit breaker for search providers.

Wraps any SearchProvider with automatic retries (exponential backoff) and
a per-provider circuit breaker that prevents hammering a failing service.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from enum import Enum

from searchprobe.core.exceptions import ProviderError, RateLimitError
from searchprobe.providers.base import SearchProvider, is_transient_error
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
        self._probe_in_flight = False

    def record_success(self) -> None:
        """Record a successful request."""
        self._probe_in_flight = False
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker closing (probe succeeded)")
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.state == CircuitState.HALF_OPEN:
            self._probe_in_flight = False
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker re-opened (probe failed)")
            return
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker opened after %d failures", self.failure_count)

    def can_execute(self) -> bool:
        """Check whether a request is allowed right now."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self._probe_in_flight = True
                logger.info("Circuit breaker entering half-open state (probing)")
                return True
            return False
        # HALF_OPEN -- allow exactly one probe at a time
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True


# ---------------------------------------------------------------------------
# Resilient Provider Wrapper
# ---------------------------------------------------------------------------


class ResilientProvider:
    """Wraps a SearchProvider with retries + circuit breaker.

    - Only transient failures are retried (rate limits, timeouts, 5xx,
      connection errors); auth and request errors fail immediately.
    - ``before_attempt`` (e.g. a rate limiter's ``acquire``) runs before every
      attempt, so retries are rate limited too.
    - ``RateLimitError.retry_after`` is honored.

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
        before_attempt: Callable[[], Awaitable[None]] | None = None,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        self.provider = provider
        self.max_retries = max_retries
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.before_attempt = before_attempt
        self.base_delay = base_delay
        self.max_delay = max_delay

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

    def estimate_cost(self, request: SearchRequest) -> float:
        return self.provider.estimate_cost(request)

    def _error_response(self, request: SearchRequest, message: str) -> SearchResponse:
        return SearchResponse(
            provider=self.provider.NAME,
            search_mode=request.search_mode,
            query=request.query,
            results=[],
            cost_usd=0.0,
            error=message,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Search with retries on transient failures and circuit-breaker protection."""
        if not self.circuit_breaker.can_execute():
            return self._error_response(request, f"Circuit breaker open for {self.provider.NAME}")

        last_error = "Unknown error"
        for attempt in range(1, self.max_retries + 1):
            if self.before_attempt is not None:
                await self.before_attempt()

            retry_after: float | None = None
            try:
                response = await self.provider.search(request)
            except RateLimitError as exc:
                last_error, transient, retry_after = str(exc), True, exc.retry_after
            except ProviderError as exc:
                last_error, transient = str(exc), is_transient_error(str(exc))
            except Exception as exc:  # network/SDK errors
                last_error, transient = str(exc), is_transient_error(str(exc))
            else:
                if not response.error:
                    self.circuit_breaker.record_success()
                    return response
                last_error, transient = response.error, is_transient_error(response.error)

            if not transient or attempt == self.max_retries:
                break
            delay = retry_after if retry_after else self.base_delay * 2 ** (attempt - 1)
            await asyncio.sleep(min(delay, self.max_delay))

        self.circuit_breaker.record_failure()
        logger.warning("Provider %s failed: %s", self.provider.NAME, last_error)
        return self._error_response(request, last_error)

    async def health_check(self) -> bool:
        """Proxy health check to the wrapped provider."""
        return await self.provider.health_check()

    async def close(self) -> None:
        """Proxy close to the wrapped provider."""
        await self.provider.close()
