"""Base class for Aris data sources with resilience."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aris.core.exceptions import RateLimitError, SourceError
from aris.core.models import Document

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-source circuit breaker."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def record_success(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker opened after %d failures", self.failure_count)

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True


class DataSource(ABC):
    """Abstract base class for all data sources."""

    NAME: str = "base"

    def __init__(self, max_retries: int = 3) -> None:
        self._circuit_breaker = CircuitBreaker()
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return self.NAME

    @abstractmethod
    async def _fetch(self, query: str, num_results: int) -> list[Document]:
        """Fetch documents from the source. Implement in subclasses."""
        ...

    async def search(self, query: str, num_results: int = 10) -> list[Document]:
        """Search with automatic retry and circuit breaker protection."""
        if not self._circuit_breaker.can_execute():
            logger.warning("Circuit breaker open for %s, skipping", self.NAME)
            return []

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                retry=retry_if_exception_type((SourceError, RateLimitError)),
                reraise=True,
            ):
                with attempt:
                    results = await self._fetch(query, num_results)
                    self._circuit_breaker.record_success()
                    return results
        except RetryError as exc:
            self._circuit_breaker.record_failure()
            last = exc.last_attempt
            error_msg = str(last.exception()) if last.exception() else "Unknown error"
            logger.warning("Source %s failed after %d retries: %s", self.NAME, self._max_retries, error_msg)
            return []
        except Exception as exc:
            self._circuit_breaker.record_failure()
            logger.warning("Source %s unexpected error: %s", self.NAME, exc)
            return []

        return []

    async def close(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass
