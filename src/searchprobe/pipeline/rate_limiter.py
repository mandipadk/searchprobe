"""Rate limiting for API calls."""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling API request rates.

    The token bucket algorithm allows for burst traffic while
    maintaining an average rate limit over time.
    """

    rate: float  # Tokens (requests) per second
    burst: int  # Maximum burst size (bucket capacity)
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        """Initialize tokens and timestamp."""
        self.tokens = float(self.burst)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.burst, self.tokens + tokens_to_add)
        self.last_refill = now

    async def acquire(self, tokens: int = 1) -> None:
        """Wait until the requested number of tokens is available.

        Args:
            tokens: Number of tokens to acquire (default 1)
        """
        async with self._lock:
            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate

                # Release lock while waiting
                await asyncio.sleep(wait_time)

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    @property
    def available_tokens(self) -> float:
        """Get current number of available tokens."""
        self._refill()
        return self.tokens


class RateLimiterPool:
    """Pool of rate limiters for multiple providers."""

    # Default rate limits per provider (requests per second)
    DEFAULT_RATES = {
        "exa": 10.0,
        "tavily": 10.0,
        "brave": 10.0,
        "serpapi": 5.0,
    }

    # Default burst sizes
    DEFAULT_BURSTS = {
        "exa": 20,
        "tavily": 20,
        "brave": 20,
        "serpapi": 10,
    }

    def __init__(self) -> None:
        """Initialize the pool."""
        self._limiters: dict[str, TokenBucketRateLimiter] = {}

    def get_limiter(self, provider: str) -> TokenBucketRateLimiter:
        """Get or create a rate limiter for a provider.

        Args:
            provider: Provider name

        Returns:
            Rate limiter for the provider
        """
        if provider not in self._limiters:
            rate = self.DEFAULT_RATES.get(provider, 5.0)
            burst = self.DEFAULT_BURSTS.get(provider, 10)
            self._limiters[provider] = TokenBucketRateLimiter(rate=rate, burst=burst)

        return self._limiters[provider]

    async def acquire(self, provider: str, tokens: int = 1) -> None:
        """Acquire tokens from a provider's rate limiter.

        Args:
            provider: Provider name
            tokens: Number of tokens to acquire
        """
        limiter = self.get_limiter(provider)
        await limiter.acquire(tokens)
