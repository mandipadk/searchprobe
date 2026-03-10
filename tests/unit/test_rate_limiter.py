"""Tests for pipeline rate limiter."""

import asyncio
import time

import pytest

from searchprobe.pipeline.rate_limiter import RateLimiterPool, TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    def test_init(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=20)
        assert limiter.rate == 10.0
        assert limiter.burst == 20
        assert limiter.tokens == 20.0

    def test_available_tokens_snapshot(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)
        assert limiter.available_tokens == 5.0

    @pytest.mark.asyncio
    async def test_acquire_single(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)
        await limiter.acquire(1)
        assert limiter.available_tokens == pytest.approx(4.0, abs=0.2)

    @pytest.mark.asyncio
    async def test_acquire_multiple(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)
        await limiter.acquire(3)
        assert limiter.available_tokens == pytest.approx(2.0, abs=0.2)

    @pytest.mark.asyncio
    async def test_acquire_all_burst(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=3)
        await limiter.acquire(3)
        assert limiter.available_tokens < 1.0

    @pytest.mark.asyncio
    async def test_try_acquire_success(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)
        result = await limiter.try_acquire(3)
        assert result is True

    @pytest.mark.asyncio
    async def test_try_acquire_fail(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=2)
        result = await limiter.try_acquire(5)
        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_waits_and_refills(self):
        limiter = TokenBucketRateLimiter(rate=100.0, burst=1)
        await limiter.acquire(1)
        # Bucket is empty, next acquire must wait for refill
        start = time.monotonic()
        await limiter.acquire(1)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.005  # Should have waited for refill

    @pytest.mark.asyncio
    async def test_concurrent_acquire_doesnt_deadlock(self):
        """Verify lock is released during sleep so other coroutines proceed."""
        limiter = TokenBucketRateLimiter(rate=100.0, burst=2)

        async def acquire_one():
            await limiter.acquire(1)

        # Launch 3 concurrent acquires with burst=2 — should complete, not deadlock
        await asyncio.wait_for(
            asyncio.gather(acquire_one(), acquire_one(), acquire_one()),
            timeout=2.0,
        )


class TestRateLimiterPool:
    def test_get_limiter_creates_on_demand(self):
        pool = RateLimiterPool()
        limiter = pool.get_limiter("exa")
        assert limiter.rate == 10.0
        assert limiter.burst == 20

    def test_get_limiter_reuses(self):
        pool = RateLimiterPool()
        limiter1 = pool.get_limiter("exa")
        limiter2 = pool.get_limiter("exa")
        assert limiter1 is limiter2

    def test_get_limiter_unknown_provider(self):
        pool = RateLimiterPool()
        limiter = pool.get_limiter("unknown_provider")
        assert limiter.rate == 5.0
        assert limiter.burst == 10

    @pytest.mark.asyncio
    async def test_pool_acquire(self):
        pool = RateLimiterPool()
        await pool.acquire("exa")
        limiter = pool.get_limiter("exa")
        assert limiter.available_tokens < 20.0
