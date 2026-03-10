"""Tests for ResilientProvider retry and error handling."""

import pytest

from searchprobe.core.exceptions import ProviderError, RateLimitError
from searchprobe.providers.models import SearchRequest, SearchResponse, SearchResult
from searchprobe.providers.resilient import CircuitBreaker, CircuitState, ResilientProvider


class FakeProvider:
    """Fake search provider for testing."""

    NAME = "fake"
    SUPPORTED_MODES = ["auto"]
    COST_PER_QUERY = {"auto": 0.005}

    def __init__(self):
        self.call_count = 0
        self.responses: list[SearchResponse | Exception] = []

    @property
    def name(self):
        return self.NAME

    def get_cost(self, mode=None):
        return 0.005

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.call_count += 1
        if self.call_count <= len(self.responses):
            r = self.responses[self.call_count - 1]
            if isinstance(r, Exception):
                raise r
            return r
        return SearchResponse(
            provider="fake", query=request.query, results=[],
            cost_usd=0.005,
        )

    async def health_check(self):
        return True

    async def close(self):
        pass


def _ok_response(query: str = "test") -> SearchResponse:
    return SearchResponse(
        provider="fake", query=query,
        results=[SearchResult(title="T", url="http://example.com", snippet="S")],
        cost_usd=0.005,
    )


def _error_response(query: str = "test", error: str = "fail") -> SearchResponse:
    return SearchResponse(
        provider="fake", query=query, results=[], cost_usd=0.0, error=error,
    )


class TestResilientProviderSuccess:
    @pytest.mark.asyncio
    async def test_passes_through_success(self):
        fake = FakeProvider()
        fake.responses = [_ok_response()]
        provider = ResilientProvider(fake, max_retries=3)
        request = SearchRequest(query="test")
        response = await provider.search(request)
        assert response.success
        assert fake.call_count == 1

    @pytest.mark.asyncio
    async def test_exposes_provider_metadata(self):
        fake = FakeProvider()
        provider = ResilientProvider(fake)
        assert provider.NAME == "fake"
        assert provider.name == "fake"
        assert provider.SUPPORTED_MODES == ["auto"]


class TestResilientProviderRetry:
    @pytest.mark.asyncio
    async def test_retries_on_error_response(self):
        """Error responses are converted to ProviderError and retried."""
        fake = FakeProvider()
        fake.responses = [
            _error_response(error="temporary"),
            _error_response(error="temporary"),
            _ok_response(),
        ]
        provider = ResilientProvider(fake, max_retries=3)
        request = SearchRequest(query="test")
        response = await provider.search(request)
        assert response.success
        assert fake.call_count == 3

    @pytest.mark.asyncio
    async def test_returns_error_after_max_retries(self):
        fake = FakeProvider()
        fake.responses = [
            _error_response(error="fail1"),
            _error_response(error="fail2"),
            _error_response(error="fail3"),
        ]
        provider = ResilientProvider(fake, max_retries=3)
        request = SearchRequest(query="test")
        response = await provider.search(request)
        assert response.error is not None
        assert fake.call_count == 3


class TestResilientProviderExceptions:
    @pytest.mark.asyncio
    async def test_provider_error_returns_response(self):
        """ProviderError should return an error SearchResponse, not re-raise."""
        fake = FakeProvider()
        fake.responses = [
            ProviderError("API down", provider="fake"),
            ProviderError("API down", provider="fake"),
            ProviderError("API down", provider="fake"),
        ]
        provider = ResilientProvider(fake, max_retries=3)
        request = SearchRequest(query="test")
        # Should NOT raise — should return error response
        response = await provider.search(request)
        assert response.error is not None
        assert "API down" in response.error

    @pytest.mark.asyncio
    async def test_rate_limit_error_returns_response(self):
        """RateLimitError should return an error SearchResponse, not re-raise."""
        fake = FakeProvider()
        fake.responses = [
            RateLimitError("429", provider="fake"),
            RateLimitError("429", provider="fake"),
            RateLimitError("429", provider="fake"),
        ]
        provider = ResilientProvider(fake, max_retries=3)
        request = SearchRequest(query="test")
        response = await provider.search(request)
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_generic_exception_returns_response(self):
        fake = FakeProvider()
        fake.responses = [
            RuntimeError("unexpected"),
            RuntimeError("unexpected"),
            RuntimeError("unexpected"),
        ]
        provider = ResilientProvider(fake, max_retries=3)
        request = SearchRequest(query="test")
        response = await provider.search(request)
        assert response.error is not None


class TestResilientProviderCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        fake = FakeProvider()
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        provider = ResilientProvider(fake, max_retries=1, circuit_breaker=breaker)

        # Two failures to open the circuit
        fake.responses = [_error_response(error="fail")]
        await provider.search(SearchRequest(query="q1"))
        fake.call_count = 0
        fake.responses = [_error_response(error="fail")]
        await provider.search(SearchRequest(query="q2"))

        # Circuit should be open, next call should return immediately
        fake.call_count = 0
        fake.responses = [_ok_response()]
        response = await provider.search(SearchRequest(query="q3"))
        assert response.error is not None
        assert "Circuit breaker open" in response.error
        assert fake.call_count == 0  # Didn't even call the provider
