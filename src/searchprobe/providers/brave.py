"""Brave Search API provider implementation."""

import time
from datetime import datetime
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx

from searchprobe.providers.base import SearchProvider
from searchprobe.providers.models import SearchRequest, SearchResponse, SearchResult


class BraveProvider(SearchProvider):
    """Brave Search API provider.

    Brave has an independent search index (not wrapping Google)
    with 30+ billion pages.
    """

    NAME: ClassVar[str] = "brave"
    SUPPORTED_MODES: ClassVar[list[str]] = ["web"]
    COST_PER_QUERY: ClassVar[dict[str, float]] = {
        "web": 0.005,
    }
    API_ENDPOINT: ClassVar[str] = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str) -> None:
        """Initialize Brave client."""
        super().__init__(api_key)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.api_key,
                },
            )
        return self._client

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute search using Brave Search API."""
        start_time = time.perf_counter()
        mode = request.search_mode or "web"

        try:
            client = await self._get_client()

            # Build request parameters
            params: dict[str, Any] = {
                "q": request.query,
                "count": min(request.num_results, 20),  # Brave max is 20 per request
            }

            # Execute search
            response = await client.get(self.API_ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse results
            results = []
            web_results = data.get("web", {}).get("results", [])

            for i, result in enumerate(web_results):
                # Extract domain
                url = result.get("url", "")
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.replace("www.", "")

                # Brave doesn't provide relevance scores, use position-based
                # Higher position = higher relevance
                score = 1.0 - (i / max(len(web_results), 1)) * 0.5  # 1.0 to 0.5 range

                # Get content (Brave provides description, not full content)
                snippet = result.get("description", "")
                content = None
                if request.include_content:
                    # Brave doesn't provide full page content in basic API
                    # Would need to fetch separately
                    content = snippet

                results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=url,
                        snippet=snippet,
                        content=content,
                        score=score,
                        published_date=None,
                        source_domain=domain,
                        position=i,
                        provider_raw=result,
                    )
                )

            # Calculate cost
            total_cost = self.COST_PER_QUERY.get(mode, 0.005)

            return SearchResponse(
                provider=self.NAME,
                search_mode=mode,
                query=request.query,
                results=results,
                latency_ms=latency_ms,
                cost_usd=total_cost,
                timestamp=datetime.utcnow(),
                metadata={
                    "results_requested": request.num_results,
                    "results_returned": len(results),
                },
            )

        except httpx.HTTPStatusError as e:
            from searchprobe.core.exceptions import ProviderError, RateLimitError

            latency_ms = (time.perf_counter() - start_time) * 1000
            if e.response.status_code == 429:
                raise RateLimitError(str(e), provider=self.NAME)
            return SearchResponse(
                provider=self.NAME,
                search_mode=mode,
                query=request.query,
                results=[],
                latency_ms=latency_ms,
                cost_usd=0.0,
                timestamp=datetime.utcnow(),
                error=str(e),
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return SearchResponse(
                provider=self.NAME,
                search_mode=mode,
                query=request.query,
                results=[],
                latency_ms=latency_ms,
                cost_usd=0.0,
                timestamp=datetime.utcnow(),
                error=str(e),
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
