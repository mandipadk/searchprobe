"""SerpAPI (Google) search provider implementation."""

import time
from datetime import datetime
from typing import Any, ClassVar
from urllib.parse import urlparse

from searchprobe.providers.base import SearchProvider
from searchprobe.providers.models import SearchRequest, SearchResponse, SearchResult


class SerpAPIProvider(SearchProvider):
    """SerpAPI provider for Google Search results.

    SerpAPI scrapes Google and returns structured JSON data.
    This gives us access to Google's search quality for comparison.
    """

    NAME: ClassVar[str] = "serpapi"
    SUPPORTED_MODES: ClassVar[list[str]] = ["google"]
    COST_PER_QUERY: ClassVar[dict[str, float]] = {
        "google": 0.01,  # SerpAPI is more expensive
    }

    def __init__(self, api_key: str) -> None:
        """Initialize SerpAPI client."""
        super().__init__(api_key)
        self._client = None

    @property
    def client(self):
        """Lazy-load SerpAPI client."""
        if self._client is None:
            from serpapi import GoogleSearch

            self._client_class = GoogleSearch
        return self._client_class

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute search using SerpAPI."""
        import asyncio

        start_time = time.perf_counter()
        mode = request.search_mode or "google"

        try:
            # Build search parameters
            params: dict[str, Any] = {
                "q": request.query,
                "num": min(request.num_results, 100),  # Google max is 100
                "api_key": self.api_key,
                "engine": "google",
            }

            # Execute search (SerpAPI client is synchronous)
            loop = asyncio.get_event_loop()

            def do_search():
                search = self.client(params)
                return search.get_dict()

            response = await loop.run_in_executor(None, do_search)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse results
            results = []
            organic_results = response.get("organic_results", [])

            for i, result in enumerate(organic_results):
                # Extract domain
                url = result.get("link", "")
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.replace("www.", "")

                # Google/SerpAPI doesn't provide relevance scores
                # Use position-based scoring
                total = len(organic_results)
                score = 1.0 - (i / max(total, 1)) * 0.6  # 1.0 to 0.4 range

                # Get snippet
                snippet = result.get("snippet", "")

                results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=url,
                        snippet=snippet,
                        content=snippet if request.include_content else None,
                        score=score,
                        published_date=None,
                        source_domain=domain,
                        position=i,
                        provider_raw=result,
                    )
                )

            # Calculate cost
            total_cost = self.COST_PER_QUERY.get(mode, 0.01)

            return SearchResponse(
                provider=self.NAME,
                search_mode=mode,
                query=request.query,
                results=results,
                latency_ms=latency_ms,
                cost_usd=total_cost,
                timestamp=datetime.utcnow(),
                metadata={
                    "engine": "google",
                    "results_requested": request.num_results,
                    "results_returned": len(results),
                    "search_information": response.get("search_information", {}),
                },
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
