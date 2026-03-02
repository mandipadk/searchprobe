"""Tavily search provider implementation."""

import time
from datetime import datetime
from typing import Any, ClassVar
from urllib.parse import urlparse

from searchprobe.providers.base import SearchProvider
from searchprobe.providers.models import SearchRequest, SearchResponse, SearchResult


class TavilyProvider(SearchProvider):
    """Tavily search provider.

    Tavily is a search API optimized for AI agents with built-in
    content extraction and safety features.
    """

    NAME: ClassVar[str] = "tavily"
    SUPPORTED_MODES: ClassVar[list[str]] = ["basic", "advanced"]
    COST_PER_QUERY: ClassVar[dict[str, float]] = {
        "basic": 0.005,
        "advanced": 0.01,
    }

    def __init__(self, api_key: str) -> None:
        """Initialize Tavily client."""
        super().__init__(api_key)
        self._client = None

    @property
    def client(self):
        """Lazy-load Tavily client."""
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute search using Tavily API."""
        import asyncio

        start_time = time.perf_counter()
        mode = request.search_mode or "advanced"

        try:
            # Build search parameters
            search_kwargs: dict[str, Any] = {
                "query": request.query,
                "max_results": request.num_results,
                "search_depth": mode,
            }

            # Add content if requested
            if request.include_content:
                search_kwargs["include_raw_content"] = True

            # Execute search (Tavily client is synchronous)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.search(**search_kwargs)
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Normalize results
            results = []
            for i, result in enumerate(response.get("results", [])):
                # Extract domain
                url = result.get("url", "")
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.replace("www.", "")

                # Get score
                score = result.get("score")
                if score is not None:
                    score = min(1.0, max(0.0, float(score)))

                # Get content
                content = None
                if request.include_content:
                    content = result.get("raw_content") or result.get("content", "")
                    if content and len(content) > request.max_content_chars:
                        content = content[: request.max_content_chars]

                results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=url,
                        snippet=result.get("content", "")[:500],
                        content=content,
                        score=score,
                        published_date=None,  # Tavily doesn't always provide this
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
                    "search_depth": mode,
                    "results_requested": request.num_results,
                    "results_returned": len(results),
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
