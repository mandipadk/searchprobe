"""Exa search provider implementation."""

import time
from datetime import UTC, datetime
from typing import Any, ClassVar
from urllib.parse import urlparse

from exa_py import Exa

from searchprobe.providers.base import SearchProvider, looks_rate_limited
from searchprobe.providers.models import SearchRequest, SearchResponse, SearchResult


class ExaProvider(SearchProvider):
    """Exa.ai search provider.

    Supports multiple search modes:
    - auto: Intelligently combines neural and keyword methods (default)
    - neural: Pure embeddings-based semantic search
    - fast: Optimized for speed, lower quality
    - deep: Multi-step agentic search with query expansion
    """

    NAME: ClassVar[str] = "exa"
    # Modes are passed to the API verbatim (exa-py ``SearchType``).
    SUPPORTED_MODES: ClassVar[list[str]] = [
        "auto",
        "neural",
        "fast",
        "instant",
        "deep-lite",
        "deep",
        "deep-reasoning",
    ]
    # Rough per-query estimates used only for budget reservation; check Exa's
    # pricing page for authoritative numbers. Deep variants run multi-step
    # searches and are priced conservatively high.
    COST_PER_QUERY: ClassVar[dict[str, float]] = {
        "auto": 0.005,
        "neural": 0.005,
        "fast": 0.005,
        "instant": 0.005,
        "deep-lite": 0.012,
        "deep": 0.015,
        "deep-reasoning": 0.025,
    }
    # Content extraction cost per page
    CONTENT_COST_PER_PAGE: ClassVar[float] = 0.001

    def estimate_cost(self, request: SearchRequest) -> float:
        """Search cost plus per-page content cost when content is requested."""
        cost = self.COST_PER_QUERY.get(request.search_mode or "auto", 0.005)
        if request.include_content:
            cost += request.num_results * self.CONTENT_COST_PER_PAGE
        return cost

    def __init__(self, api_key: str) -> None:
        """Initialize Exa client."""
        super().__init__(api_key)
        self.client = Exa(api_key=api_key)

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute search using Exa API.

        Args:
            request: Normalized search request

        Returns:
            Normalized search response
        """
        start_time = time.perf_counter()
        mode = request.search_mode or "auto"

        try:
            # Build search parameters
            search_kwargs: dict[str, Any] = {
                "query": request.query,
                "num_results": request.num_results,
                "type": mode,
            }

            # Add content retrieval if requested (exa-py v2+ uses 'contents')
            if request.include_content:
                from exa_py.api import ContentsOptions, TextContentsOptions

                search_kwargs["contents"] = ContentsOptions(
                    text=TextContentsOptions(max_characters=request.max_content_chars)
                )

            # Execute search (Exa SDK is synchronous, run in executor for async)
            import asyncio

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.client.search(**search_kwargs))

            # Calculate timing
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Normalize results
            results = []
            for i, result in enumerate(response.results):
                # Extract domain from URL
                parsed_url = urlparse(str(result.url))
                domain = parsed_url.netloc.replace("www.", "")

                # Parse published date if available
                published_date = None
                if hasattr(result, "published_date") and result.published_date:
                    try:
                        published_date = datetime.fromisoformat(
                            result.published_date.replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

                # Get score (may not be available in all modes)
                score = None
                if hasattr(result, "score") and result.score is not None:
                    # Exa scores are typically 0-1 already
                    score = min(1.0, max(0.0, float(result.score)))

                # Get content
                content = None
                if request.include_content and hasattr(result, "text"):
                    content = result.text

                results.append(
                    SearchResult(
                        title=result.title or "",
                        url=result.url,
                        snippet=result.text[:500] if result.text else "",
                        content=content,
                        score=score,
                        published_date=published_date,
                        source_domain=domain,
                        position=i,
                        provider_raw={"id": result.id} if hasattr(result, "id") else None,
                    )
                )

            # Calculate cost
            search_cost = self.COST_PER_QUERY.get(mode, 0.005)
            content_cost = (
                len(results) * self.CONTENT_COST_PER_PAGE if request.include_content else 0
            )
            total_cost = search_cost + content_cost

            return SearchResponse(
                provider=self.NAME,
                search_mode=mode,
                query=request.query,
                results=results,
                latency_ms=latency_ms,
                cost_usd=total_cost,
                timestamp=datetime.now(UTC),
                metadata={
                    "search_type": mode,
                    # For "auto", Exa reports whether it resolved to neural or keyword.
                    "resolved_search_type": getattr(response, "resolved_search_type", None),
                    "results_requested": request.num_results,
                    "results_returned": len(results),
                },
            )

        except Exception as e:
            from searchprobe.core.exceptions import RateLimitError

            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = str(e)

            # Detect rate limiting from error message
            if looks_rate_limited(error_msg):
                raise RateLimitError(error_msg, provider=self.NAME)

            return SearchResponse(
                provider=self.NAME,
                search_mode=mode,
                query=request.query,
                results=[],
                latency_ms=latency_ms,
                cost_usd=0.0,
                timestamp=datetime.now(UTC),
                error=error_msg,
            )
