"""ArisSearchProvider -- SearchProbe adapter that implements SearchProvider protocol.

This allows Aris to be benchmarked alongside Exa, Brave, Tavily, and SerpAPI
using SearchProbe's adversarial evaluation framework.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import ClassVar

from searchprobe.providers.base import SearchProvider
from searchprobe.providers.models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)

from aris.agent.search_agent import SearchAgent
from aris.core.config import ArisConfig
from aris.index.manager import IndexManager
from aris.que.engine import QueryUnderstandingEngine
from aris.ranking.engine import RankingEngine
from aris.retrieval.engine import RetrievalEngine
from aris.sources.registry import SourceRegistry
from aris.verification.engine import ConstraintVerificationEngine


class ArisSearchProvider(SearchProvider):
    """Aris as a SearchProbe-compatible search provider.

    Modes:
    - "full": All stages (QUE -> Retrieve -> Verify -> Rank -> Agent loop)
    - "dense_only": Embedding retrieval only (no verification or agent)
    - "no_verification": Skip constraint verification stage
    """

    NAME: ClassVar[str] = "aris"
    SUPPORTED_MODES: ClassVar[list[str]] = ["full", "dense_only", "no_verification"]
    COST_PER_QUERY: ClassVar[dict[str, float]] = {
        "full": 0.005,
        "dense_only": 0.001,
        "no_verification": 0.003,
    }

    def __init__(self, api_key: str = "") -> None:
        super().__init__(api_key=api_key)
        self._config = ArisConfig()
        self._registry = SourceRegistry(self._config)
        self._index_manager = IndexManager(self._config)

        self._que = QueryUnderstandingEngine(self._config)
        self._retrieval = RetrievalEngine(
            self._config,
            dense_store=self._index_manager.dense,
            sparse_store=self._index_manager.sparse,
            structured_store=self._index_manager.structured,
        )
        self._verification = ConstraintVerificationEngine()
        self._ranking = RankingEngine(self._config)
        self._agent = SearchAgent(
            self._config, self._que, self._retrieval, self._verification, self._ranking,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute search and return SearchProbe-compatible response."""
        start_time = time.perf_counter()
        mode = request.search_mode or "full"

        try:
            sources = self._registry.get_available()
            aris_response = await self._agent.search(
                query=request.query,
                sources=sources,
                num_results=request.num_results,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            results = []
            for i, r in enumerate(aris_response.results):
                content = None
                if request.include_content:
                    content = r.snippet[:request.max_content_chars]

                results.append(SearchResult(
                    title=r.title,
                    url=r.url,
                    snippet=r.snippet[:300],
                    content=content,
                    score=r.score,
                    published_date=r.published_date,
                    source_domain=SearchResult.extract_domain(r.url),
                    position=i,
                ))

            return SearchResponse(
                provider=self.NAME,
                search_mode=mode,
                query=request.query,
                results=results,
                latency_ms=latency_ms,
                cost_usd=self.COST_PER_QUERY.get(mode, 0.005),
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "iterations": aris_response.iterations,
                    "total_candidates": aris_response.total_candidates,
                    "strategy": aris_response.strategy_used,
                    "predicted_failure_modes": aris_response.predicted_failure_modes,
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
                timestamp=datetime.now(timezone.utc),
                error=str(e),
            )

    async def close(self) -> None:
        await self._registry.close_all()
        self._index_manager.close()
