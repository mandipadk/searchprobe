"""Agentic search with progressive refinement loop."""

from __future__ import annotations

import logging
import time

from aris.agent.evaluator import ResultEvaluator
from aris.agent.reformulator import QueryReformulator
from aris.core.config import ArisConfig
from aris.core.models import ArisResponse, SearchResult, ScoredDocument
from aris.que.engine import QueryUnderstandingEngine
from aris.que.models import StructuredQuery
from aris.ranking.engine import RankingEngine
from aris.retrieval.engine import RetrievalEngine
from aris.sources.base import DataSource
from aris.verification.engine import ConstraintVerificationEngine

logger = logging.getLogger(__name__)


class SearchAgent:
    """Progressive refinement search agent.

    Runs the full pipeline (QUE -> Retrieve -> Verify -> Rank), evaluates
    result quality, and iterates if results don't meet criteria.
    """

    def __init__(
        self,
        config: ArisConfig,
        que: QueryUnderstandingEngine,
        retrieval: RetrievalEngine,
        verification: ConstraintVerificationEngine,
        ranking: RankingEngine,
    ) -> None:
        self._config = config
        self._que = que
        self._retrieval = retrieval
        self._verification = verification
        self._ranking = ranking
        self._evaluator = ResultEvaluator()
        self._reformulator = QueryReformulator()

    async def search(
        self,
        query: str,
        sources: list[DataSource],
        num_results: int | None = None,
    ) -> ArisResponse:
        """Execute a search with iterative refinement."""
        start = time.perf_counter()
        num_results = num_results or self._config.default_num_results
        max_iterations = self._config.max_iterations

        # Stage 1: Query Understanding
        structured = await self._que.understand(query)

        best_results: list[ScoredDocument] = []
        total_candidates = 0

        for iteration in range(1, max_iterations + 1):
            # Stage 2: Retrieve
            candidates = await self._retrieval.retrieve(structured, sources)
            total_candidates += len(candidates)

            if not candidates:
                if iteration < max_iterations:
                    structured = self._reformulator.reformulate(
                        structured, "No results", iteration
                    )
                    continue
                break

            # Stage 3: Verify constraints
            candidates = await self._verification.verify(structured, candidates)

            # Stage 4: Rank
            ranked = await self._ranking.rank(structured, candidates, num_results=num_results)

            # Stage 5: Evaluate quality
            is_sufficient, reason = self._evaluator.evaluate(structured, ranked)

            if is_sufficient or iteration == max_iterations:
                best_results = ranked
                break

            logger.info("Iteration %d: %s -- reformulating", iteration, reason)
            structured = self._reformulator.reformulate(structured, reason, iteration)

            # Keep best results so far
            if not best_results or (ranked and ranked[0].final_score > best_results[0].final_score):
                best_results = ranked

        latency_ms = (time.perf_counter() - start) * 1000

        return ArisResponse(
            query=query,
            results=[SearchResult.from_scored(doc) for doc in best_results],
            total_candidates=total_candidates,
            iterations=min(iteration, max_iterations),
            latency_ms=latency_ms,
            strategy_used=str(structured.retriever_weights),
            predicted_failure_modes=structured.predicted_failure_modes,
        )
