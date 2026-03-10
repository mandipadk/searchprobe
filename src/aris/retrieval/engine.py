"""Retrieval engine -- orchestrates multi-modal retrieval and fusion."""

from __future__ import annotations

import asyncio
import logging

from aris.core.config import ArisConfig
from aris.core.models import Document, ScoredDocument
from aris.index.dense_store import DenseStore
from aris.index.sparse_store import SparseStore
from aris.index.structured_store import StructuredStore
from aris.que.models import StructuredQuery
from aris.retrieval.decomposer import decompose_query
from aris.retrieval.dense import DenseRetriever
from aris.retrieval.fusion import reciprocal_rank_fusion
from aris.retrieval.negation_filter import filter_negations
from aris.retrieval.sparse import SparseRetriever
from aris.retrieval.structured import StructuredRetriever
from aris.sources.base import DataSource

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Orchestrates multi-modal retrieval from sources and local index.

    Retrieval paths:
    1. External sources (DuckDuckGo, Brave, SerpAPI) -- always available
    2. Dense (ChromaDB embedding search) -- from local index
    3. Sparse (BM25) -- keyword-sensitive queries
    4. Structured (SQLite metadata) -- numeric/temporal/entity queries
    5. HyDE (hypothetical document embeddings) -- answer-quality retrieval
    """

    def __init__(
        self,
        config: ArisConfig,
        dense_store: DenseStore | None = None,
        sparse_store: SparseStore | None = None,
        structured_store: StructuredStore | None = None,
    ) -> None:
        self._config = config
        self._dense_retriever = DenseRetriever(dense_store) if dense_store else None
        self._sparse_retriever = SparseRetriever(sparse_store) if sparse_store else None
        self._structured_retriever = StructuredRetriever(structured_store) if structured_store else None

    async def retrieve(
        self,
        query: StructuredQuery,
        sources: list[DataSource],
        max_candidates: int | None = None,
    ) -> list[ScoredDocument]:
        """Run multi-modal retrieval and return fused candidates."""
        max_candidates = max_candidates or self._config.max_candidates
        search_query = query.positive_query or query.core_intent or query.original_query
        weights = query.retriever_weights

        result_lists: list[list[ScoredDocument]] = []
        retriever_weights: list[float] = []

        # 1. External sources (always try)
        external_results = await self._retrieve_external(search_query, sources, max_candidates)
        if external_results:
            result_lists.append(external_results)
            retriever_weights.append(weights.get("dense", 0.4))  # external gets dense weight

        # 2. Dense retrieval from index
        if self._dense_retriever:
            sub_queries = decompose_query(query) if query.use_decomposition else [search_query]
            for sq in sub_queries:
                dense_results = await self._dense_retriever.retrieve(sq, num_results=max_candidates)
                if dense_results:
                    result_lists.append(dense_results)
                    retriever_weights.append(weights.get("dense", 0.4) / max(len(sub_queries), 1))

        # 3. Sparse BM25 retrieval
        if self._sparse_retriever:
            sparse_results = await self._sparse_retriever.retrieve(search_query, num_results=max_candidates)
            if sparse_results:
                result_lists.append(sparse_results)
                retriever_weights.append(weights.get("sparse", 0.25))

        # 4. Structured retrieval
        if self._structured_retriever and query.constraints:
            struct_results = await self._structured_retriever.retrieve_by_constraints(
                query.constraints, num_results=max_candidates
            )
            if struct_results:
                result_lists.append(struct_results)
                retriever_weights.append(weights.get("structured", 0.2))

        # Fuse all result lists
        if not result_lists:
            return []

        fused = reciprocal_rank_fusion(result_lists, weights=retriever_weights, max_results=max_candidates)

        # Apply negation filter
        if query.negations:
            fused = filter_negations(fused, query.negations)

        logger.info(
            "Retrieved %d candidates from %d retrieval paths",
            len(fused), len(result_lists),
        )
        return fused

    async def _retrieve_external(
        self, query: str, sources: list[DataSource], max_per_source: int
    ) -> list[ScoredDocument]:
        """Retrieve from all external sources in parallel."""
        if not sources:
            return []

        tasks = [
            source.search(query, num_results=min(30, max_per_source))
            for source in sources
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_docs: list[Document] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Source %s failed: %s", sources[i].name, result)
                continue
            all_docs.extend(result)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_docs: list[Document] = []
        for doc in all_docs:
            if doc.url not in seen_urls:
                seen_urls.add(doc.url)
                unique_docs.append(doc)

        # Convert to ScoredDocuments with position-based scores
        return [
            ScoredDocument(
                document=doc,
                retrieval_score=max(0.1, 1.0 - (i / max(len(unique_docs), 1)) * 0.8),
                final_score=max(0.1, 1.0 - (i / max(len(unique_docs), 1)) * 0.8),
            )
            for i, doc in enumerate(unique_docs)
        ]
