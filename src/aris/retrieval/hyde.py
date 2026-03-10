"""Hypothetical Document Embeddings (HyDE) retrieval.

Generates a hypothetical ideal document using Claude, then embeds it
to find real documents with similar content.
"""

from __future__ import annotations

import logging

import anthropic

from aris.core.config import ArisConfig
from aris.core.models import ScoredDocument
from aris.index.dense_store import DenseStore

logger = logging.getLogger(__name__)

HYDE_PROMPT = """Write a short paragraph (100-150 words) that would be the ideal answer to this search query.
Write it as if it were an excerpt from a real document that perfectly answers the question.
Do not explain what you're doing, just write the document excerpt.

Query: {query}"""


class HyDERetriever:
    """Generates hypothetical documents and uses them for embedding retrieval."""

    def __init__(self, config: ArisConfig, dense_store: DenseStore) -> None:
        self._config = config
        self._dense_store = dense_store
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def retrieve(self, query: str, num_results: int = 50) -> list[ScoredDocument]:
        """Generate hypothetical document and retrieve similar real documents."""
        if self._dense_store.count == 0:
            return []

        try:
            # Generate hypothetical document
            response = await self._client.messages.create(
                model=self._config.parser_model,
                max_tokens=256,
                messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
            )
            hypothetical = response.content[0].text

            # Use hypothetical document as query for dense retrieval
            results = self._dense_store.query(hypothetical, n_results=num_results)
            return [
                ScoredDocument(
                    document=doc,
                    retrieval_score=score * 0.9,  # Slight discount for indirect retrieval
                    final_score=score * 0.9,
                )
                for doc, score in results
            ]
        except Exception as e:
            logger.warning("HyDE retrieval failed: %s", e)
            return []
