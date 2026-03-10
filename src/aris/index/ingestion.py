"""Document processing pipeline for ingestion into indices."""

from __future__ import annotations

import logging

from aris.core.config import ArisConfig
from aris.core.models import Document
from aris.index.content_extractor import extract_content
from aris.index.dense_store import DenseStore
from aris.index.document import IndexedDocument
from aris.index.sparse_store import SparseStore
from aris.index.structured_store import StructuredStore
from aris.sources.web import WebSource

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Fetches, extracts, and indexes documents across all index types."""

    def __init__(
        self,
        config: ArisConfig,
        dense: DenseStore,
        sparse: SparseStore,
        structured: StructuredStore,
    ) -> None:
        self._config = config
        self._dense = dense
        self._sparse = sparse
        self._structured = structured
        self._web = WebSource()

    async def ingest_urls(self, urls: list[str]) -> int:
        """Fetch URLs, extract content, and index into all stores."""
        documents = await self._web.fetch_urls(urls)
        return self.ingest_documents(documents)

    def ingest_documents(self, documents: list[Document]) -> int:
        """Index pre-fetched documents into all stores."""
        if not documents:
            return 0

        self._dense.add_documents(documents)
        self._sparse.add_documents(documents)
        self._structured.add_documents(documents)

        logger.info("Ingested %d documents into all indices", len(documents))
        return len(documents)

    async def close(self) -> None:
        await self._web.close()
