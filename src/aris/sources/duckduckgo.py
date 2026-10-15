"""DuckDuckGo search data source -- free, no API key needed."""

from __future__ import annotations

import logging

from aris.core.exceptions import SourceError
from aris.core.models import Document
from aris.sources.base import DataSource

logger = logging.getLogger(__name__)


class DuckDuckGoSource(DataSource):
    """DuckDuckGo search via duckduckgo-search library."""

    NAME = "duckduckgo"

    async def _fetch(self, query: str, num_results: int) -> list[Document]:
        import asyncio

        try:  # duckduckgo-search >= 7 only ships the synchronous DDGS client
            from duckduckgo_search import DDGS
        except ImportError as e:  # surfaced as a source error, not silently empty
            raise SourceError(f"duckduckgo-search is not installed: {e}", source=self.NAME) from e

        def _search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=num_results))

        raw_results = await asyncio.to_thread(_search)

        documents = []
        for r in raw_results:
            documents.append(
                Document(
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                    content=r.get("body", ""),
                    source=self.NAME,
                )
            )
        return documents
