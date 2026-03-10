"""DuckDuckGo search data source -- free, no API key needed."""

from __future__ import annotations

import logging

from aris.core.models import Document
from aris.sources.base import DataSource

logger = logging.getLogger(__name__)


class DuckDuckGoSource(DataSource):
    """DuckDuckGo search via duckduckgo-search library."""

    NAME = "duckduckgo"

    async def _fetch(self, query: str, num_results: int) -> list[Document]:
        from duckduckgo_search import AsyncDDGS

        async with AsyncDDGS() as ddgs:
            raw_results = await ddgs.atext(query, max_results=num_results)

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
