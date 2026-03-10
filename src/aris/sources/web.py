"""Direct web crawling data source using httpx + trafilatura."""

from __future__ import annotations

import logging

import httpx

from aris.core.models import Document
from aris.sources.base import DataSource

logger = logging.getLogger(__name__)


class WebSource(DataSource):
    """Fetch and extract content from URLs directly."""

    NAME = "web"

    def __init__(self, max_retries: int = 2) -> None:
        super().__init__(max_retries=max_retries)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "Aris/0.1 (search engine)"},
            )
        return self._client

    async def fetch_url(self, url: str) -> Document | None:
        """Fetch and extract content from a single URL."""
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            html = response.text

            try:
                import trafilatura
                content = trafilatura.extract(html) or ""
                metadata = trafilatura.extract(html, output_format="json", include_links=False)
            except ImportError:
                content = html[:5000]
                metadata = None

            title = ""
            if metadata:
                import json
                try:
                    meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                    title = meta_dict.get("title", "")
                except (json.JSONDecodeError, AttributeError):
                    pass

            if not title:
                # Try extracting title from HTML
                start = html.find("<title>")
                end = html.find("</title>")
                if start != -1 and end != -1:
                    title = html[start + 7:end].strip()

            return Document(
                url=url,
                title=title,
                content=content,
                snippet=content[:300] if content else "",
                source=self.NAME,
            )
        except Exception as e:
            logger.debug("Failed to fetch %s: %s", url, e)
            return None

    async def _fetch(self, query: str, num_results: int) -> list[Document]:
        # WebSource is not a search engine -- it fetches known URLs.
        # Used by the ingestion pipeline, not for query-time retrieval.
        return []

    async def fetch_urls(self, urls: list[str]) -> list[Document]:
        """Fetch and extract content from multiple URLs."""
        import asyncio
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, Document)]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
